//===------ ft.cpp - Common MPI fault tolerance implementations----------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of heartbeat and functions related to
// fault tolerance.
//
// Heartbeat algorithm is based on the paper:
// Bosilca, George, et al. "A failure detector for HPC platforms." The
// International Journal of High Performance Computing Applications 32.1 (2018):
// 139-158.
//
//===----------------------------------------------------------------------===//

#include "ft.h"

#include <cassert>
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <random>
#include <sstream>

// Debug macros
#define FTDEBUG(...) { }

#define assertm(expr, msg) assert(((void)msg, expr));

// Static definitions for fault tolerance
// ===----------------------------------------------------------------------===
static ft::FaultTolerance *ft_handler = nullptr;
static bool disable_ft = false;
static bool using_veloc = false;
// TODO: Since asserts are temporary, the disabling option will be removed when
// checkpointing is integrated
static bool disable_wrappers_asserts = false;

namespace ft {

/// Triggers whenever a MPI error occurs
void errorHandlerFunction(MPI_Comm *pcomm, int *perr, ...) {
  int len, ec;
  char errstr[MPI_MAX_ERROR_STRING];
  MPI_Error_string(*perr, errstr, &len);
  MPI_Error_class(*perr, &ec);
  FTDEBUG("Unhandled error!\n");
  FTDEBUG("Error: %s\n", errstr);
  FTDEBUG("Errorclass: %d\n", ec);
}

/// Init with \p global_comm as main comm, \p hb_time as heartbeat period and \p
/// susp_time as heartbeat suspect time
FaultTolerance::FaultTolerance(int hb_time, int susp_time,
                               MPI_Comm global_comm) {
  disable_ft = false;
  if (char *env_str = getenv("OMPCLUSTER_FT_DISABLE")) {
    if (std::stoi(env_str) == 1)
      disable_ft = true;
    if (disable_ft) {
      FTDEBUG("Disabling Fault Tolerance feature\n");
      return;
    }
  }

  main_comm = global_comm;

  MPI_Comm_dup(main_comm, &hb_comm);
  MPI_Comm_rank(hb_comm, &rank);
  MPI_Comm_size(hb_comm, &size);
  MPI_Comm_size(main_comm, &app_size);

  ft_handler = this;

  eta = hb_time;
  delta = susp_time / DEFAULT_HB_TIME_STEP;
  time_step = DEFAULT_HB_TIME_STEP;
  hb_done = false;

  setErrorHandling();

  // Start heartbeat only if there are more than one process
  if (size > 1) {
    hb_started_mutex.lock();
    hb_started = false;
    hb_started_mutex.unlock();
    heartbeat = std::thread(&FaultTolerance::hbMain, this);
    // Wait for HB to be completely started
    std::unique_lock<std::mutex> lock(hb_started_mutex);
    hb_started_cv.wait(lock, [&] { return hb_started; });
    hb_started_mutex.unlock();
  } else {
    // In case of not using HB, start structures so wrappers can use
    c_state = ft::CommState::VALID;
    if (size > 0) {
      p_states = new std::atomic<ProcessState>[size];
      p_states[0] = ft::ProcessState::ALIVE; // there is only one process
    }
  }
}

/// Finishes the main loop of heartbeat thread and synchronize with main thread
FaultTolerance::~FaultTolerance() {
  if (disable_ft)
    return;

  hb_done = true;
  if (heartbeat.joinable() == true) {
    // After receiving the `done` flag, the heartbeat threads would still wait 
    // for another HB period before stopping. Asking for cancellation to the HB
    // threads avoids the waiting period and terminates the program faster.
    pthread_cancel(heartbeat.native_handle());
    heartbeat.join();
    FTDEBUG("[Rank %d FT] Heartbeat thread joined\n", rank);
  }   
  if (alive.joinable() == true) {
    pthread_cancel(alive.native_handle());
    alive.join();
    FTDEBUG("[Rank %d FT] Alive thread joined\n", rank);
  } 
}

// MPI related functions
// ===----------------------------------------------------------------------===
/// Non-blocking MPI wrapper for probe and receive specific message
template <typename T, typename S, typename R>
int FaultTolerance::MPIw_IProbeRecv(T *buffer, int size, S type, int source,
                                    int tag, MPI_Comm comm, R status) {
  msg_recv_flag = 0;
  MPI_Improbe(source, tag, comm, &msg_recv_flag, &msg_recv, MPI_STATUS_IGNORE);
  if (msg_recv_flag) {
    MPI_Mrecv(buffer, size, type, &msg_recv, status);
    return 0;
  }
  return 1;
}

// Heartbeat related functions
// ===----------------------------------------------------------------------===
/// Configures all necessary variables to start the heartbeat
void FaultTolerance::hbInit() {
  hb_need_repair = false;
  c_state = CommState::VALID;

  // Check if there are FT environment variables
  if (char *env_str = getenv("OMPCLUSTER_HB_TIMESTEP")) {
    time_step = std::stoi(env_str);
    FTDEBUG(
        "[Rank %d FT] Using HB timestep defined in environment variables: %d\n",
        rank, time_step);
  }
  // The timeout represents the suspect time, must be multiple of TIMESTEP
  if (char *env_str = getenv("OMPCLUSTER_HB_TIMEOUT")) {
    assertm((std::stoi(env_str) % time_step) == 0,
            "Timeout must be multiple of Timestep.");
    delta = std::stoi(env_str) / time_step;
    FTDEBUG(
        "[Rank %d FT] Using HB timeout defined in environment variables: %d\n",
        rank, std::stoi(env_str));
  }
  // The period represents heartbeat period, must be multiple of TIMESTEP
  if (char *env_str = getenv("OMPCLUSTER_HB_PERIOD")) {
    assertm((std::stoi(env_str) % time_step) == 0,
            "Period must be multiple of Timestep.");
    eta = std::stoi(env_str) / time_step;
    FTDEBUG(
        "[Rank %d FT] Using HB period defined in environment variables: %d\n",
        rank, std::stoi(env_str));
  }

  // Start heartbeat period and suspect time counters
  delta_to = delta;
  eta_to = eta;

  // Set all nodes as neighbors alive state
  p_states = new std::atomic<ProcessState>[size];
  for (int i = 0; i < size; i++) {
    neighbors.push_back(i);
    p_states[i] = ProcessState::ALIVE;
  }

  // Shuffle and find emitter and observer
  shuffle(neighbors.begin(), neighbors.end(),
          std::default_random_engine(size * size));
  for (std::size_t i = 0; i < neighbors.size(); i++) {
    if (rank == neighbors[i]) {
      n_pos = i;
      emitter = neighbors[(i == 0) ? size - 1 : i - 1];
      observer = neighbors[(i + 1) % size];
      break;
    }
  }
  // Extend delta_to 5 times if the rank observed is 0, since rank 0 has no
  // recovery procedures.
  if (emitter == 0)
    delta_to *= 5;
  FTDEBUG(
      "[Rank %d FT] Heartbeat initialized with emitter %d and observer %d\n",
      rank, emitter, observer);

  // Notifies the constructor the heartbeat finished starting
  std::lock_guard<std::mutex> lock(hb_started_mutex);
  hb_started = true;
  hb_started_cv.notify_all();
}

/// Heartbeat thread main loop function
void FaultTolerance::hbMain() {
  thread_id = std::this_thread::get_id();
  FTDEBUG("[Rank %d FT] Started rank %d of %d heartbeat thread\n", rank, rank,
          size - 1);
  hbInit();
  int flag;
  int message[3];
  MPI_Status status;
  alive = std::thread(&FaultTolerance::hbSendAlive, this);
  while (!hb_done) {
    if (hb_need_repair) {
      // Waits for the user thread notifies that repair is complete
      std::unique_lock<std::mutex> lock(hb_need_repair_mutex);
      hb_need_repair_cv.wait(lock, [&] { return !hb_need_repair; });
      hb_need_repair_mutex.unlock();
    }

    // If received an alive message from emitter, reset the suspect timeout
    flag = MPIw_IProbeRecv(message, 3, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                           hb_comm, &status);

    if (!flag) {
      int tag = status.MPI_TAG;
      switch (tag) {
      case TAG_HB_ALIVE: {
        if (message[0] == emitter) {
          hbResetObsTimeout();
        } else {
          // If sender isn't in neighbors, means that is a false positive
          auto known = std::find(neighbors.begin(), neighbors.end(), message[0]);
          if (known == neighbors.end()) {
            // Gives extra time before suspecting the emitter
            delta_to = 2 * delta;
            // Resets current emitter observer
            int new_obs[3] = {message[0], -1, -1};
            MPI_Isend(new_obs, 3, MPI_INT, emitter, TAG_HB_NEWOBS, hb_comm,
                      &send_request);
            MPI_Request_free(&send_request);
            // Resets current emitter to the old one
            emitter = message[0];
            // Broadcast the inclusion of the false positive
            hbBroadcast(HB_BC_ALIVE, emitter, n_pos);
          }
        }
        FTDEBUG("[Rank %d FT] Received alive message from %d\n", rank, message[0]);
      } break;
      case TAG_HB_NEWOBS: {
        hbSetNewObs(message[0]);
        FTDEBUG("[Rank %d FT] %d is the new observer\n", rank, message[0]);
      } break;
      case TAG_HB_BCAST: {
        auto valid_source =
            std::find(neighbors.begin(), neighbors.end(), status.MPI_SOURCE);
        auto unknown =
            std::find(neighbors.begin(), neighbors.end(), message[1]);
        switch (message[0]) {
        case HB_BC_FAILURE:
          // Only replicates if failure is unknown and from valid source
          if (unknown != neighbors.end() && valid_source != neighbors.end()) {
            hbBroadcast(HB_BC_FAILURE, message[1], message[2]);
            FTDEBUG("[Rank %d FT] Received broadcast with failed node: %d\n",
                    rank, message[1]);
          }
          break;
        case HB_BC_ALIVE:
          // Only replicates if process is not added in neighbors yet
          if (unknown == neighbors.end() && valid_source != neighbors.end()) {
            hbBroadcast(HB_BC_ALIVE, message[1], message[2]);
            FTDEBUG("[Rank %d FT] Received broadcast of false positive: %d\n",
                    rank, message[1]);
          }
          break;
        case HB_BC_REPAIR:
          if (c_state == CommState::INVALID &&
              valid_source != neighbors.end()) {
            if (hb_need_repair == false) {
              hbBroadcast(HB_BC_REPAIR, message[1], message[2]);
              FTDEBUG(
                  "[Rank %d FT] Received broadcast of repair operation: %d\n",
                  rank, message);
            }
          }
          break;
        default:
          FTDEBUG("[Rank %d FT] Ignoring unknown broadcast\n", rank);
          break;
        }
      } break;
      }
    }

    if (delta_to <= 0)
      hbFindDeadNode();
    else
      delta_to--;

    // Heartbeat thread time step
    std::this_thread::sleep_for(std::chrono::milliseconds(time_step));
  }
} // namespace ft

/// Sends a message to the observer saying it is alive
void FaultTolerance::hbSendAlive() {
  while (!hb_done) {
    int alive[3] = {rank, -1, -1};
    MPI_Isend(alive, 3, MPI_INT, observer, TAG_HB_ALIVE, hb_comm,
              &send_request);
    MPI_Request_free(&send_request);
    std::this_thread::sleep_for(std::chrono::milliseconds(eta));
  }
}

/// Resets suspect time upon receiving alive messages from emitter
void FaultTolerance::hbResetObsTimeout() {
  delta_to = delta;
  // Extend delta_to 5 times if the rank observed is 0, since rank 0 has no
  // recovery procedures.
  if (emitter == 0)
    delta_to *= 5;
}

/// Do all the necessary procedures after noticing the emitter failed
void FaultTolerance::hbFindDeadNode() {
  // Gives extra time before suspecting the emitter
  delta_to = 2 * delta;

  assertm(emitter != 0, "Fatal Error: Head process failed");
  FTDEBUG("[Rank %d FT] Found a failed process: %d\n", rank, emitter);

  // Broadcast failure to other nodes
  hbBroadcast(HB_BC_FAILURE, emitter, -1);

  // Find a new emitter and send a message saying its new observer
  emitter = hbFindEmitter();
  int new_obs[3] = {rank, -1, -1};
  MPI_Isend(new_obs, 3, MPI_INT, emitter, TAG_HB_NEWOBS, hb_comm,
            &send_request);
  MPI_Request_free(&send_request);
  // Extend delta_to 5 times if the emitter is 0, since rank 0 has no recovery
  // procedures.
  if (emitter == 0)
    delta_to *= 5;
  FTDEBUG("[Rank %d FT] New emitter is: %d\n", rank, emitter);
}

/// Look backwards in the ring to find a new emitter
int FaultTolerance::hbFindEmitter() {
  // Since failed node was already removed from neighbors
  int new_emitter = (n_pos == 0) ? size - 1 : n_pos - 1;
  return neighbors[new_emitter];
}

/// Set the new observer after the previous one failed
void FaultTolerance::hbSetNewObs(int new_obs) {
  observer = new_obs;
  // Sends an alive message to new observer in the next heartbeat iteration
  eta_to = 0;
  FTDEBUG("[Rank %d FT] New observer is: %d\n", rank, observer);
}

/// Internal broadcast for heartbeat
void FaultTolerance::hbBroadcast(int type, int value, int pos) {
  int message[3] = {type, value, pos};

  // Does the broadcast
  for (std::size_t i = 1; i < neighbors.size(); i *= 2) {
    int index = (n_pos + i) % neighbors.size();
    if (neighbors[index] != rank) {
      if (type != HB_BC_FAILURE || neighbors[index] != value) {
        MPI_Isend(message, 3, MPI_INT, neighbors[index], TAG_HB_BCAST,
                  hb_comm, &send_request);
        MPI_Request_free(&send_request);
      }
    }
  }

  switch (type) {
  case HB_BC_FAILURE: {
    // Updates states of the process and MPI communicators
    c_state = CommState::INVALID;
    p_states[value] = ProcessState::DEAD;
    for (const auto &callback : notify_callbacks)
      callback({FTNotificationID::FAILURE, value});

    // Removes failed node from neighbors
    auto d_pos = std::find(neighbors.begin(), neighbors.end(), value);
    if (d_pos != neighbors.end()) {
      neighbors.erase(d_pos);
      // Update current position if necessary
      if (d_pos < neighbors.begin() + n_pos)
        n_pos--;
    }
    size = neighbors.size();
    FTDEBUG("[Rank %d FT] Broadcasting failure\n", rank);
  } break;
  case HB_BC_ALIVE: {
    p_states[value] = ProcessState::ALIVE;
    // Validate comm again if the false positve was the only dead process
    c_state = CommState::VALID;
    for (int i = 0; i < size; i++) {
      if (p_states[i] == ProcessState::DEAD) {
        c_state = CommState::INVALID;
        break;
      }
    }
    for (const auto &callback : notify_callbacks)
      callback({FTNotificationID::FALSE_POSITIVE, value});

    // Adds false positive to neighbors again and fix current position
    neighbors.insert(neighbors.begin() + pos, value);
    if (pos <= n_pos)
      n_pos++;
    size = neighbors.size();
    FTDEBUG("[Rank %d FT] Broadcasting false positive\n", rank);
  } break;
  case HB_BC_REPAIR: {
    int sum = 0;
    for (std::size_t i = 0; i < neighbors.size(); i += 2)
      sum += neighbors[i] * ((i < neighbors.size()) ? neighbors[i + 1] : 1);

    // Sum and size must be the same as received in broadcast
    assertm((sum == value && (int)neighbors.size() == pos),
            "Inconsistency between alive groups, can't repair comm.\n");

    // This notifies the requestCommRepair waiting cv
    std::lock_guard<std::mutex> lock(hb_need_repair_mutex);
    hb_need_repair = true;
    hb_need_repair_cv.notify_all();

    FTDEBUG("[Rank %d FT] Broadcasting repair operation\n", rank);
  } break;
  default: {
    FTDEBUG("[Rank %d FT] Ignoring unknown broadcast\n", rank);
  } break;
  }
}

// General fault tolerance functions
// ===----------------------------------------------------------------------===
/// Configure error handling for main and heartbeat communicators
void FaultTolerance::setErrorHandling() {
  MPI_Comm_create_errhandler(errorHandlerFunction, &main_error_handler);
  MPI_Comm_set_errhandler(main_comm, main_error_handler);
  MPI_Comm_create_errhandler(errorHandlerFunction, &hb_error_handler);
  MPI_Comm_set_errhandler(hb_comm, hb_error_handler);
}

/// Execute all procedures to make a new communicator with alive processes
void FaultTolerance::commRepair() {
  FTDEBUG("[Rank %d FT] Starting communicator repair process\n", rank);

  MPI_Group main_comm_group;
  MPI_Comm_group(main_comm, &main_comm_group);

  FTDEBUG("[Rank %d FT] Creating group with alive processes\n", rank);
  int n_size = neighbors.size();
  std::sort(neighbors.begin(), neighbors.end());
  int *alive_ranks = neighbors.data();

  MPI_Group alive_group;
  MPI_Group_incl(main_comm_group, n_size, (const int *)alive_ranks,
                 &alive_group);

  // Create a new Comm. Point of sync
  FTDEBUG("[Rank %d FT] Creating new communicator\n", rank);
  MPI_Comm new_comm;
  MPI_Comm_create_group(main_comm, alive_group, 0, &new_comm);

  assertm(new_comm != MPI_COMM_NULL, "Created invalid communicator");
  FTDEBUG("[Rank %d FT] Created new communicator\n", rank);

  int l_rank;
  MPI_Comm_rank(new_comm, &l_rank);
  MPI_Comm_size(new_comm, &size);
  FTDEBUG("[Rank %d FT] Is now rank %d of %d.\n", rank, l_rank, size);
  rank = l_rank;

  main_comm = new_comm;
  MPI_Comm_dup(main_comm, &hb_comm);
  setErrorHandling();

  c_state = CommState::VALID;
  delete p_states;
  neighbors.clear();
  hbInit();

  FTDEBUG("[Rank %d FT] Finished repair\n", rank);
  MPI_Barrier(hb_comm);
}

// User level functions
// ===----------------------------------------------------------------------===
/// Return current state of process with rank equals to \p id
ProcessState FaultTolerance::getProcessState(int id) {
  if (disable_ft)
    return ProcessState::DEAD;
  return p_states[id];
}

/// Return current state of MPI communicators
CommState FaultTolerance::getCommState() {
  if (disable_ft)
    return CommState::INVALID;
  return c_state;
}

/// Request by main thread to repair communicator
MPI_Comm FaultTolerance::requestCommRepair() {
  if (disable_ft)
    return MPI_COMM_NULL;
  if (c_state == CommState::INVALID) {
    if (rank == 0) {
      // Value and pos (BC parameters) are used to build an agree function
      int sum = 0;
      for (std::size_t i = 0; i < neighbors.size(); i += 2)
        sum += neighbors[i] * ((i < neighbors.size()) ? neighbors[i + 1] : 1);

      // Starts a request to se if all processes agree
      hbBroadcast(HB_BC_REPAIR, sum, neighbors.size());
    }
    // This waits for heartbeat thread to verify if comm repair is possible
    std::unique_lock<std::mutex> w_lock(hb_need_repair_mutex);
    hb_need_repair_cv.wait(w_lock, [&] { return hb_need_repair; });
    hb_need_repair_mutex.unlock();

    // Start repair, the HB thread can't repair since it could repair before
    // the main thread enters this function
    commRepair();

    // Free the HB thread - This notifies the heartbeat thread there is waiting
    std::lock_guard<std::mutex> n_lock(hb_need_repair_mutex);
    hb_need_repair = false;
    hb_need_repair_cv.notify_all();
  }

  return main_comm;
}

/// Public gather function that returns MPI main communicator
MPI_Comm FaultTolerance::getMainComm() {
  if (disable_ft)
    return MPI_COMM_NULL;
  // Update application size
  MPI_Comm_size(main_comm, &app_size);
  return main_comm;
}

/// Register an notification callback
void FaultTolerance::registerNotifyCallback(
    std::function<void(FTNotification)> callback) {
  if (disable_ft)
    return;
  notify_callbacks.push_back(callback);
}

// Test purpose functions
// ===----------------------------------------------------------------------===
/// Test purpose function that returns the heartbeat thread id
uint64_t FaultTolerance::getID() {
  std::stringstream ss;
  ss << thread_id;
  uint64_t id = std::stoull(ss.str());
  return id;
}

/// Test purpose function that returns the heartbeat current emitter
int FaultTolerance::getEmitter() { return emitter; }

// Wrappers helping functions
// ===----------------------------------------------------------------------===
/// Returns the current MPI rank
int FaultTolerance::getRank() {
  if (disable_ft) {
    return -1;
  }
  return rank;
}

/// Returns the current size the application have. In case of commRepair shrink,
/// this size only will be updated if application calls getMainComm()
int FaultTolerance::getSize() {
  if (disable_ft) {
    return -1;
  }
  return app_size;
}

} // namespace ft
