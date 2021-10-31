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
#include <cstdio>
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <random>
#include <sstream>
#include <unistd.h>

#include <nng/nng.h>
#include <nng/protocol/pair1/pair.h>

// Debug macros
#define FTDEBUG(...) printf(__VA_ARGS__);

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

/// Init with \p global_comm as main comm, \p hb_time as heartbeat period and \p
/// susp_time as heartbeat suspect time
FaultTolerance::FaultTolerance(int hb_time, int susp_time, int my_rank, int total_size) {
  disable_ft = false;
  if (char *env_str = getenv("OMPCLUSTER_FT_DISABLE")) {
    if (std::stoi(env_str) == 1)
      disable_ft = true;
    if (disable_ft) {
      FTDEBUG("Disabling Fault Tolerance feature\n");
      return;
    }
  }

  rank = my_rank;
  size = total_size;
  app_size = total_size;

  sockets.resize(size);
  urls.resize(size);
  open_sockets.resize(size);

  eta = hb_time / DEFAULT_HB_TIME_STEP;
  delta = susp_time / DEFAULT_HB_TIME_STEP;
  time_step = DEFAULT_HB_TIME_STEP;
  hb_done = false;

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
    // After receiving the `done` flag, the heartbeat threads would still wait 
    // for another HB period before stopping. Asking for cancellation to the HB
    // threads avoids the waiting period and terminates the program faster.
    pthread_cancel(alive.native_handle());
    alive.join();
    FTDEBUG("[Rank %d FT] Alive thread joined\n", rank);
  } 
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
  delta_to = 2 * delta; // Start double the time to avoid startup failures
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

  // Start sockets (a socke between each pair of processes)
  int base_port = 52000;
  for (int i = 0; i < size; i++) {
    int port = i;
    if (i != rank) { // No self Sockets
      nng_pair_open(&sockets[i]);
      open_sockets[i] = true;
      // Higher ranks as dialers in the socket
      if (rank < i) {
        int result;
        int port = base_port + rank;
        urls[i] =
            "ws://localhost:" + std::to_string(port) + "/" + std::to_string(i);
        do {
          result = nng_listen(sockets[i], urls[i].c_str(), NULL, 0);
          FTDEBUG("[Rank %d] result of listen %d (%s)\n", rank, result,
                  urls[i].c_str());
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
        } while (result != 0);
      } else {
        int result;
        int port = base_port + i;
        urls[i] = "ws://localhost:" + std::to_string(port) + "/" +
                  std::to_string(rank);
        do {
          result = nng_dial(sockets[i], urls[i].c_str(), NULL, 0);
          FTDEBUG("[Rank %d] result of dial %d (%s)\n", rank, result,
                  urls[i].c_str());
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
        } while (result != 0);
      }
    } else {
      open_sockets[i] = false;
    }
  }

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
  int message[4]; // [Type, info, info, info] - Type == MPI_TAGS
  alive = std::thread(&FaultTolerance::hbSendAlive, this);
  while (!hb_done) {
    // Did we receive any message
    for (int sender = 0; sender < app_size; sender++) {
      if (open_sockets[sender]) {
        size_t size;
        int result =
            nng_recv(sockets[sender], message, &size, NNG_FLAG_NONBLOCK);
        if (result == NNG_ECLOSED) {
          open_sockets[sender] = false;
          nng_close(sockets[sender]);
        }
        if (result == 0) {
          // Received a message, what type?
          switch (message[0]) {
          case TAG_HB_ALIVE: {
            if (message[1] == emitter) {
              hbResetObsTimeout();
            } else {
              // If sender isn't in neighbors, means that is a false positive
              auto known =
                  std::find(neighbors.begin(), neighbors.end(), message[1]);
              if (known == neighbors.end()) {
                // Gives extra time before suspecting the emitter
                delta_to = 2 * delta;
                // Resets current emitter observer
                int new_obs[4] = {TAG_HB_NEWOBS, message[1], -1, -1};
                nng_send(sockets[emitter], new_obs, 4 * sizeof(int),
                         NNG_FLAG_NONBLOCK);
                // Resets current emitter to the old one
                emitter = message[1];
                // Broadcast the inclusion of the false positive
                hbBroadcast(HB_BC_ALIVE, emitter, n_pos);
              }
            }
            FTDEBUG("[Rank %d FT] Received alive message from %d\n", rank,
                    message[1]);
          } break;
          case TAG_HB_NEWOBS: {
            hbSetNewObs(message[1]);
            FTDEBUG("[Rank %d FT] %d is the new observer\n", rank, message[1]);
          } break;
          case TAG_HB_BCAST: {
            auto valid_source =
                std::find(neighbors.begin(), neighbors.end(), sender);
            auto unknown =
                std::find(neighbors.begin(), neighbors.end(), message[2]);
            switch (message[1]) {
            case HB_BC_FAILURE:
              // Only replicates if failure is unknown and from valid source
              if (unknown != neighbors.end() &&
                  valid_source != neighbors.end()) {
                hbBroadcast(HB_BC_FAILURE, message[2], message[3]);
                FTDEBUG(
                    "[Rank %d FT] Received broadcast with failed node: %d\n",
                    rank, message[2]);
              }
              break;
            case HB_BC_ALIVE:
              // Only replicates if process is not added in neighbors yet
              if (unknown == neighbors.end() &&
                  valid_source != neighbors.end()) {
                hbBroadcast(HB_BC_ALIVE, message[2], message[3]);
                FTDEBUG(
                    "[Rank %d FT] Received broadcast of false positive: %d\n",
                    rank, message[2]);
              }
              break;
            default:
              FTDEBUG("[Rank %d FT] Ignoring unknown broadcast\n", rank);
              break;
            }
          } break;
          }
        }
      }
    }

    // If no alive messages were received before suspect time, it is a failure
    if (delta_to <= 0)
      hbFindDeadNode();
    else
      delta_to--;

    // Heartbeat thread time step
    std::this_thread::sleep_for(std::chrono::milliseconds(time_step));
  }
}

/// Sends a message to the observer saying it is alive
void FaultTolerance::hbSendAlive() {
  while(!hb_done) {
    if (eta_to <= 0) {
      eta_to = eta;
      int alive[4] = {TAG_HB_ALIVE, rank, -1, -1};
      nng_send(sockets[observer], alive, 4*sizeof(int), NNG_FLAG_NONBLOCK);
    } else {
      eta_to--;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(time_step));
  }
}

/// Resets suspect time upon receiving alive messages from emitter
void FaultTolerance::hbResetObsTimeout() {
  delta_to = delta;
}

/// Do all the necessary procedures after noticing the emitter failed
void FaultTolerance::hbFindDeadNode() {
  // Gives extra time before suspecting the emitter
  delta_to = 2 * delta;

  FTDEBUG("[Rank %d FT] Found a failed process: %d\n", rank, emitter);

  // Broadcast failure to other nodes
  hbBroadcast(HB_BC_FAILURE, emitter, -1);
  nng_close(sockets[emitter]);

  // Find a new emitter and send a message saying its new observer
  emitter = hbFindEmitter();
  int new_obs[4] = {TAG_HB_NEWOBS, rank, -1, -1};
  nng_send(sockets[emitter], new_obs, 4*sizeof(int), NNG_FLAG_NONBLOCK);
 
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
  int bc_message[4] = {TAG_HB_BCAST, type, value, pos};

  // Does the broadcast
  for (std::size_t i = 1; i < neighbors.size(); i *= 2) {
    int index = (n_pos + i) % neighbors.size();
    if (neighbors[index] != rank) {
      if (type != HB_BC_FAILURE || neighbors[index] != value) {
        nng_send(sockets[neighbors[index]], bc_message, 4 * sizeof(int),
                 NNG_FLAG_NONBLOCK);
      }
    }
  }

  switch (type) {
  case HB_BC_FAILURE: {
    // Updates states of the process and MPI communicators
    c_state = CommState::INVALID;
    p_states[value] = ProcessState::DEAD;
    open_sockets[value] = false;
    nng_close(sockets[value]);
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
  default: {
    FTDEBUG("[Rank %d FT] Ignoring unknown broadcast\n", rank);
  } break;
  }
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

} // namespace ft
