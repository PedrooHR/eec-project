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

#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <random>
#include <sstream>

#include <stdio.h>

#if COMPILE_WITH_VELOC
#include "veloc.h"
#endif

// Debug macros
#define FTDEBUG(...) { }

#define assertm(expr, msg) assert(((void)msg, expr));

// Static definitions for fault tolerance
// ===----------------------------------------------------------------------===
static FaultTolerance *ft_handler = nullptr;
static bool disable_ft = false;
static bool using_veloc = false;
// TODO: Since asserts are temporary, the disabling option will be removed when
// checkpointing is integrated
static bool disable_wrappers = false;
static bool disable_wrappers_asserts = false;
// Failure injection control
static bool inject_failure = false;

// MPI wrappers
// These wrappers use --wrap compiler linkage option that redefines the MPI
// functions, calling a MPI function that was wrapped execute this version
// instead of MPI distribution version. Calling __real_MPI_...() will call the
// original function
// Asserts in case of result equals to FT_ERROR are temporary
// ===----------------------------------------------------------------------===
/// MPI_Iwait replaces the MPI_Wait functions, as well as the regular MPI_Wait,
/// it waits until the /p request is complete. The additional parameter /p proc
/// represents the other side of the communication and is used to verify FT
/// parameters, it can be the rank of a process, or comm size if it is a
/// collective operation request. If /p proc equals to FT_WAIT_LEGACY, it
/// will be executing regular MPI_Wait.
int MPI_Iwait(MPI_Request *request, MPI_Status *status, int proc) {
  if (disable_ft || ft_handler == nullptr || proc == FT_WAIT_LEGACY || disable_wrappers) {
    return __real_MPI_Wait(request, status);
  }
  assertm((proc >= FT_MPI_COLLECTIVE) && (proc < ft_handler->getSize()),
          "Waiting for request with invalid rank participating.");
  int test_flag = 0;
  while (!test_flag) {
    __real_MPI_Test(request, &test_flag, status);
    // Since some MPI distribution have some error related to failed processes
    // we always check if the other communication part failed
    if (proc == FT_MPI_COLLECTIVE) {
      // If it is a collective call
      if (ft_handler->getCommState() != CommState::VALID) {
        MPI_Request_free(request);
        if (!disable_wrappers_asserts)
          assertm(false, "MPI_Iwait could not complete a collective call");
        return FT_ERROR;
      }
    } else {
      // If it is a point-to-point call
      if (ft_handler->getProcessState(proc) != ProcessState::ALIVE) {
        MPI_Request_free(request);
        if (!disable_wrappers_asserts)
          assertm(false, "MPI_Iwait could not complete a p2p call");
        return FT_ERROR;
      }
    }
  }
  return FT_SUCCESS;
}

/// Warns user about the safety (in terms of fault tolerance) of using MPI_Wait
/// and redirects to the use of MPI_Iwait version
int __wrap_MPI_Wait(MPI_Request *request, MPI_Status *status) {
  if (disable_ft || ft_handler == nullptr || disable_wrappers) {
    return __real_MPI_Wait(request, status);
  }
  assertm(
      false,
      "Using MPI_Wait(MPI_Request *, MPI_Status *) is unsafe in terms of FT. "
      "Please use the MPI_Iwait(MPI_Request *, MPI_Status *, int) version with "
      "the last argument being the rank of the other side of communication or "
      "FT_MPI_COLLECTIVE if it is collective. If still wanting to call the "
      "original unsafe MPI_Wait use MPI_Iwait(request, status, "
      "FT_WAIT_LEGACY)");
  return FT_ERROR;
}

/// MPI_Itest replaces MPI_Test. The objective of this function is to return an
/// error if the other side of comm died. It is intended to be uses when the \p
/// flag is used as a loop condition. Which simulates the use of MPI_Wait
/// function.
int MPI_Itest(MPI_Request *request, int *flag, MPI_Status *status, int proc) {
  if (disable_ft || ft_handler == nullptr || proc == FT_TEST_LEGACY || disable_wrappers) {
    return __real_MPI_Test(request, flag, status);
  }
  assertm((proc >= FT_MPI_COLLECTIVE) && (proc < ft_handler->getSize()),
          "Waiting for request with invalid rank participating.");

  if (proc == FT_MPI_COLLECTIVE) {
    // If it is a collective call
    if (ft_handler->getCommState() != CommState::VALID) {
      MPI_Request_free(request);
      if (!disable_wrappers_asserts)
        assertm(false, "MPI_Iwait could not complete a collective call");
      return FT_ERROR;
    }
  } else {
    // If it is a point-to-point call
    if (ft_handler->getProcessState(proc) != ProcessState::ALIVE) {
      MPI_Request_free(request);
      if (!disable_wrappers_asserts)
        assertm(false, "MPI_Iwait could not complete a p2p call");
      return FT_ERROR;
    }
  }

  // If there is no errors associated to the processes. return the real call
  __real_MPI_Test(request, flag, status);
  return FT_SUCCESS;
}

/// Warns user about the safety (in terms of fault tolerance) of using MPI_Test
/// in case of using MPI_Test on loops
int __wrap_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {
  if (disable_ft || ft_handler == nullptr || disable_wrappers) {
    return __real_MPI_Test(request, flag, status);
  }
  assertm(
      false,
      "Using MPI_Test(MPI_Request *, int *, MPI_Status *)  is unsafe in terms "
      "of FT if one is using MPI_Test on a waiting loop, since the destination "
      "can fail and not be reached, this function could not return from the "
      "loop if the loop condition is the flag (simulating a call to MPI_Wait). "
      "Please use the MPI_Itestt(MPI_Request *, int *, MPI_Status *, int) "
      "version with the last argument being the rank of the other side of "
      "communication or FT_MPI_COLLECTIVE if it is collective. If still "
      "wanting to call the original not using its flag as a loop condition use "
      "MPI_Iwait(request, flag, status, FT_TEST_LEGACY)");
  return FT_ERROR;
}

/// Wraps MPI_Barrier by replacing the MPI_Barrier by the use of MPI_Ibarrier
/// with MPI_Iwait. Also tries to repair the communicator if it is invalid
int __wrap_MPI_Barrier(MPI_Comm comm) {
  if (disable_ft || ft_handler == nullptr || disable_wrappers) {
    return __real_MPI_Barrier(comm);
  };
  int result = FT_ERROR;
  // Assuming comm holds all processes
  MPI_Request barrier_req;
  if (ft_handler->getCommState() == CommState::VALID) {
    // Although comm is valid, it is still possible that a failure occur, using
    // MPI_Ibarrier and MPI_Iwait prevents it from deadlocking if failure occur
    // in the middle of Barrier call
    MPI_Ibarrier(comm, &barrier_req);
    result = MPI_Iwait(&barrier_req, MPI_STATUS_IGNORE, FT_MPI_COLLECTIVE);
  }
  // If comm is invalid or previouse call to MPI_Ibarrier failed
  while (result == FT_ERROR) {
    // If comm is not valid, let us return an error, but try to fix the comm and
    // call MPI_Ibarrier
    comm = ft_handler->requestCommRepair();
    assertm(comm != MPI_COMM_NULL,
            "MPI_Ibarrier was called within an invalid MPI Comm. FT library "
            "could not the reestablish comm.");
    MPI_Ibarrier(comm, &barrier_req);
    result = MPI_Iwait(&barrier_req, MPI_STATUS_IGNORE, FT_MPI_COLLECTIVE);
    if (result == FT_SUCCESS) {
      result = FT_SUCCESS_NEW_COMM;
      if (ft_handler->getRank() == 0)
        FTDEBUG("[Rank %d FT] - MPI_Barrier was called within an invalid MPI "
                "Comm. Executing in a repaired comm\n",
                ft_handler->getRank());
    }
  }
  // At this point, user should check for the return value, and if it equals
  // to FT_ERROR, the user should call getMainComm() to get the repaired comm
  if (!disable_wrappers_asserts)
    assertm(result != FT_ERROR, "MPI_Barrier could not complete");
  return result;
}

/// Verifies if it is possible (comm is valid) to execute MPI_Comm_free,
/// otherwise, it does not execute and return an error
int __wrap_MPI_Comm_free(MPI_Comm *comm) {
  if (disable_ft || ft_handler == nullptr || disable_wrappers) {
    return __real_MPI_Comm_free(comm);
  }
  // Assuming comm holds all processes
  if (ft_handler->getCommState() == CommState::VALID) {
    __real_MPI_Comm_free(comm);
    return FT_SUCCESS;
  } else {
    // Cannot free a comm without all processes in it but no need to repair it
    // either.
    if (ft_handler->getRank() == 0)
      FTDEBUG(
          "[Rank %d FT] - Could not free communicator with a failed process\n",
          ft_handler->getRank());
    return FT_ERROR;
  }
}

/// Replaces blocking Mprobe by polling calls to MPI_Improbe, it will leave the
/// polling if probe results true or if the other side of communication failed
int __wrap_MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *message,
                      MPI_Status *status) {
  if (disable_ft || ft_handler == nullptr || disable_wrappers) {
    return __real_MPI_Mprobe(source, tag, comm, message, status);
  }
  int probe_flag = 0;
  MPI_Improbe(source, tag, comm, &probe_flag, message, status);

  while (!probe_flag) {
    if (ft_handler->getProcessState(source) != ProcessState::ALIVE) {
      if (!disable_wrappers_asserts)
        assertm(false, "MPI_Mprobe could not complete the operation");
      return FT_ERROR;
    }
    MPI_Improbe(source, tag, comm, &probe_flag, message, status);
  }
  return FT_SUCCESS;
}

/// Replaces the blocking send by Isend and wait for its completition, returns
/// error if the other side of communication failed or success if it completes
int __wrap_MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
                    int tag, MPI_Comm comm) {
  if (disable_ft || ft_handler == nullptr || disable_wrappers) {
    // If FT is disabled
    return __real_MPI_Send(buf, count, datatype, dest, tag, comm);
  }
  // Replaces regular Send by Isend
  MPI_Request w_send_request;
  MPI_Isend(buf, count, datatype, dest, tag, comm, &w_send_request);
  int result = MPI_Iwait(&w_send_request, MPI_STATUS_IGNORE, dest);
  if (result == FT_ERROR) {
    MPI_Request_free(&w_send_request);
    if (!disable_wrappers_asserts)
      assertm(false, "MPI_Send could not complete");
  }
  return result;
}

/// Replaces the blocking Recv by a Probe-Recv function using the Mprobe wrapper
int __wrap_MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
                    int tag, MPI_Comm comm, MPI_Status *status) {
  if (disable_ft || ft_handler == nullptr || disable_wrappers) {
    // If FT is disabled
    return __real_MPI_Recv(buf, count, datatype, source, tag, comm, status);
  }
  // Keep probing until the message is received and stored in msg
  int probe_flag = 0;
  MPI_Message msg;
  MPI_Improbe(source, tag, comm, &probe_flag, &msg, status);
  while (!probe_flag) {
    if (ft_handler->getProcessState(source) != ProcessState::ALIVE) {
      if (!disable_wrappers_asserts)
        assertm(false, "MPI_Recv could not complete the operation");
      return FT_ERROR;
    }
    MPI_Improbe(source, tag, comm, &probe_flag, &msg, status);
  }
  
  MPI_Mrecv(buf, count, datatype, &msg, MPI_STATUS_IGNORE);
  return FT_SUCCESS;
}

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
  main_comm = global_comm;

  MPI_Comm_dup(main_comm, &hb_comm);
  MPI_Comm_rank(hb_comm, &rank);
  MPI_Comm_size(hb_comm, &size);
  MPI_Comm_size(main_comm, &app_size);

  disable_ft = false;
  if (char *env_str = getenv("OMPCLUSTER_FT_DISABLE")) {
    if (std::stoi(env_str) == 1)
      disable_ft = true;
    if (disable_ft) {
      FTDEBUG("Disabling Fault Tolerance feature\n");
      if (rank == 0)
      	printf("Disabling FT\n");
      return;
    }
  }

  disable_wrappers = false;
  if (char *env_str = getenv("OMPCLUSTER_WRAPPERS_DISABLE")) {
    if (std::stoi(env_str) == 1) {
      disable_wrappers = true;
      if (rank == 0)
        printf("Disabling FT wrappers\n");
    }
  }

  // Which type of broadcast
  if (char *env_str = getenv("FTLIB_WHICH_BC")) {
    which_bc = std::stoi(env_str);
  }
  
  ft_handler = this;

  eta = hb_time / DEFAULT_HB_TIME_STEP;
  delta = susp_time / DEFAULT_HB_TIME_STEP;
  time_step = DEFAULT_HB_TIME_STEP;
  hb_done = false;

  setErrorHandling();

  // Checkpointing start procedures, check if using veloc
  if (char *env_str = getenv("OMPCLUSTER_CP_USEVELOC")) {
    if (std::stoi(env_str) == 1) {
      using_veloc = true;
      // OMPCLUSTER_CP_EXECCFG represents application cfg file location
      if (char *exec_file = getenv("OMPCLUSTER_CP_EXECCFG")) {
        cpInit(exec_file);
      } else {
        // OMPCLUSTER_CP_TESTCFG is the cfg file location for llvm lit test
        if (char *test_file = getenv("OMPCLUSTER_CP_TESTCFG")) {
          cpInit(test_file);
        } else {
          cpInit("veloc.cfg");
        }
      }
    }
  }

  // Check if there is a failure inject - This is temporary
  if (char *env_str = getenv("OMPCLUSTER_HB_INJECT_FAILURE")) {
    if (std::stoi(env_str) == 1) 
      inject_failure = true;
  }

  // Benchmarking
  failure_presence = false;
  failures = 0;
  total_failures = 0;
  start_hb = false;

  // Which type of broadcast
  if (char *env_str = getenv("FTLIB_TOTAL_FAILURES")) {
    total_failures = std::stoi(env_str);
  }

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
    c_state = CommState::VALID;
    if (size > 0) {
      p_states = new std::atomic<ProcessState>[size];
      p_states[0] = ProcessState::ALIVE; // there is only one process
    }
  }
}

/// Finishes the main loop of heartbeat thread and synchronize with main thread
FaultTolerance::~FaultTolerance() {
  if (disable_ft)
    return;

  // fprintf(stderr, "Rank %d sent %d and recv %d hb messages \n", rank,
     //     hb_messages_sent, hb_messages_recv);

  // End checkpoint operations
  cpEnd();

  hb_done = true;
  if (heartbeat.joinable() == true) {
    heartbeat.join();
    FTDEBUG("[Rank %d FT] Heartbeat thread joined\n", rank);
  } else {
    FTDEBUG("[Rank %d FT] Error: heartbeat was not started\n", rank);
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

  if (rank == 0) {
    printf("[%d] HB Period: %d - HB Heartbeat: %d - HB Timestep: %d\n", rank,
           eta * time_step, delta * time_step, time_step);
  }

  // Start heartbeat period and suspect time counters
  delta_to = delta - 1;
  eta_to = eta - 1;

  // Set all nodes as neighbors alive state
  p_states = new std::atomic<ProcessState>[size];
  for (int i = 0; i < size; i++) {
    neighbors.push_back(i);
    p_states[i] = ProcessState::ALIVE;
  }

  int shuffle_ring = 0;
  if (char *env_str = getenv("FTLIB_RING_SHUFFLE")) {
    shuffle_ring = std::stoi(env_str);
  }  
  // Shuffle and find emitter and observer
  if (shuffle_ring == 1)
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
 
  // Currently, this vector is access even if CP is not used, so we just start
  // it here instead of in cpInit()
  cp_completed_ranks.resize(size, false);

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
  int message, bc_message[3];
  MPI_Status status;

  // Wait for benchmarking to start
  while(start_hb == false) {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
  
  bc_messages_sent = 0;
  bc_messages_recv = 0;
  hb_messages_sent = 0;
  hb_messages_recv = 0;
  while (!hb_done) {
    if (hb_need_repair) {
      // Waits for the user thread notifies that repair is complete
      std::unique_lock<std::mutex> lock(hb_need_repair_mutex);
      hb_need_repair_cv.wait(lock, [&] { return !hb_need_repair; });
      hb_need_repair_mutex.unlock();
    }

    // Send an alive message to observer if heartbeat period has passed
    if (eta_to <= 0) {
      hbSendAlive();
    } else {
      eta_to--;
    }
    // If received an alive message from emitter, reset the suspect timeout
    flag = MPIw_IProbeRecv(&message, 1, MPI_INT, MPI_ANY_SOURCE, TAG_HB_ALIVE,
                           hb_comm, &status);
    if (!flag) {
      if (message == emitter) {
        hbResetObsTimeout();
        hb_messages_recv++;
      } else {
        // If sender isn't in neighbors, means that is a false positive
        auto known = std::find(neighbors.begin(), neighbors.end(), message);
        if (known == neighbors.end()) {
          // Gives extra time before suspecting the emitter
          delta_to = (2 * delta) - 1;
          // Resets current emitter observer
          int new_obs = message;
          MPI_Isend(&new_obs, 1, MPI_INT, emitter, TAG_HB_NEWOBS, hb_comm,
                    &send_request);
          MPI_Request_free(&send_request);
          // Resets current emitter to the old one
          emitter = message;
          // Broadcast the inclusion of the false positive
          hbBroadcast(HB_BC_ALIVE, emitter, n_pos);
        }
      }
      FTDEBUG("[Rank %d FT] Received alive message from %d\n", rank, message);
    }

    // If no alive messages were received before suspect time, it is a failure
    if (delta_to <= 0)
      hbFindDeadNode();
    else
      delta_to--;

    // Check if there is a new observer, occurs when the current observer fails
    flag = MPIw_IProbeRecv(&message, 1, MPI_INT, MPI_ANY_SOURCE, TAG_HB_NEWOBS,
                           hb_comm, &status);
    if (!flag) {
      hbSetNewObs(message);
      FTDEBUG("[Rank %d FT] %d is the new observer\n", rank, message);
      printf("[Rank %d FT] %d is now observing me\n", rank, message);
    }

    // Check if received a broadcast of failed node
    flag = MPIw_IProbeRecv(bc_message, 3, MPI_INT, MPI_ANY_SOURCE, TAG_HB_BCAST,
                           hb_comm, &status);
    if (!flag) {
      auto valid_source =
          std::find(neighbors.begin(), neighbors.end(), status.MPI_SOURCE);
      auto unknown =
          std::find(neighbors.begin(), neighbors.end(), bc_message[1]);
      switch (bc_message[0]) {
      case HB_BC_FAILURE:
        bc_messages_recv++;
        // Only replicates if failure is unknown and from valid source
        if (unknown != neighbors.end() && valid_source != neighbors.end()) {
          failure_presence = true;
          failures++;
          hbBroadcast(HB_BC_FAILURE, bc_message[1], bc_message[2]);
          FTDEBUG("[Rank %d FT] Received broadcast with failed node: %d\n",
                  rank, message);
        }
        break;
      case HB_BC_ALIVE:
        bc_messages_recv++;
        // Only replicates if process is not added in neighbors yet
        if (unknown == neighbors.end() && valid_source != neighbors.end()) {
          hbBroadcast(HB_BC_ALIVE, bc_message[1], bc_message[2]);
          FTDEBUG("[Rank %d FT] Received broadcast of false positive: %d\n",
                  rank, message);
        }
        break;
      case HB_BC_REPAIR:
        bc_messages_recv++;
        if (c_state == CommState::INVALID && valid_source != neighbors.end()) {
          if (hb_need_repair == false) {
            hbBroadcast(HB_BC_REPAIR, bc_message[1], bc_message[2]);
            FTDEBUG("[Rank %d FT] Received broadcast of repair operation: %d\n",
                    rank, message);
          }
        }
        break;
      case CP_COMPLETED:
        bc_messages_recv++;
        if (cp_completed_ranks[bc_message[2]] == false) {
          hbBroadcast(CP_COMPLETED, bc_message[1], bc_message[2]);
          FTDEBUG(
              "[Rank %d FT] Received broadcast of checkpoint completion: %d\n",
              rank, message);
        }
        break;
      default:
        FTDEBUG("[Rank %d FT] Ignoring unknown broadcast\n", rank);
        break;
      }
    }
    // Heartbeat thread time step
    std::this_thread::sleep_for(std::chrono::milliseconds(time_step));
  }
}

/// Sends a message to the observer saying it is alive
void FaultTolerance::hbSendAlive() {
  eta_to = eta - 1;
  MPI_Isend(&rank, 1, MPI_INT, observer, TAG_HB_ALIVE, hb_comm, &send_request);
  MPI_Request_free(&send_request);
  hb_messages_sent++;
}

/// Resets suspect time upon receiving alive messages from emitter
void FaultTolerance::hbResetObsTimeout() {
  delta_to = delta - 1;
}

/// Do all the necessary procedures after noticing the emitter failed
void FaultTolerance::hbFindDeadNode() {
  failure_presence = true;
  failures++;

  // Gives extra time before suspecting the emitter
  delta_to = (2 * delta) - 1;

  FTDEBUG("[Rank %d FT] Found a failed process: %d\n", rank, emitter);
  fprintf(stderr, "[Rank %d FT] Found one failed process: %d\n", rank, emitter);

  // Broadcast failure to other nodes
  hbBroadcast(HB_BC_FAILURE, emitter, n_pos);

  // Find a new emitter and send a message saying its new observer
  emitter = hbFindEmitter();
  printf("[Rank %d FT] - New Emitter: %d\n",rank, emitter);
  MPI_Isend(&rank, 1, MPI_INT, emitter, TAG_HB_NEWOBS, hb_comm,
            &send_request);
  MPI_Request_free(&send_request);

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
  int bc_message[3] = {type, value, pos};

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
      cp_completed_ranks.erase(cp_completed_ranks.begin() + pos);
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
    cp_completed_ranks.insert(cp_completed_ranks.begin() + pos, false);
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
  case CP_COMPLETED: {
    cp_completed_ranks[pos] = true;
    int completed_all = true;
    // If all ranks completed the 
    for (bool entry : cp_completed_ranks)
      if (entry == false)
        completed_all = false;
    if (completed_all) {
      for (const auto &callback : notify_callbacks)
        callback({FTNotificationID::CHECKPOINT_DONE, 0});
      cp_completed_ranks.resize(size, false);
    }
  } break;
  default: {
    FTDEBUG("[Rank %d FT] Ignoring unknown broadcast\n", rank);
  } break;
  }

  // First Option: Chord-like broadcast
  if (which_bc == 0) {
    for (std::size_t i = 1; i < neighbors.size(); i *= 2) {
      int index = (n_pos + i) % neighbors.size();
      if (neighbors[index] != rank) {
        bc_messages_sent++;
        MPI_Isend(bc_message, 3, MPI_INT, neighbors[index], TAG_HB_BCAST,
                  hb_comm, &send_request);
        MPI_Request_free(&send_request);
      }
    }
  } else if (which_bc == 1) {
    // Second Option: BMG broadcast
    for (std::size_t i = 1; i < neighbors.size(); i *= 2) {
      // Successor
      int index_s = (n_pos + i) % neighbors.size();
      if (neighbors[index_s] != rank) {
        bc_messages_sent++;
        MPI_Isend(bc_message, 3, MPI_INT, neighbors[index_s], TAG_HB_BCAST,
                  hb_comm, &send_request);
        MPI_Request_free(&send_request);
      }
      // Predecessor
      int index_p = (n_pos - i) % neighbors.size();
      index_p = index_p < 0 ? neighbors.size() + index_p : index_p;
      if (neighbors[index_p] != rank) {
        bc_messages_sent++;
        MPI_Isend(bc_message, 3, MPI_INT, neighbors[index_p], TAG_HB_BCAST,
                  hb_comm, &send_request);
        MPI_Request_free(&send_request);
      }
    }
  } else if (which_bc == 2) {
    // Third Option: HBA broadcast k = floor(log2(n))
    int k = static_cast<int>(std::floor(std::log2(neighbors.size())));
    for (std::size_t i = 0; i < k; i++) {
      // Successor
      int index_s = (n_pos + i) % neighbors.size();
      if (neighbors[index_s] != rank) {
        bc_messages_sent++;
        MPI_Isend(bc_message, 3, MPI_INT, neighbors[index_s], TAG_HB_BCAST,
                  hb_comm, &send_request);
        MPI_Request_free(&send_request);
      }
      // Predecessor
      int index_p = (n_pos - i);
      index_p = index_p < 0 ? neighbors.size() + index_p : index_p;
      if (neighbors[index_p] != rank) {
        bc_messages_sent++;
        MPI_Isend(bc_message, 3, MPI_INT, neighbors[index_p], TAG_HB_BCAST,
                  hb_comm, &send_request);
        MPI_Request_free(&send_request);
      }
    }
  }
}

// Checkpointing related functions
// ===----------------------------------------------------------------------===
// Default MTBF value
constexpr int DEFAULT_MBTF = 86400; // TODO: Change to reference
// Default write speed in MB/s
constexpr int DEFAULT_WRITE_SPEED = 10;
// Veloc control variables
static bool test_veloc = false;

/// Initializes checkpoint related variables and VELOC passing \p loc as veloc
/// configuration file path
void FaultTolerance::cpInit(const char *loc) {
#if COMPILE_WITH_VELOC
  cp_cfg_file_path = std::string(loc);
  cp_reg_pointers.clear();
  cp_next_id = 0;
  cp_is_done = false;
  cp_completed = false;
  cp_next_region_id = 0;
  cp_rank = -1;
  // Mean time between failures should be in seconds
  cp_mtbf = DEFAULT_MBTF;
  if (char *env_str = getenv("OMPCLUSTER_CP_MTBF")) {
    cp_mtbf = std::stoi(env_str);
    FTDEBUG("[Rank %d FT] Using CP mtbf defined in environment variables: %d\n",
            rank, cp_mtbf);
  }
  // Write speed should be in MB/s
  cp_wspeed = DEFAULT_WRITE_SPEED;
  if (char *env_str = getenv("OMPCLUSTER_CP_WSPEED")) {
    cp_wspeed = std::stoi(env_str);
    FTDEBUG("[Rank %d FT] Using CP write speed defined in environment "
            "variables: %d\n",
            rank, cp_wspeed);
  }

  // First cost time will be one.
  cp_cost = 1;

  if (VELOC_Init_single(rank, loc) != VELOC_SUCCESS) {
    using_veloc = false;
    return;
  }

  cp_rank = rank;
  cpSetNextInterval();
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
#endif
}

/// Clean up checkpoint directories uppon successfully application conclusion
/// and ends checkpointing thread
void FaultTolerance::cpEnd() {
#if COMPILE_WITH_VELOC
  if (using_veloc == false)
    return;

  VELOC_Finalize(0);
  // End checkpoint thread
  cp_is_done = true;
  cp_is_done_cv.notify_all();
  // Join checkpoint thread when it finishes, if started
  if (checkpoint.joinable())
    checkpoint.join();
  // Join wait transfer thread whe it finishes, if started
  if (wait_transfer.joinable())
    wait_transfer.join();
  // Join test thread if exists
  if (test_thread.joinable())
    test_thread.join();
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
#endif
}

/// Calculates the interval till next checkpoint and instantiate a new
/// checkpointing thread
void FaultTolerance::cpSetNextInterval() {
#if COMPILE_WITH_VELOC
  if (using_veloc == false)
    return;

  // Young's Formula. Source: John W. Young. 1974. A first order approximation
  // to the optimum checkpoint interval. Commun. ACM 17, 9 (Sept. 1974),
  // 530–531.
  int cp_next_interval = floor(sqrt(2 * cp_cost * cp_mtbf));

  FTDEBUG("[Rank %d FT] Next checkpoint will be in %d seconds\n", rank,
          cp_next_interval);
  if (checkpoint.joinable())
    checkpoint.join();
  checkpoint = std::thread(&FaultTolerance::cpRunSave, this, cp_next_interval);
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
#endif
}

static bool send = false;
/// Waits until user calls the checkpointing save function and then execute the
/// procedures to save a new checkpoint
void FaultTolerance::cpRunSave(int cp_next_interval) {
#if COMPILE_WITH_VELOC
  if (using_veloc == false)
    return;

  // sleep until next checkpoint time or program finish
  std::unique_lock<std::mutex> done_lock(cp_is_done_mutex);
  auto now = std::chrono::system_clock::now();
  if (!cp_is_done_cv.wait_until(done_lock,
                                now + std::chrono::seconds(cp_next_interval),
                                [this]() { return cp_is_done == true; })) {
    // Notify application
    if (!notify_callbacks.empty() && !cp_is_done) {
      for (const auto &callback : notify_callbacks)
        callback({FTNotificationID::CHECKPOINT, rank});
    } else {
      FTDEBUG("[Rank %d FT] No checkpointing callback set, application was "
              "finished  or cp was cancelled\n",
              rank);
      return;
    }
  }

  // Temporary
  if (inject_failure == true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    if (send == false) {
      send = true;
      for (const auto &callback : notify_callbacks)
        callback({FTNotificationID::FAILURE, 2});
    }
  }
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
#endif
}

/// User level function to call the procedures to save a new checkpoint
int FaultTolerance::cpSaveCheckpoint() {
#if COMPILE_WITH_VELOC
  if (using_veloc == false)
    return FT_VELOC_ERROR;
  // Temporary
  test_veloc = true;

  // Save the checkpoint
  if (cp_reg_pointers.size() > 0) {
    std::string cp_name = "ckpt";
    if (VELOC_Checkpoint(cp_name.c_str(), cp_next_id) != VELOC_SUCCESS) {
      FTDEBUG("[Rank %d FT] - Could not save the checkpoint\n", rank);
      return FT_VELOC_ERROR;
    }

    cp_cost = 0;
    for (size_t i = 0; i < cp_reg_pointers.size(); i++) {
      cp_reg_pointers[i].cp_version = cp_next_id;
      cp_reg_pointers[i].cp_rank = cp_rank;
      cp_cost += cp_reg_pointers[i].size + cp_reg_pointers[i].base_size;
    }
    // Transform cost in MB and cacalute time to write based on write speed
    cp_cost = (cp_cost / 1024 / 1024) / cp_wspeed;
    // Define a minimun value of 1 to cp_cost;
    cp_cost = (cp_cost < 1) ? 1 : cp_cost;
  }
  // Writting or not a checkpoint. Increment the counter
  cp_next_id++;

  // Wait for Veloc to send the checkpoint from scratch to persistent directory
  // in another thread to wait in background
  if (wait_transfer.joinable())
    wait_transfer.join();
  wait_transfer = std::thread(&FaultTolerance::cpWaitTransfer, this);
  
  // Calculate the next checkpoint after completing the checkpoint
  if (!test_veloc)
    cpSetNextInterval();

  return (cp_next_id - 1);
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
  return 0;
#endif
}

void FaultTolerance::cpWaitTransfer() {
#if COMPILE_WITH_VELOC
  if (using_veloc == false)
    return;
  // Wait only if saved anything (Veloc seems to fail if doing procedures
  // without registered pointers)
  int result = VELOC_SUCCESS;
  if (cp_reg_pointers.size() > 0) 
    result = VELOC_Checkpoint_wait();

  if (result == VELOC_SUCCESS) {
    // If operation succeed, send checkpoint complete notification
    cp_completed = true;
    hbBroadcast(CP_COMPLETED, -1, n_pos);
  } else {
    // If operation doens't succeed, something is wrong with veloc configuration
    // or with the system (e.g: permissions)
    using_veloc = false;
    FTDEBUG("[Rank %d FT] - Veloc couldn't transfer checkpoint from scratch to "
            "persistent directory, please check config options and "
            "permissions, disabling checkpoint\n",
            rank);
  }
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
#endif
}

/// Returns a copy of the current protect memory regions
const std::vector<CPAllocatedMem> &FaultTolerance::cpGetAllocatedMem() {
  return cp_reg_pointers;
}

/// Ends current Veloc context, to enable loading other's rank context
void FaultTolerance::cpLoadStart() {
#if COMPILE_WITH_VELOC
  if (using_veloc == false)
    return;

  // The load process should be done by calling cpLoadStart() one time, one or
  // multiple calls to cpLoadMem(), and one final call to cpLoadEnd(), so we
  // will lock the mutex in cpLoadStart and free it in cpLoadEnd().
  cp_load_mutex.lock();

  VELOC_Finalize(0);
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
#endif
}

// TODO: LoadMem function is currently implemented such that every memory region
// will start and finalize Veloc. In the future, we should be able to restore
// all regions that belong to the same checkpoint file.
/// Recovers a specific region of memory saved with \p id in the cp \p version
/// by the process of \p s_cp_rank
int FaultTolerance::cpLoadMem(int id, int s_cp_rank, int ver, size_t count,
                              size_t base_size, void *memregion) {
#if COMPILE_WITH_VELOC
  if (using_veloc == false)
    return FT_VELOC_ERROR;

  // Initialize Veloc on the s_cp_rank, so Veloc will route the right file.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (VELOC_Init_single(s_cp_rank, cp_cfg_file_path.c_str()) != VELOC_SUCCESS) {
    using_veloc = false;
    FTDEBUG("[Rank %d FT] - Could not start Veloc with rank %d - %s\n", rank,
            s_cp_rank, cp_cfg_file_path.c_str());
    return FT_VELOC_ERROR;
  }

  std::string cp_name = "ckpt";
  // result of v should be equal to the version argument
  int v = VELOC_Restart_test(cp_name.c_str(), ver + 1);
  if (v == ver) {
    // Protect the buffer argument with the id saved in checkpoint
    VELOC_Mem_protect(id, memregion, count, base_size);
    FTDEBUG("[Rank %d FT] - Recovering checkpoint version %d\n", rank, v);
    // Load the memory region
    VELOC_Restart_begin(cp_name.c_str(), v);
    int ids[1] = {id};
    int r = VELOC_Recover_selective(VELOC_RECOVER_SOME, ids, 1);
    VELOC_Restart_end(0);
    FTDEBUG("Result of restart: %s\n",
            (r == VELOC_FAILURE) ? "Failure" : "Success");
    // Ends this context of Veloc
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    VELOC_Finalize(0);
    // Register in allocated mem vector to protect on the new rank at the end of
    // the checkpointing loading
    cp_reg_pointers.push_back(
        {cp_next_region_id, memregion, count, base_size, -1, -1});
    cp_next_region_id++;
    return FT_VELOC_SUCCESS;
  } else {
    FTDEBUG("[Rank %d FT] - Last valid CP and version doesn't match\n", rank);
    return FT_VELOC_ERROR;
  }
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
  return FT_VELOC_ERROR;
#endif
}

/// Restores the context of Veloc before loading memory regions
void FaultTolerance::cpLoadEnd() {
#if COMPILE_WITH_VELOC
  if (using_veloc == false)
    return;

  // Initialize Veloc on this rank again
  if (VELOC_Init_single(rank, cp_cfg_file_path.c_str()) != VELOC_SUCCESS) {
    using_veloc = false;
    return;
  }
  // Protect the Memory regions again
  for (size_t i = 0; i < cp_reg_pointers.size(); i++) {
    VELOC_Mem_protect(cp_reg_pointers[i].id, cp_reg_pointers[i].pointer,
                      cp_reg_pointers[i].size, cp_reg_pointers[i].base_size);
  }

  // Free the load process mutex
  cp_load_mutex.unlock();
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
#endif
}

/// Registers the region of address \p ptr to allocated memory vector and sets
/// VELOC to save the region
void FaultTolerance::cpRegisterPointer(void *ptr, size_t count,
                                       size_t base_size, int *id) {
#if COMPILE_WITH_VELOC
  if (using_veloc == false) {
    *id = -1;
    return;
  }

  cp_regions_mutex.lock();
  *id = cp_next_region_id;
  cp_reg_pointers.push_back({cp_next_region_id, ptr, count, base_size, -1, -1});
  VELOC_Mem_protect(cp_next_region_id, ptr, count, base_size);
  cp_next_region_id++;
  cp_regions_mutex.unlock();
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
#endif
}

/// Unregisters the region of address \p ptr to allocated memory vector and
/// unset VELOC to save the region
void FaultTolerance::cpUnregisterPointer(void *ptr) {
#if COMPILE_WITH_VELOC
  if (using_veloc == false)
    return;

  cp_regions_mutex.lock();
  for (size_t i = 0; i < cp_reg_pointers.size(); i++) {
    if (cp_reg_pointers[i].pointer == ptr) {
      VELOC_Mem_unprotect(cp_reg_pointers[i].id);
      cp_reg_pointers.erase(cp_reg_pointers.begin() + i);
      // If the pointer is foud in the vector, free the lock before returning
      cp_regions_mutex.unlock();
      return;
    }
  }
  // If no pointer was found in the vector, free the lock
  cp_regions_mutex.unlock();
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
#endif
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

  // Temporary
  if (inject_failure == true)
    forceCP(2);
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

/// Test purpose function that sets to save a checkpoint in the next second
int FaultTolerance::forceCP(int cp_interval) {
#if COMPILE_WITH_VELOC
  if (using_veloc == false)
    return FT_VELOC_ERROR;

  test_veloc = true;
  test_thread = std::thread(&FaultTolerance::cpRunSave, this, cp_interval);
  return FT_VELOC_SUCCESS;
#else
  FTDEBUG("[Rank %d FT] - Checkpointing isn't enabled\n", rank);
  return FT_VELOC_ERROR;
#endif
}

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

/// Disables asserts for testing
void FaultTolerance::disableAsserts() {
  if (disable_ft) {
    return;
  }
  disable_wrappers_asserts = true;
}

int FaultTolerance::getFailures() {
  if (disable_ft)
    return 0;
  return failures;
}

int FaultTolerance::getTotalFailures() {
  if (disable_ft)
    return 0;
  return total_failures;
}
