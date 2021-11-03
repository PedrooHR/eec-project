//===------ ft.h - Common MPI fault tolerance declarations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of fault tolerance class, functions,
// types and macros.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_FTLIB_H_
#define _OMPTARGET_FTLIB_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <nng/nng.h>

namespace ft {

// Heartbeat message tags definition
enum HBMPITags {
  TAG_HB_ALIVE = 0x70,
  TAG_HB_NEWOBS = 0x71,
  TAG_HB_BCAST = 0x72
};

enum HBBCTypes { HB_BC_FAILURE = 1, HB_BC_ALIVE = 2 };

enum class ProcessState { DEAD, ALIVE };

enum class CommState { VALID, INVALID };

enum class FTNotificationID {
  FAILURE,
  FALSE_POSITIVE
};

struct FTNotification {
  FTNotificationID notification_id;
  int process_id;
};

// Default values for FT library
constexpr int DEFAULT_HB_TIMEOUT = 5000;  // in ms
constexpr int DEFAULT_HB_TIME = 1000;     // in ms
constexpr int DEFAULT_HB_TIME_STEP = 50;  // in ms

// Fault tolerance class declaration
class FaultTolerance {
private:
  // MPI definitions
  int size, app_size;                // MPI size of communicator
  int rank;                          // MPI rank of node
  int msg_recv_flag;                 // Indicates received messages

  std::vector<nng_socket> sockets;
  std::vector<std::string> urls;
  std::vector<bool> open_sockets;

  // Heartbeat variables
  int delta;     // Time for suspecting a failure
  int delta_to;  // Suspect time counter
  int eta;       // Period of hearbeat
  int eta_to;    // Heartbeat period counter
  int time_step; // Heartbeat thread loop time step
  int observer, emitter, n_pos;
  std::thread heartbeat, alive;
  std::mutex hb_need_repair_mutex, hb_started_mutex;
  std::condition_variable hb_need_repair_cv, hb_started_cv;
  bool hb_started, hb_need_repair;
  std::atomic<bool> hb_done;
  std::atomic<ProcessState> *p_states;
  std::atomic<CommState> c_state;
  std::vector<int> neighbors;
  std::thread::id thread_id;
  std::vector<std::function<void(FTNotification)>> notify_callbacks;
  // Heartbeat functions
  void hbInit();                 // Initialize heartbeat
  void hbMain();                 // Main function of heartbeat
  void hbSendAlive();            // Send alive message for observer
  void hbResetObsTimeout();      // Reset suspect time-out
  void hbFindDeadNode();         // Set and broadcast emitter failure
  int hbFindEmitter();           // Find new emitter after failure
  void hbSetNewObs(int new_obs); // Set new observer after failure
  void hbBroadcast(int type, int value, int pos); // Internal broadcast

public:
  // Constructors
  FaultTolerance(int hb_time, int susp_time, int my_rank, int total_size);
  ~FaultTolerance();

  // User level general fault tolerance functions
  ProcessState getProcessState(int id); // Get state of desired process
  CommState getCommState();             // Get state of communicator
  void registerNotifyCallback(std::function<void(FTNotification)> callback);

  // Test purpose functions
  uint64_t getID(); // Return the thread id
  int getEmitter(); // Return current emitter
};

} // namespace ft

#endif // _OMPTARGET_FTLIB_H_
