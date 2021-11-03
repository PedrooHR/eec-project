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
#define MPICH_SKIP_MPICXX
#include <mpi.h>
#include <thread>
#include <vector>

namespace ft {

// Heartbeat message tags definition
enum HBMPITags {
  TAG_HB_ALIVE = 0x70,
  TAG_HB_NEWOBS = 0x71,
  TAG_HB_BCAST = 0x72
};

enum HBBCTypes { HB_BC_FAILURE = 1, HB_BC_ALIVE = 2, HB_BC_REPAIR = 3 };

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
constexpr int DEFAULT_HB_TIMEOUT = 3000;  // in ms
constexpr int DEFAULT_HB_TIME = 1000;     // in ms
constexpr int DEFAULT_HB_TIME_STEP = 50;  // in ms

// Fault tolerance class declaration
class FaultTolerance {
private:
  // MPI definitions
  int size, app_size;                // MPI size of communicator
  int rank;                          // MPI rank of node
  MPI_Comm main_comm, hb_comm;       // Communicator handlers
  MPI_Errhandler main_error_handler; // main_comm error handling object
  MPI_Errhandler hb_error_handler;   // hb_comm error handling object
  MPI_Request send_request;          // MPI required request in send functions
  MPI_Message msg_recv;              // Holds probed messages
  int msg_recv_flag;                 // Indicates received messages
  // MPI related functions
  template <typename T, typename S, typename R>
  int MPIw_IProbeRecv(T *buffer, int size, S type, int source, int tag,
                      MPI_Comm comm, R status);

  // Fault tolerance general functions
  void setErrorHandling(); // Configure error handling procedures
  void commRepair();       // Repair communicator

  // Heartbeat variables
  int delta;     // Time for suspecting a failure
  int delta_to;  // Suspect time counter
  int eta;       // Period of hearbeat
  int eta_to;    // Heartbeat period counter
  int time_step; // Heartbeat thread loop time step
  int observer, emitter, n_pos;
  std::thread heartbeat;
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
  FaultTolerance(int hb_time, int susp_time, MPI_Comm global_comm);
  ~FaultTolerance();

  // User level general fault tolerance functions
  ProcessState getProcessState(int id); // Get state of desired process
  CommState getCommState();             // Get state of communicator
  MPI_Comm requestCommRepair();         // Host request to repair communicator
  MPI_Comm getMainComm();               // Return the main communicator
  void registerNotifyCallback(std::function<void(FTNotification)> callback);

  // Test purpose functions
  uint64_t getID(); // Return the thread id
  int getEmitter(); // Return current emitter

  // Wrapper helping functions
  int getRank();
  int getSize();
};

} // namespace ft

#endif // _OMPTARGET_FTLIB_H_
