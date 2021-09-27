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

// These wrapping functions will be located outside of ft namespace since it
// will auto replace the MPI calls that are being wrapped
extern "C" {
// Reference for real wrapped functions (this refers to the MPI distribution
// implementations)
extern int __real_MPI_Wait(MPI_Request *request, MPI_Status *status);
extern int __real_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
extern int __real_MPI_Barrier(MPI_Comm comm);
extern int __real_MPI_Comm_free(MPI_Comm *comm);
extern int __real_MPI_Mprobe(int source, int tag, MPI_Comm comm,
                             MPI_Message *message, MPI_Status *status);
extern int __real_MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                           int dest, int tag, MPI_Comm comm);
extern int __real_MPI_Recv(void *buf, int count, MPI_Datatype datatype,
                           int source, int tag, MPI_Comm comm,
                           MPI_Status *status);
// Custom wrapped functions (this refers to the functions implemented by this
// library)
int MPI_Iwait(MPI_Request *request, MPI_Status *status, int proc);
int MPI_Itest(MPI_Request *request, int *flag, MPI_Status *status, int proc);
int __wrap_MPI_Wait(MPI_Request *request, MPI_Status *status);
int __wrap_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
int __wrap_MPI_Barrier(MPI_Comm comm);
int __wrap_MPI_Comm_free(MPI_Comm *comm);
int __wrap_MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *message,
                      MPI_Status *status);
int __wrap_MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
                    int tag, MPI_Comm comm);
int __wrap_MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
                    int tag, MPI_Comm comm, MPI_Status *status);
}

// Heartbeat message tags definition
enum FTMPITags {
  TAG_HB_ALIVE = 0x70,
  TAG_HB_NEWOBS = 0x71,
  TAG_HB_BCAST = 0x72,
  TAG_CP_DONE = 0x73
};

enum HBBCTypes {
  HB_BC_FAILURE = 1,
  HB_BC_ALIVE = 2,
  HB_BC_REPAIR = 3,
  CP_COMPLETED
};

enum class ProcessState { DEAD, ALIVE };

enum class CommState { VALID, INVALID };

enum class FTNotificationID {
  FAILURE,
  FALSE_POSITIVE,
  CHECKPOINT,
  CHECKPOINT_DONE
};

struct FTNotification {
  FTNotificationID notification_id;
  int process_id;
};

// (void *) to function pointer conversion
typedef void (*FtCallbackTy)(FTNotification); 

struct CPAllocatedMem {
  int id;           // Id given to a pointer
  void *pointer;    // Address of allocated memory
  size_t size;      // Size (in bytes) of allocated memory
  size_t base_size; // Control variable (true if needs cp)
  int cp_version;   // Version of the last valid cp
  int cp_rank;      // Rank of the process that saved the cp
};

// Default values for FT library
constexpr int DEFAULT_HB_TIMEOUT = 30000; // in ms
constexpr int DEFAULT_HB_TIME = 1000;     // in ms
constexpr int DEFAULT_HB_TIME_STEP = 50;  // in ms

// Return values definition
constexpr int FT_VELOC_ERROR = -1;
constexpr int FT_VELOC_SUCCESS = 0;

// Fault tolerance definitions for wrappers
constexpr int FT_SUCCESS = 0;
constexpr int FT_ERROR = 1;
constexpr int FT_SUCCESS_NEW_COMM = 2;
constexpr int FT_MPI_COLLECTIVE = -1;
constexpr int FT_WAIT_LEGACY = -2;
constexpr int FT_TEST_LEGACY = -2;

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

  // Checkpointing variables
  std::string cp_cfg_file_path;
  std::thread checkpoint, test_thread, wait_transfer;
  std::condition_variable cp_is_ready_cv, cp_is_done_cv;
  std::mutex cp_is_ready_mutex, cp_is_done_mutex;
  std::mutex cp_regions_mutex, cp_load_mutex;
  std::vector<CPAllocatedMem> cp_reg_pointers;
  std::vector<bool> cp_completed_ranks;
  double cp_cost;
  int cp_mtbf, cp_wspeed, cp_next_region_id;
  int cp_next_id, cp_rank;
  bool cp_is_done, cp_completed;
  // Checkpointing functions
  void cpSetNextInterval();
  void cpRunSave(int cp_next_interval);
  void cpWaitTransfer();
  void cpInit(const char *loc);
  void cpEnd();

public:
  // Constructors
  FaultTolerance(int hb_time, int susp_time, MPI_Comm global_comm);
  ~FaultTolerance();

  // User level checkpoint functions
  int cpSaveCheckpoint();
  void cpRegisterPointer(void *ptr, size_t count, size_t base_size, int *id);
  void cpUnregisterPointer(void *ptr);
  const std::vector<CPAllocatedMem> &cpGetAllocatedMem();
  void cpLoadStart();
  int cpLoadMem(int id, int rank, int ver, size_t count, size_t base_size,
                void *memregion);
  void cpLoadEnd();

  // User level general fault tolerance functions
  ProcessState getProcessState(int id); // Get state of desired process
  CommState getCommState();             // Get state of communicator
  MPI_Comm requestCommRepair();         // Host request to repair communicator
  MPI_Comm getMainComm();               // Return the main communicator
  void registerNotifyCallback(std::function<void(FTNotification)> callback);

  // Test purpose functions
  uint64_t getID(); // Return the thread id
  int getEmitter(); // Return current emitter
  int forceCP(int cp_interval);

  // Wrapper helping functions
  int getRank();
  int getSize();
  void disableAsserts();

  // Benchmarking
  bool failure_presence;
  std::atomic<int> failures;
  int total_failures;
  int bc_messages_recv, bc_messages_sent;
  int hb_messages_recv, hb_messages_sent;
  int which_bc;
  std::atomic<bool> start_hb;
  int getFailures();
  int getTotalFailures();
};

#endif // _OMPTARGET_FTLIB_H_
