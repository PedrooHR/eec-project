#include <chrono>
#include <mpi.h>
#include <thread>

#include "ft.h"

#define MAX_SECONDS 120

int main() {
  int rank, size, provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //ft::FaultTolerance *hb = new ft::FaultTolerance(
  //    ft::DEFAULT_HB_TIME, ft::DEFAULT_HB_TIMEOUT, rank, size);

  ft::FaultTolerance *hb = new ft::FaultTolerance(
      ft::DEFAULT_HB_TIME, ft::DEFAULT_HB_TIMEOUT, MPI_COMM_WORLD);


  for (int i = 0; i < MAX_SECONDS; i++) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  delete hb;
  MPI_Finalize();
}
