#include <chrono>
#include <mpi.h>
#include <thread>
#include <cstdlib>
#include <time.h>

#include "ft.h"

#define MAX_SECONDS 120

int main() {
  int rank, size, provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  ft::FaultTolerance *hb = new ft::FaultTolerance(
      ft::DEFAULT_HB_TIME, ft::DEFAULT_HB_TIMEOUT, rank, size);

  srand(time(NULL));

  for (int i = 1; i <= MAX_SECONDS; i++) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (i % 9 == 0) {
      if (2 == rank) {
        delete hb;
        MPI_Finalize();
      }
    }
  }

  delete hb;
  MPI_Finalize();
}