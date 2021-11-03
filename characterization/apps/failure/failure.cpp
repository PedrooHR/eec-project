#include <chrono>
#include <mpi.h>
#include <thread>
#include <cstdlib>
#include <cstdio>
#include <time.h>

#define MAX_SECONDS 120

int main() {
  int rank, size, provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  int fail_rank = 2;
  double t1, t2;
  t1 = MPI_Wtime();	
  for (int i = 1; i <= MAX_SECONDS; i++) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (i % 19 == 0) {
      if (fail_rank == rank) {
        MPI_Finalize();
        return 0;
      }
      fail_rank += 2;
    }
  }
  t2 = MPI_Wtime();
  if (rank == 0)
    printf("Total time: %1.2f\n", t2 - t1);

  MPI_Finalize();
  return 0;
}
