#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <unistd.h>
#include <vector>

#include <thread>

#include <nng/nng.h>
#include <nng/protocol/pair1/pair.h>

using namespace std;

#define MAX_RANKS 3

int send_msg(nng_socket sock, int rank) {
  // char msg[10];
  // strcpy(msg, to_string(rank).c_str());
  int msg[2] = {rank, 0};
  int result = nng_send(sock, msg, 2*sizeof(int), NNG_FLAG_NONBLOCK);
  return (result);
}

int recv_msg(nng_socket sock, int myrank) {
  // char msg[10] = {"-1"};
  int msg[2];
  size_t size;
  int result = nng_recv(sock, msg, &size, NNG_FLAG_NONBLOCK);
  if (result == 0)
    printf("[Rank %d] - Received alive from %d, size %ld\n", myrank, msg[0], size);
  return (result);
}

void worker(int rank) {
  vector<nng_socket> sockets;
  vector<string> urls;
  sockets.resize(MAX_RANKS);
  urls.resize(MAX_RANKS);

  // Start sockets (a socke between each pair of processes)
  for (int i = 0; i < MAX_RANKS; i++) {
    if (i != rank) { // No self Sockets
      nng_pair_open(&sockets[i]);
      // Higher ranks as dialers in the socket
      if (rank < i)
        urls[i] = std::string("ws://localhost:55555/l") + std::to_string(rank) +
                  "-d" + std::to_string(i);
      else
        urls[i] = std::string("ws://localhost:55555/l") + std::to_string(i) +
                  "-d" + std::to_string(rank);
    }
  }

  // Initialize listenes
  for (int i = 0; i < MAX_RANKS; i++) {
    if (i != rank) {
      // Higher ranks as dialers in the socket
      if (rank < i) {
        int result;
        do {
          result = nng_listen(sockets[i], urls[i].c_str(), NULL, 0);
          printf("[Rank %d] result of listen %d (%s)\n", rank, result,
                  urls[i].c_str());
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
        } while (result != 0);
      }
    }
  }
  sleep(1);
  // Initialize dilers
  for (int i = 0; i < MAX_RANKS; i++) {
    if (i != rank) {
      // Higher ranks as dialers in the socket
      if (rank > i) {
        int result;
        do {
          result = nng_dial(sockets[i], urls[i].c_str(), NULL, 0);
          printf("[Rank %d] result of dial %d (%s)\n", rank, result,
                  urls[i].c_str());
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
        } while (result != 0);
      }
    }
  }


  // for (int i = 0; i < MAX_RANKS; i++) {
  //   if (i != rank) {
  //     nng_pair_open(&sockets[i]);

  //     // Higher ranks as dialers in the socket
  //     if (rank < i) {
  //       urls[i] = string("ws://localhost:56456/") + to_string(rank) + "-" +
  //                 to_string(i);
  //       int result;
  //       do {
  //         result = nng_listen(sockets[i], urls[i].c_str(), NULL, 0);
  //         printf("[Rank %d] result of listen %d (%s)\n", rank, result,
  //                urls[i].c_str());
  //       } while (result != 0);
  //     } else {
  //       urls[i] = string("ws://localhost:56456/") + to_string(i) + "-" +
  //                 to_string(rank);
  //       int result;
  //       do {
  //         result = nng_dial(sockets[i], urls[i].c_str(), NULL, 0);
  //         printf("[Rank %d] result of dial %d (%s)\n", rank, result,
  //                urls[i].c_str());
  //       } while (result != 0);
  //     }
  //   }
  // }
  printf("Initializing -----------------------------------\n");
  int emit_to = (rank + 1) % MAX_RANKS;

  while (true) {
    // Send alive message to observer
    send_msg(sockets[emit_to], rank);

    // Iterate over the sockets to see if any message was received
    for (int i = 0; i < MAX_RANKS; i++) {
      if (i != rank) {
        recv_msg(sockets[i], rank);
      }
    }
    sleep(1);
  }
}

int main() {

  vector<thread> threads(MAX_RANKS);
  for (int i = 0; i < MAX_RANKS; i++) {
    threads[i] = thread(worker, i);
  }

  for (int i = 0; i < MAX_RANKS; i++) {
    threads[i].join();
  }

  return 0;
}