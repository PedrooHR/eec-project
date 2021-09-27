#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

#define BOOTSTRAP_ITERATIONS 10000

template <class T>
void bootstrap(std::vector<T> values, float *vmin, float *vmax, float alfa) {
  unsigned int size = values.size();
  std::vector<float> avg_values;
  avg_values.resize(BOOTSTRAP_ITERATIONS, 0);

  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist(0, size - 1);
#pragma omp parallel for
  for (int i = 0; i < BOOTSTRAP_ITERATIONS; i++) {
    /* One instace of bootstraping */
    float ith_avg = 0;

    for (int j = 0; j < size; j++) {
      ith_avg += values[uniform_dist(e1)];
    }

    ith_avg = ith_avg / (size * 1.0);
    avg_values[i] = ith_avg;
  }

  /* Using accumulated probability */
  std::sort(avg_values.begin(), avg_values.end());
  int pos_min = floor(BOOTSTRAP_ITERATIONS * (alfa / 2.0));
  int pos_max =
      BOOTSTRAP_ITERATIONS - floor(BOOTSTRAP_ITERATIONS * (alfa / 2.0));

  *vmin = avg_values[pos_min];
  *vmax = avg_values[pos_max];
}

int main(int argc, char *argv[]) {
  /* ARGS description
    argv[0]: program bin
    argv[1]: file prefix (Ex: spC)
    argv[2]: number of files
  */

  // Common
  std::vector<float> times;
  float sum_times = 0;
  int count = 0;

  // get file prefix
  string prefix = string(argv[1]);

  // Evaluate each file - Search for time in each output
  for (int file_id = 1; file_id <= stoi(argv[2]); file_id++) {
    string file_path =
        string(prefix + string("-") + to_string(file_id) + string(".out"));
    std::ifstream infile(file_path);
    std::string line;

    while (std::getline(infile, line)) {
      std::size_t found = line.find(string("Time in seconds"));
      if (found != std::string::npos) {
        std::istringstream iss(line);
        string dummy;
        float time;
        if (iss >> dummy >> dummy >> dummy >> dummy >> time)
          // std::cout << "Time: " << time << "\n";
          times.push_back(time);
          sum_times += time;
          count++;
      }
    }
  }

  // // Calculations
  float time_avg = sum_times / count;
  float max, min;
  bootstrap(times, &min, &max, 0.05); // 95% confidence interval

  cout << fixed << std::setprecision(2) << time_avg << ", [" << min
       << "; " << max << "]\n";

  return 0;
}