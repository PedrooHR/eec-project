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
  std::vector<float> package, dram, psys, pp0, thermal;
  float sums[5] = {0, 0, 0, 0, 0};
  int counters[5] = {0, 0, 0, 0, 0};

  // get file prefix
  string prefix = string(argv[1]);

  // Evaluate each file - Search for time in each output
  for (int file_id = 1; file_id <= stoi(argv[2]); file_id++) {
    string file_path =
        string(prefix + string("-") + to_string(file_id) + string(".log"));
    std::ifstream infile(file_path);
    std::string line;

    while (std::getline(infile, line)) {
      std::size_t found;
      std::istringstream iss(line);
      string dummy;
      float value;
      /* First Case: PACKAGE_ENERGY */
      found = line.find(string("PACKAGE_ENERGY"));
      if (found != std::string::npos) {
        if (iss >> dummy >> dummy >> dummy >> dummy >> dummy >> value) {
          // std::cout << "PACKAGE_ENERGY: " << value << "\n";
          package.push_back(value);
          sums[0] += value;
          counters[0]++;
        }
      }
      /* Second Case: DRAM_ENERGY */
      found = line.find(string("DRAM_ENERGY"));
      if (found != std::string::npos) {
        if (iss >> dummy >> dummy >> dummy >> dummy >> dummy >> value) {
          // std::cout << "DRAM_ENERGY: " << value << "\n";
          dram.push_back(value);
          sums[1] += value;
          counters[1]++;
        }
      }
      /* Third Case: PSYS_ENERGY */
      found = line.find(string("PSYS_ENERGY"));
      if (found != std::string::npos) {
        if (iss >> dummy >> dummy >> dummy >> dummy >> dummy >> value) {
          // std::cout << "PSYS_ENERGY: " << value << "\n";
          psys.push_back(value);
          sums[2] += value;
          counters[2]++;
        }
      }
      /* Fourth Case: PP0_ENERGY */
      found = line.find(string("PP0_ENERGY"));
      if (found != std::string::npos) {
        if (iss >> dummy >> dummy >> dummy >> dummy >> dummy >> value) {
          // std::cout << "PP0_ENERGY: " << value << "\n";
          pp0.push_back(value);
          sums[3] += value;
          counters[3]++;
        }
      }
      /* Fifth Case: THERMAL_SPEC */
      found = line.find(string("THERMAL_SPEC"));
      if (found != std::string::npos) {
        if (iss >> dummy >> dummy >> dummy >> dummy >> dummy >> value) {
          // std::cout << "THERMAL_SPEC: " << value << "\n";
          thermal.push_back(value);
          sums[4] += value;
          counters[4]++;
        }
      }
    }
  }

  // Calculations
  std::vector<std::vector<float>> energy = {package, dram, psys, pp0, thermal};
  float avgs[5], max[5], min[5];
  for (int i = 0; i < 4; i++) {
    avgs[i] = sums[i] / counters[i];
    bootstrap(energy[i], &min[i], &max[i], 0.05); // 95% confidence interval
    std::cout << fixed << std::setprecision(2);
    std::cout << avgs[i] << ", [" << min[i] << "; " << max[i] << "]\n"; 
  }

  return 0;
}