#include <scitbx/array_family/flex_types.h>
#include <scitbx/vec3.h>
#include <scitbx/mat3.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <map>
#include <stack>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <math.h>

#define _USE_MATH_DEFINES
#include <cmath>

std::tuple<scitbx::af::shared<int>,
           std::vector<int>,
           scitbx::af::shared<scitbx::vec3<double>>>
flood_fill(scitbx::af::shared<double> grid,
           double rmsd_cutoff = 15.0,
           int n_points = 256) {
  auto start = std::chrono::system_clock::now();
  // int n_points = 256;
  //  double fft_cell_length = n_points * d_min / 2;
  //  first calc rmsd and use this to create a binary grid
  double sumg = 0.0;
  for (int i = 0; i < grid.size(); ++i) {
    sumg += grid[i];
  }
  double meang = sumg / grid.size();
  double sum_delta_sq = 0.0;
  for (int i = 0; i < grid.size(); ++i) {
    sum_delta_sq += std::pow(grid[i] - meang, 2);
  }
  double rmsd = std::pow(sum_delta_sq / grid.size(), 0.5);
  scitbx::af::shared<int> grid_binary(n_points * n_points * n_points, 0);
  double cutoff = rmsd_cutoff * rmsd;

  for (int i = 0; i < grid.size(); i++) {
    if (grid[i] >= cutoff) {
      grid_binary[i] = 1;
    }
  }

  // now do flood fill
  int n_voids = 0;

  std::stack<int> stack;
  std::vector<std::vector<int>> accumulators;
  int target = 1;
  int replacement = 2;
  std::vector<int> grid_points_per_void;
  int accumulator_index = 0;
  int total = n_points * n_points * n_points;
  int n_sq = n_points * n_points;
  int n_sq_minus_n = n_points * (n_points - 1);
  int nn_sq_minus_n = n_points * n_points * (n_points - 1);

  for (int i = 0; i < grid_binary.size(); i++) {
    if (grid_binary[i] == target) {
      // std::cout << i << std::endl;
      stack.push(i);
      grid_binary[i] = replacement;
      std::vector<int> this_accumulator;
      accumulators.push_back(this_accumulator);
      n_voids++;
      grid_points_per_void.push_back(0);
      int x_plus = 0;
      int x_minus = 0;
      int y_plus = 0;
      int y_minus = 0;
      int z_plus = 0;
      int z_minus = 0;
      while (!stack.empty()) {
        int index = stack.top();
        stack.pop();
        accumulators[accumulator_index].push_back(index);
        grid_points_per_void[accumulator_index]++;
        // when finding nearest neighbours, need to check we don't step over the edge in
        // each dimension likely not very efficient right now!
        // std::cout << "index " << index << std::endl;
        /*while (index < 0){
            index += total;
        }
        while (index >= total) {
            index -= total;
        }*/
        if ((index + 1) % n_points != 0) {  // if not on top boundary
          x_plus = index + 1;
        } else {
          x_plus = index + 1 - n_points;
        }
        std::cout << x_plus << std::endl;
        if (grid_binary[x_plus] == target) {
          grid_binary[x_plus] = replacement;
          stack.push(index + 1);
        }
        //}
        if (index % n_points != 0) {
          x_minus = index - 1;
        } else {
          x_minus = index - 1 + n_points;
        }
        std::cout << x_minus << std::endl;
        if (grid_binary[x_minus] == target) {
          grid_binary[x_minus] = replacement;
          stack.push(index - 1);
        }
        //}
        if ((index % n_sq) < n_sq_minus_n) {
          y_plus = index + n_points;
        } else {
          y_plus = index + n_points - n_sq;
        }
        if (grid_binary[y_plus] == target) {
          grid_binary[y_plus] = replacement;
          stack.push(index + n_points);
        }
        if ((index % n_sq) >= n_points) {
          y_minus = index - n_points;
        } else {
          y_minus = index - n_points + n_sq;
        }
        if (grid_binary[y_minus] == target) {
          grid_binary[y_minus] = replacement;
          stack.push(index - n_points);
        }
        if ((index % total) < nn_sq_minus_n) {
          z_plus = index + n_sq;
        } else {
          z_plus = index + n_sq - total;
        }
        if (grid_binary[z_plus] == target) {
          grid_binary[z_plus] = replacement;
          stack.push(index + n_sq);
        }
        if ((index % total) >= n_sq) {
          z_minus = index - n_sq;
        } else {
          z_minus = index - n_sq + total;
        }
        if (grid_binary[z_minus] == target) {
          grid_binary[z_minus] = replacement;
          stack.push(index - n_sq);
        }
      }
      replacement++;
      accumulator_index++;
    }
  }
  scitbx::af::shared<scitbx::vec3<double>> centres_of_mass_frac(n_voids);
  for (int i = 0; i < accumulators.size(); i++) {
    std::vector<int> values = accumulators[i];
    int n = values.size();
    int divisor = n;  // * n_points;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    for (int j = 0; j < n; j++) {
      std::cout << values[j] << " " << (values[j] % n_points) << " "
                << ((values[j] % n_sq) / n_points) << " " << (values[j] / n_sq)
                << std::endl;
      x += (values[j] % n_points);
      y += ((values[j] % n_sq) / n_points);
      z += (values[j] / n_sq);
    }
    x /= divisor;
    y /= divisor;
    z /= divisor;
    centres_of_mass_frac[i] = {z, y, x};
  }
  return std::make_tuple(grid_binary, grid_points_per_void, centres_of_mass_frac);
}