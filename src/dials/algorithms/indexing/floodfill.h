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

int modulo(int i, int n) {
  return (i % n + n) % n;
}

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

  std::stack<scitbx::vec3<int>> stack;
  std::vector<std::vector<scitbx::vec3<int>>> accumulators;
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
      int x = i % n_points;
      int y = (i % n_sq) / n_points;
      int z = i / n_sq;
      scitbx::vec3<int> xyz = {x, y, z};
      stack.push(xyz);
      grid_binary[i] = replacement;
      std::vector<scitbx::vec3<int>> this_accumulator;
      accumulators.push_back(this_accumulator);
      n_voids++;
      grid_points_per_void.push_back(0);

      while (!stack.empty()) {
        scitbx::vec3<int> this_xyz = stack.top();
        stack.pop();
        accumulators[accumulator_index].push_back(this_xyz);
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

        // std::cout << "xyz " << x << " " << y << " " << z << std::endl;
        //  increment x and calculate the array index
        // std::cout << "xyz " << this_xyz[0] << " " << this_xyz[1] << " " <<
        // this_xyz[2] << std::endl;
        int x_plus = this_xyz[0] + 1;
        int modx = modulo(this_xyz[0], n_points);
        int mody = modulo(this_xyz[1], n_points) * n_points;
        int modz = modulo(this_xyz[2], n_points) * n_sq;
        int array_index = modulo(x_plus, n_points) + mody + modz;
        /// std::cout << "xplus idx " << array_index << std::endl;
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          scitbx::vec3<int> new_xyz = {x_plus, this_xyz[1], this_xyz[2]};
          stack.push(new_xyz);
        }
        int x_minus = this_xyz[0] - 1;
        array_index = modulo(x_minus, n_points) + mody + modz;
        /// std::cout << "xminus idx " << marray_index << std::endl;
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          scitbx::vec3<int> new_xyz = {x_minus, this_xyz[1], this_xyz[2]};
          stack.push(new_xyz);
        }

        int y_plus = this_xyz[1] + 1;
        array_index = modx + (modulo(y_plus, n_points) * n_points) + modz;
        /// std::cout << "xplus idx " << array_index << std::endl;
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          scitbx::vec3<int> new_xyz = {this_xyz[0], y_plus, this_xyz[2]};
          stack.push(new_xyz);
        }
        int y_minus = this_xyz[1] - 1;
        array_index = modx + (modulo(y_minus, n_points) * n_points) + modz;
        /// std::cout << "xminus idx " << marray_index << std::endl;
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          scitbx::vec3<int> new_xyz = {this_xyz[0], y_minus, this_xyz[2]};
          stack.push(new_xyz);
        }

        int z_plus = this_xyz[2] + 1;
        array_index = modx + mody + (modulo(z_plus, n_points) * n_sq);
        /// std::cout << "xplus idx " << array_index << std::endl;
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          scitbx::vec3<int> new_xyz = {this_xyz[0], this_xyz[1], z_plus};
          stack.push(new_xyz);
        }
        int z_minus = this_xyz[2] - 1;
        array_index = modx + mody + (modulo(z_minus, n_points) * n_sq);
        /// std::cout << "xminus idx " << marray_index << std::endl;
        if (grid_binary[array_index] == target) {
          grid_binary[array_index] = replacement;
          scitbx::vec3<int> new_xyz = {this_xyz[0], this_xyz[1], z_minus};
          stack.push(new_xyz);
        }
      }
      replacement++;
      accumulator_index++;
    }
  }
  scitbx::af::shared<scitbx::vec3<double>> centres_of_mass_frac(n_voids);
  for (int i = 0; i < accumulators.size(); i++) {
    std::vector<scitbx::vec3<int>> values = accumulators[i];
    int n = values.size();
    int divisor = n * n_points;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    for (int j = 0; j < n; j++) {
      /*std::cout << values[j] << " " << (values[j] % n_points) << " "
                << ((values[j] % n_sq) / n_points) << " " << (values[j] / n_sq)
                << std::endl;*/
      x += values[j][0];  // % n_points);
      y += values[j][1];  // % n_sq) / n_points);
      z += values[j][2];  // / n_sq);
    }
    x /= divisor;
    y /= divisor;
    z /= divisor;
    centres_of_mass_frac[i] = {z, y, x};
  }
  return std::make_tuple(grid_binary, grid_points_per_void, centres_of_mass_frac);
}