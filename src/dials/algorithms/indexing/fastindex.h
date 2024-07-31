#include <scitbx/array_family/flex_types.h>
#include <scitbx/vec3.h>
#include <scitbx/mat3.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include "gemmi/third_party/pocketfft_hdronly.h"
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

using namespace pocketfft;

std::tuple<std::vector<std::complex<double>>, scitbx::af::shared<bool>>
map_centroids_to_reciprocal_space_grid_cpp(
  af::const_ref<scitbx::vec3<double>> const& reciprocal_space_vectors,
  double d_min,
  double b_iso = 0) {
  const int n_points = 256;
  const double rlgrid = 2 / (d_min * n_points);
  const double one_over_rlgrid = 1 / rlgrid;
  const int half_n_points = n_points / 2;
  scitbx::af::shared<bool> selection(reciprocal_space_vectors.size(), true);

  std::vector<std::complex<double>> data_in(256 * 256 * 256);
  for (int i = 0; i < reciprocal_space_vectors.size(); i++) {
    const scitbx::vec3<double> v = reciprocal_space_vectors[i];
    const double v_length = v.length();
    const double d_spacing = 1 / v_length;
    if (d_spacing < d_min) {
      selection[i] = false;
      continue;
    }
    scitbx::vec3<int> coord;
    for (int j = 0; j < 3; j++) {
      coord[j] = ((int)round(v[j] * one_over_rlgrid)) + half_n_points;
    }
    if ((coord.max() >= n_points) || coord.min() < 0) {
      selection[i] = false;
      continue;
    }
    double T;
    if (b_iso != 0) {
      T = std::exp(-b_iso * v_length * v_length / 4.0);
    } else {
      T = 1;
    }
    size_t index = coord[2] + (256 * coord[1]) + (256 * 256 * coord[0]);
    data_in[index] = {T, 0.0};
  }
  return std::make_tuple(data_in, selection);
}

class VectorGroup {
public:
  void add(scitbx::vec3<double> vec, int weight) {
    vectors.push_back(vec);
    weights.push_back(weight);
  }
  scitbx::vec3<double> mean() {
    int n = vectors.size();
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    for (const scitbx::vec3<double>& i : vectors) {
      sum_x += i[0];
      sum_y += i[1];
      sum_z += i[2];
    }
    scitbx::vec3<double> m = {sum_x / n, sum_y / n, sum_z / n};
    return m;
  }
  scitbx::af::shared<scitbx::vec3<double>> vectors{};
  std::vector<int> weights{};
};

struct SiteData {
  scitbx::vec3<double> site;
  double length;
  int volume;
};
bool compare_site_data(const SiteData& a, const SiteData& b) {
  return a.length < b.length;
}
bool compare_site_data_volume(const SiteData& a, const SiteData& b) {
  return a.volume > b.volume;
}

double vector_length(scitbx::vec3<double> v) {
  return std::pow(std::pow(v[0], 2) + std::pow(v[1], 2) + std::pow(v[2], 2), 0.5);
}

double angle_between_vectors_degrees(scitbx::vec3<double> v1, scitbx::vec3<double> v2) {
  double l1 = vector_length(v1);
  double l2 = vector_length(v2);
  double dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  double normdot = dot / (l1 * l2);
  if (std::abs(normdot - 1.0) < 1E-6) {
    return 0.0;
  }
  if (std::abs(normdot + 1.0) < 1E-6) {
    return 180.0;
  }
  double angle = std::acos(normdot) * 180.0 / M_PI;
  return angle;
}

bool is_approximate_integer_multiple(scitbx::vec3<double> v1,
                                     scitbx::vec3<double> v2,
                                     double relative_length_tolerance = 0.2,
                                     double angular_tolerance = 5.0) {
  double angle = angle_between_vectors_degrees(v1, v2);
  if ((angle < angular_tolerance) || (std::abs(180 - angle) < angular_tolerance)) {
    double l1 = vector_length(v1);
    double l2 = vector_length(v2);
    if (l1 > l2) {
      std::swap(l1, l2);
    }
    double n = l2 / l1;
    if (std::abs(std::round(n) - n) < relative_length_tolerance) {
      return true;
    }
  }
  return false;
}

scitbx::af::shared<scitbx::vec3<double>> sites_to_vecs(
  scitbx::af::shared<scitbx::vec3<double>> centres_of_mass_frac,
  std::vector<int> grid_points_per_void,
  double d_min,
  double min_cell = 3.0,
  double max_cell = 92.3) {
  auto start = std::chrono::system_clock::now();
  int n_points = 256;
  double fft_cell_length = n_points * d_min / 2.0;
  // sites_mod_short and convert to cartesian
  for (int i = 0; i < centres_of_mass_frac.size(); i++) {
    for (size_t j = 0; j < 3; j++) {
      if (centres_of_mass_frac[i][j] > 0.5) {
        centres_of_mass_frac[i][j]--;
      }
      centres_of_mass_frac[i][j] *= fft_cell_length;
    }
  }

  // now do some filtering
  std::vector<SiteData> filtered_data;
  for (int i = 0; i < centres_of_mass_frac.size(); i++) {
    auto v = centres_of_mass_frac[i];
    double length =
      std::pow(std::pow(v[0], 2) + std::pow(v[1], 2) + std::pow(v[2], 2), 0.5);
    if ((length > min_cell) && (length < 2 * max_cell)) {
      SiteData site_data = {centres_of_mass_frac[i], length, grid_points_per_void[i]};
      filtered_data.push_back(site_data);
    }
  }
  // now sort filtered data

  // need to sort volumes and sites by length for group_vectors, and also filter by max
  // and min cell
  std::sort(filtered_data.begin(), filtered_data.end(), compare_site_data);

  // now 'group vectors'
  double relative_length_tolerance = 0.1;
  double angular_tolerance = 5.0;
  std::vector<VectorGroup> vector_groups{};
  for (int i = 0; i < filtered_data.size(); i++) {
    bool matched_group = false;
    double length = filtered_data[i].length;
    for (int j = 0; j < vector_groups.size(); j++) {
      scitbx::vec3<double> mean_v = vector_groups[j].mean();
      double mean_v_length = vector_length(mean_v);
      if ((std::abs(mean_v_length - length) / std::max(mean_v_length, length))
          < relative_length_tolerance) {
        double angle = angle_between_vectors_degrees(mean_v, filtered_data[i].site);
        if (angle < angular_tolerance) {
          vector_groups[j].add(filtered_data[i].site, filtered_data[i].volume);
          matched_group = true;
          break;
        } else if (std::abs(180 - angle) < angular_tolerance) {
          vector_groups[j].add(-1.0 * filtered_data[i].site, filtered_data[i].volume);
          matched_group = true;
          break;
        }
      }
    }
    if (!matched_group) {
      VectorGroup group = VectorGroup();
      group.add(filtered_data[i].site, filtered_data[i].volume);
      vector_groups.push_back(group);
    }
  }
  std::vector<SiteData> grouped_data;
  for (int i = 0; i < vector_groups.size(); i++) {
    scitbx::vec3<double> site = vector_groups[i].mean();
    int max = *std::max_element(vector_groups[i].weights.begin(),
                                vector_groups[i].weights.end());
    SiteData site_data = {site, vector_length(site), max};
    grouped_data.push_back(site_data);
  }
  std::sort(grouped_data.begin(), grouped_data.end(), compare_site_data_volume);
  std::sort(grouped_data.begin(), grouped_data.end(), compare_site_data);

  // scitbx::af::shared<scitbx::vec3<double>> unique_vectors;
  // std::vector<int> unique_volumes;
  std::vector<SiteData> unique_sites;
  for (int i = 0; i < grouped_data.size(); i++) {
    bool is_unique = true;
    scitbx::vec3<double> v = grouped_data[i].site;
    for (int j = 0; j < unique_sites.size(); j++) {
      if (unique_sites[j].volume > grouped_data[i].volume) {
        if (is_approximate_integer_multiple(unique_sites[j].site, v)) {
          std::cout << "rejecting " << vector_length(v) << ": is integer multiple of "
                    << vector_length(unique_sites[j].site) << std::endl;
          is_unique = false;
          break;
        }
      }
    }
    if (is_unique) {
      // std::cout << v[0] << " " << v[1] << " " << v[2] << std::endl;
      // unique_vectors.push_back(v);
      // unique_volumes.push_back(grouped_data[i].volume);
      SiteData site{v, 1.0, grouped_data[i].volume};
      unique_sites.push_back(site);
    }
  }
  // now sort by peak volume again
  std::sort(unique_sites.begin(), unique_sites.end(), compare_site_data_volume);
  scitbx::af::shared<scitbx::vec3<double>> unique_vectors_sorted;
  for (int i = 0; i < unique_sites.size(); i++) {
    unique_vectors_sorted.push_back(unique_sites[i].site);
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time for sites_to_vecs: " << elapsed_seconds.count() << "s"
            << std::endl;
  return unique_vectors_sorted;
}

scitbx::af::shared<scitbx::vec3<double>> do_floodfill(scitbx::af::shared<double> grid,
                                                      double rmsd_cutoff = 15.0,
                                                      double peak_volume_cutoff = 0.15,
                                                      double d_min = 1.8) {
  auto start = std::chrono::system_clock::now();
  int n_points = 256;
  // double fft_cell_length = n_points * d_min / 2;
  // first calc rmsd and use this to create a binary grid
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
  scitbx::af::shared<int> grid_binary(256 * 256 * 256, 0);
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
      stack.push(i);
      grid_binary[i] = replacement;
      std::vector<int> this_accumulator;
      accumulators.push_back(this_accumulator);
      n_voids++;
      grid_points_per_void.push_back(0);
      while (!stack.empty()) {
        int index = stack.top();
        stack.pop();
        accumulators[accumulator_index].push_back(index);
        grid_points_per_void[accumulator_index]++;
        // when finding nearest neighbours, need to check we don't step over the edge in
        // each dimension likely not very efficient right now!
        if ((index + 1) % n_points != 0) {
          int x_plus = index + 1;
          if (grid_binary[x_plus] == target) {
            grid_binary[x_plus] = replacement;
            stack.push(x_plus);
          }
        }
        if (index % n_points != 0) {
          int x_minus = index - 1;
          if (grid_binary[x_minus] == target) {
            grid_binary[x_minus] = replacement;
            stack.push(x_minus);
          }
        }
        if ((index % n_sq) < n_sq_minus_n) {
          int y_plus = index + n_points;
          if (grid_binary[y_plus] == target) {
            grid_binary[y_plus] = replacement;
            stack.push(y_plus);
          }
        }
        if ((index % n_sq) >= n_points) {
          int y_minus = index - n_points;
          if (grid_binary[y_minus] == target) {
            grid_binary[y_minus] = replacement;
            stack.push(y_minus);
          }
        }
        if ((index % total) < nn_sq_minus_n) {
          int z_plus = index + n_sq;
          if (grid_binary[z_plus] == target) {
            grid_binary[z_plus] = replacement;
            stack.push(z_plus);
          }
        }
        if ((index % total) >= n_sq) {
          int z_minus = index - n_sq;
          if (grid_binary[z_minus] == target) {
            grid_binary[z_minus] = replacement;
            stack.push(z_minus);
          }
        }
      }
      replacement++;
      accumulator_index++;
    }
  }

  // DIALS_ASSERT(n_voids > 3)

  // want centres_of_mass_frac and grid_points_per_void
  // now calc centres of mass.
  scitbx::af::shared<scitbx::vec3<double>> centres_of_mass_frac(n_voids);
  for (int i = 0; i < accumulators.size(); i++) {
    std::vector<int> values = accumulators[i];
    int n = values.size();
    int divisor = n * n_points;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    for (int j = 0; j < n; j++) {
      x += (values[j] % n_points);
      y += ((values[j] % n_sq) / n_points);
      z += (values[j] / n_sq);
    }
    x /= divisor;
    y /= divisor;
    z /= divisor;
    centres_of_mass_frac[i] = {z, y, x};
  }

  // now filter out based on iqr range and peak_volume_cutoff
  std::vector<int> grid_points_per_void_unsorted(grid_points_per_void);
  std::sort(grid_points_per_void.begin(), grid_points_per_void.end());
  int Q3_index = grid_points_per_void.size() * 3 / 4;
  int Q1_index = grid_points_per_void.size() / 4;
  int iqr = grid_points_per_void[Q3_index] - grid_points_per_void[Q1_index];
  int iqr_multiplier = 5;
  int cut = (iqr * iqr_multiplier) + grid_points_per_void[Q3_index];
  /*for (int i = grid_points_per_void.size() - 1; i >= 0; i--) {
    if (grid_points_per_void_unsorted[i] > cut) {
      grid_points_per_void_unsorted.erase(grid_points_per_void_unsorted.begin() + i);
      centres_of_mass_frac.erase(centres_of_mass_frac.begin() + i);
    }
  }*/
  while (grid_points_per_void[grid_points_per_void.size() - 1] > cut) {
    grid_points_per_void.pop_back();
  }
  int max_val = grid_points_per_void[grid_points_per_void.size() - 1];

  int peak_cutoff = (int)(peak_volume_cutoff * max_val);
  for (int i = grid_points_per_void_unsorted.size() - 1; i >= 0; i--) {
    if (grid_points_per_void_unsorted[i] <= peak_cutoff) {
      grid_points_per_void_unsorted.erase(grid_points_per_void_unsorted.begin() + i);
      centres_of_mass_frac.erase(centres_of_mass_frac.begin() + i);
    }
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time for flood_fill: " << elapsed_seconds.count() << "s"
            << std::endl;
  // for (int i=0;i<centres_of_mass_frac.size();i++){
  //   std::cout << centres_of_mass_frac[i][0] << " " << centres_of_mass_frac[i][1] << "
  //   " << centres_of_mass_frac[i][2] << " " << std::endl;
  // }
  scitbx::af::shared<scitbx::vec3<double>> sites =
    sites_to_vecs(centres_of_mass_frac, grid_points_per_void_unsorted, d_min);
  return sites;
}

std::tuple<scitbx::af::shared<double>, scitbx::af::shared<bool>> do_fft3d(
  af::const_ref<scitbx::vec3<double>> const& reciprocal_space_vectors,
  double d_min,
  double b_iso = 0) {
  auto start = std::chrono::system_clock::now();
  std::vector<std::complex<double>> complex_data_in;
  scitbx::af::shared<bool> used_in_indexing;
  std::tie(complex_data_in, used_in_indexing) =
    map_centroids_to_reciprocal_space_grid_cpp(reciprocal_space_vectors, d_min, b_iso);

  shape_t shape_in{256, 256, 256};
  stride_t stride_in{sizeof(std::complex<double>),
                     sizeof(std::complex<double>) * 256,
                     sizeof(std::complex<double>) * 256
                       * 256};  // must have the size of each element. Must have
                                // size() equal to shape_in.size()
  stride_t stride_out{sizeof(std::complex<double>),
                      sizeof(std::complex<double>) * 256,
                      sizeof(std::complex<double>) * 256
                        * 256};  // must have the size of each element. Must
                                 // have size() equal to shape_in.size()
  shape_t axes{0, 1, 2};         // 0 to shape.size()-1 inclusive
  bool forward{FORWARD};
  std::vector<std::complex<double>> data_out(256 * 256 * 256);
  double fct{1.0f};
  size_t nthreads = 0;  // use all threads available
  c2c(shape_in,
      stride_in,
      stride_out,
      axes,
      forward,
      complex_data_in.data(),
      data_out.data(),
      fct,
      nthreads);
  scitbx::af::shared<double> real_out(256 * 256 * 256);
  for (int i = 0; i < real_out.size(); ++i) {
    real_out[i] = std::pow(data_out[i].real(), 2);
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time for fft3d (pocketfft): " << elapsed_seconds.count() << "s"
            << std::endl;
  return std::make_tuple(real_out, used_in_indexing);
}

boost::python::tuple indexing_algorithm(
  af::const_ref<scitbx::vec3<double>> const& reciprocal_space_vectors,
  double d_min,
  double b_iso = 0) {
  scitbx::af::shared<double> real_fft;
  scitbx::af::shared<bool> used_in_indexing;
  std::tie(real_fft, used_in_indexing) =
    do_fft3d(reciprocal_space_vectors, d_min, b_iso);
  scitbx::af::shared<scitbx::vec3<double>> candidate_vecs =
    do_floodfill(real_fft, 15.0, 0.15, d_min);
  return boost::python::make_tuple(candidate_vecs, used_in_indexing);
}
