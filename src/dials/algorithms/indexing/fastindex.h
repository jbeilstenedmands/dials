#include <scitbx/array_family/flex_types.h>
#include <scitbx/vec3.h>
#include <scitbx/mat3.h>
#include <scitbx/math/utils.h>
#include <cctbx/miller.h>
#include <scitbx/constants.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include "gemmi/third_party/pocketfft_hdronly.h"

class SimpleBeam {
public:
  double wavelength;
  scitbx::vec3<double> s0;

  SimpleBeam(double wavelength) {
    this->wavelength = wavelength;
    s0 = {0.0, 0.0, -1.0 / wavelength};
  }
};

class SimpleDetector {
public:
  scitbx::mat3<double> d_matrix;
  double pixel_size;  // in mm

  SimpleDetector(scitbx::mat3<double> d_matrix, double pixel_size) {
    this->d_matrix = d_matrix;
    this->pixel_size = pixel_size;
  }
};

class SimpleScan {
public:
  int image_range_start;
  double osc_start;
  double osc_width;

  SimpleScan(int image_range_start, double osc_start, double osc_width) {
    this->image_range_start = image_range_start;
    this->osc_start = osc_start;
    this->osc_width = osc_width;
  }
};

class SimpleGonio {
public:
  scitbx::mat3<double> sample_rotation;
  scitbx::vec3<double> rotation_axis;
  scitbx::mat3<double> setting_rotation;
  scitbx::mat3<double> sample_rotation_inverse;
  scitbx::mat3<double> setting_rotation_inverse;

  SimpleGonio(scitbx::mat3<double> sample_rotation,
              scitbx::vec3<double> rotation_axis,
              scitbx::mat3<double> setting_rotation) {
    this->sample_rotation = sample_rotation;
    this->rotation_axis = rotation_axis;
    this->setting_rotation = setting_rotation;
    sample_rotation_inverse = sample_rotation.inverse();
    setting_rotation_inverse = setting_rotation.inverse();
  }
};

scitbx::af::shared<scitbx::vec3<double>> xyz_to_rlp(
  scitbx::af::shared<scitbx::vec3<double>> xyzobs_px,
  scitbx::mat3<double> sample_rotation,
  scitbx::mat3<double> detector_d_matrix) {
  SimpleBeam beam(1.23985);
  SimpleDetector detector(detector_d_matrix, 0.172);
  SimpleScan scan(1, 0.0, 0.5);
  SimpleGonio gonio(
    sample_rotation, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});

  float DEG2RAD = scitbx::constants::pi / 180.0;

  scitbx::af::shared<scitbx::vec3<double>> rlp(xyzobs_px.size());
  for (int i = 0; i < xyzobs_px.size(); ++i) {
    scitbx::vec3<double> xyzobs_i = xyzobs_px[i];
    double x_mm = xyzobs_i[0] * detector.pixel_size;
    double y_mm = xyzobs_i[1] * detector.pixel_size;
    double rot_angle =
      (((xyzobs_i[2] + 1 - scan.image_range_start) * scan.osc_width) + scan.osc_start)
      * DEG2RAD;
    scitbx::vec3<double> m = {x_mm, y_mm, 1.0};
    scitbx::vec3<double> s1_i = detector.d_matrix * m;
    double norm =
      std::pow(std::pow(s1_i[0], 2) + std::pow(s1_i[1], 2) + std::pow(s1_i[2], 2), 0.5);
    scitbx::vec3<double> s1_this = (s1_i / norm) * (1.0 / beam.wavelength);
    scitbx::vec3<double> S = gonio.setting_rotation_inverse * (s1_this - beam.s0);
    scitbx::vec3<double> rlp_this =
      S.rotate_around_origin(gonio.rotation_axis, -1.0 * rot_angle);
    rlp[i] = gonio.sample_rotation_inverse * rlp_this;
  }
  return rlp;
}

using namespace pocketfft;

std::vector<std::complex<double>> map_centroids_to_reciprocal_space_grid_cpp(
  af::const_ref<scitbx::vec3<double>> const& reciprocal_space_vectors,
  double d_min,
  double b_iso = 0) {
  const int n_points = 256;
  const double rlgrid = 2 / (d_min * n_points);
  const double one_over_rlgrid = 1 / rlgrid;
  const int half_n_points = n_points / 2;

  std::vector<std::complex<double>> data_in(256 * 256 * 256);
  for (int i = 0; i < reciprocal_space_vectors.size(); i++) {
    const scitbx::vec3<double> v = reciprocal_space_vectors[i];
    const double v_length = v.length();
    const double d_spacing = 1 / v_length;
    if (d_spacing < d_min) {
      // selection[i] = false;
      continue;
    }
    scitbx::vec3<int> coord;
    for (int j = 0; j < 3; j++) {
      coord[j] = scitbx::math::iround(v[j] * one_over_rlgrid) + half_n_points;
    }
    if ((coord.max() >= n_points) || coord.min() < 0) {
      // selection[i] = false;
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
  return data_in;
}

scitbx::af::shared<double> do_fft3d(
  af::const_ref<scitbx::vec3<double>> const& reciprocal_space_vectors,
  double d_min,
  double b_iso = 0) {
  std::vector<std::complex<double>> complex_data_in =
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
  return real_out;
}