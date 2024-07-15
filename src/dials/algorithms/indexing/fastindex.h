#include <scitbx/array_family/flex_types.h>
#include <scitbx/vec3.h>
#include <scitbx/mat3.h>
#include <scitbx/math/utils.h>
#include <cctbx/miller.h>
#include <scitbx/constants.h>
#include <dials/array_family/scitbx_shared_and_versa.h>

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