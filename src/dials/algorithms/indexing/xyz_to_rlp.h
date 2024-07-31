#include <scitbx/array_family/flex_types.h>
#include <scitbx/vec3.h>
#include <scitbx/mat3.h>
#include <scitbx/constants.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <eigen3/Eigen/Dense>
#include <chrono>

using Eigen::Matrix3d;
using Eigen::Vector3d;

class SimpleBeam {
public:
  double wavelength;
  Vector3d s0;

  SimpleBeam(double wavelength) {
    this->wavelength = wavelength;
    s0 = {0.0, 0.0, -1.0 / wavelength};
  }
};

class SimpleDetector {
public:
  Matrix3d d_matrix;
  double pixel_size;  // in mm

  SimpleDetector(scitbx::mat3<double> d_matrix, double pixel_size) {
    Matrix3d m{{d_matrix[0], d_matrix[1], d_matrix[2]},
               {d_matrix[3], d_matrix[4], d_matrix[5]},
               {d_matrix[6], d_matrix[7], d_matrix[8]}};
    this->d_matrix = m;
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
  Matrix3d sample_rotation;
  Vector3d rotation_axis;
  Matrix3d setting_rotation;
  Matrix3d sample_rotation_inverse;
  Matrix3d setting_rotation_inverse;

  SimpleGonio(scitbx::mat3<double> sample_rotation,
              scitbx::vec3<double> rotation_axis,
              scitbx::mat3<double> setting_rotation) {
    Matrix3d rotation{{sample_rotation[0], sample_rotation[1], sample_rotation[2]},
                      {sample_rotation[3], sample_rotation[4], sample_rotation[5]},
                      {sample_rotation[6], sample_rotation[7], sample_rotation[8]}};
    this->sample_rotation = rotation;
    Vector3d rot_axis{rotation_axis[0], rotation_axis[1], rotation_axis[2]};
    rot_axis.normalize();
    this->rotation_axis = rot_axis;
    Matrix3d setting{{setting_rotation[0], setting_rotation[1], setting_rotation[2]},
                     {setting_rotation[3], setting_rotation[4], setting_rotation[5]},
                     {setting_rotation[6], setting_rotation[7], setting_rotation[8]}};
    this->setting_rotation = setting;
    sample_rotation_inverse = this->sample_rotation.inverse();
    setting_rotation_inverse = this->setting_rotation.inverse();
  }
};

scitbx::af::shared<scitbx::vec3<double>> xyz_to_rlp(
  scitbx::af::shared<scitbx::vec3<double>> xyzobs_px,
  scitbx::mat3<double> sample_rotation,
  scitbx::mat3<double> detector_d_matrix,
  double wavelength,
  double pixel_size_mm,
  int image_range_start,
  double osc_start,
  double osc_width,
  scitbx::vec3<double> rotation_axis,
  scitbx::mat3<double> setting_rotation) {
  auto start = std::chrono::system_clock::now();
  // An equivalent to dials flex_ext.map_centroids_to_reciprocal_space method
  SimpleBeam beam(wavelength);
  SimpleDetector detector(detector_d_matrix, pixel_size_mm);
  SimpleScan scan(image_range_start, osc_start, osc_width);
  SimpleGonio gonio(sample_rotation, rotation_axis, setting_rotation);

  float DEG2RAD = scitbx::constants::pi / 180.0;

  Matrix3d m;

  scitbx::af::shared<scitbx::vec3<double>> rlp(xyzobs_px.size());
  for (int i = 0; i < xyzobs_px.size(); ++i) {
    // first convert detector pixel positions into mm
    double x1 = xyzobs_px[i][0];
    double x2 = xyzobs_px[i][1];
    double x3 = xyzobs_px[i][2];
    double x_mm = x1 * detector.pixel_size;
    double y_mm = x2 * detector.pixel_size;
    // convert the image 'z' coordinate to rotation angle based on the scan data
    double rot_angle =
      (((x3 + 1 - scan.image_range_start) * scan.osc_width) + scan.osc_start) * DEG2RAD;
    // calculate the s1 vector using the detector d matrix
    Vector3d m = {x_mm, y_mm, 1.0};
    Vector3d s1_i = detector.d_matrix * m;
    s1_i.normalize();
    // convert into inverse ansgtroms
    Vector3d s1_this = s1_i / beam.wavelength;
    // now apply the goniometer matrices
    // see https://dials.github.io/documentation/conventions.html for full conventions
    // rlp = F^-1 * R'^-1 * S^-1 * (s1-s0)
    Vector3d S = gonio.setting_rotation_inverse * (s1_this - beam.s0);
    double cos = std::cos(-1.0 * rot_angle);
    double sin = std::sin(-1.0 * rot_angle);
    Vector3d rlp_this = (S * cos)
                        + (gonio.rotation_axis * gonio.rotation_axis.dot(S) * (1 - cos))
                        + (sin * gonio.rotation_axis.cross(S));
    // lp_this = S.rotate_around_origin(gonio.rotation_axis, -1.0 * rot_angle);
    rlp_this = gonio.sample_rotation_inverse * rlp_this;
    rlp[i] = {rlp_this[0], rlp_this[1], rlp_this[2]};
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time for xyz_to_rlp: " << elapsed_seconds.count() << "s"
            << std::endl;
  return rlp;
}