#ifndef DIALS_ALGORITHMS_UBPARAMETERISATION_H
#define DIALS_ALGORITHMS_UBPARAMETERISATION_H

#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <scitbx/mat2.h>
#include <scitbx/mat3.h>
#include <cctbx/sgtbx/space_group.h>
#include <cctbx/sgtbx/tensor_rank_2.h>
#include <cctbx/crystal_orientation.h>
#include <rstbx/symmetry/constraints/a_g_conversion.h>
#include <dxtbx/model/experiment.h>

class SimpleUParameterisation {
public:
  SimpleUParameterisation(const dxtbx::model::Crystal &crystal);
  scitbx::af::shared<double> get_params();
  void set_params(scitbx::af::shared<double> p);
  scitbx::mat3<double> get_state();
  scitbx::af::shared<scitbx::mat3<double>> get_dS_dp();

private:
  scitbx::af::shared<double> params{3, 0.0};
  scitbx::af::shared<scitbx::vec3<double>> axes{3, scitbx::vec3<double>(1.0, 0.0, 0.0)};
  void compose();
  scitbx::mat3<double> istate{};
  scitbx::mat3<double> U_{};
  scitbx::af::shared<scitbx::mat3<double>> dS_dp{
    3,
    scitbx::mat3<double>(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)};
};

class SymmetrizeReduceEnlarge {
public:
  SymmetrizeReduceEnlarge(cctbx::sgtbx::space_group space_group);
  void set_orientation(scitbx::mat3<double> B);
  scitbx::af::small<double, 6> forward_independent_parameters();
  cctbx::crystal_orientation backward_orientation(
    scitbx::af::small<double, 6> independent);
  scitbx::af::shared<scitbx::mat3<double>> forward_gradients();

private:
  cctbx::sgtbx::space_group space_group_;
  cctbx::sgtbx::tensor_rank_2::constraints<double> constraints_;
  cctbx::crystal_orientation orientation_{};
  rstbx::symmetry::AG Bconverter{};
};

class SimpleCellParameterisation {
public:
  SimpleCellParameterisation(const dxtbx::model::Crystal &crystal);
  scitbx::af::shared<double> get_params();
  void set_params(scitbx::af::shared<double>);
  scitbx::mat3<double> get_state();
  scitbx::af::shared<scitbx::mat3<double>> get_dS_dp();

private:
  scitbx::af::shared<double> params;
  void compose();
  scitbx::mat3<double> B_{};
  scitbx::af::shared<scitbx::mat3<double>> dS_dp{};
  SymmetrizeReduceEnlarge SRE;
};

using namespace boost::python;

namespace dials { namespace algorithms { namespace boost_python {

  BOOST_PYTHON_MODULE(dials_algorithms_profile_model_ellipsoid_UBparameterisation_ext) {
    class_<SimpleUParameterisation>("SimpleUParameterisation", no_init)
      .def(init<dxtbx::model::Crystal>())
      .def("get_params", &SimpleUParameterisation::get_params)
      .def("set_params", &SimpleUParameterisation::set_params)
      .def("get_state", &SimpleUParameterisation::get_state)
      .def("get_dS_dp", &SimpleUParameterisation::get_dS_dp);

    class_<SimpleCellParameterisation>("SimpleCellParameterisation", no_init)
      .def(init<dxtbx::model::Crystal>())
      .def("get_params", &SimpleCellParameterisation::get_params)
      .def("set_params", &SimpleCellParameterisation::set_params)
      .def("get_state", &SimpleCellParameterisation::get_state)
      .def("get_dS_dp", &SimpleCellParameterisation::get_dS_dp);
  }

}}}  // namespace dials::algorithms::boost_python

#endif  // DIALS_ALGORITHMS_UBPARAMETERISATION_H