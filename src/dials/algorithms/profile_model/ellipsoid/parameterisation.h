#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <scitbx/mat2.h>
#include <scitbx/mat3.h>
#include <dxtbx/model/panel.h>
#include <dials/array_family/reflection_table.h>
#include <dxtbx/model/experiment.h>
#include <dials/algorithms/refinement/parameterisation/parameterisation_helpers.h>
#include <map>
#include <string>
#include <cctbx/sgtbx/space_group.h>
#include <cctbx/sgtbx/tensor_rank_2.h>
#include <cctbx/crystal_orientation.h>
#include <cctbx/crystal_orientation.h>
#include <rstbx/symmetry/constraints/a_g_conversion.h>

class BaseParameterisation {
public:
  BaseParameterisation(scitbx::af::shared<double> parameters);
  scitbx::af::shared<double> get_params();
  int num_parameters();
  void set_params(scitbx::af::shared<double> parameters);

private:
  scitbx::af::shared<double> parameters;
};

class Simple1MosaicityParameterisation : public BaseParameterisation {
public:
  Simple1MosaicityParameterisation(scitbx::af::shared<double> parameters);
  bool is_angular();
  int num_parameters();
  scitbx::mat3<double> sigma();
  scitbx::af::shared<scitbx::mat3<double>> first_derivatives();
  std::map<std::string, double> mosaicity();
};

class SimpleUParameterisation {
public:
  SimpleUParameterisation(const dxtbx::model::Crystal &crystal);
  scitbx::af::shared<double> get_params();
  void set_params(scitbx::af::shared<double>);
  scitbx::mat3<double> get_state();
  scitbx::af::shared<scitbx::mat3<double>> get_dS_dp();

private:
  scitbx::af::shared<double> params{3, 0.0};
  scitbx::af::shared<scitbx::vec3<double>> axes{3, {1.0, 0.0, 0.0}};
  void compose();
  scitbx::mat3<double> istate;
  scitbx::mat3<double> U_;
  scitbx::af::shared<scitbx::mat3<double>> dS_dp{3, {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}};
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
  cctbx::crystal_orientation orientation_;
  rstbx::symmetry::AG Bconverter;
};

class SimpleCellParameterisation {
public:
  SimpleCellParameterisation(const dxtbx::model::Crystal &crystal);
  scitbx::af::small<double, 6> get_params();
  void set_params(scitbx::af::small<double, 6>);
  scitbx::mat3<double> get_state();
  scitbx::af::shared<scitbx::mat3<double>> get_dS_dp();

private:
  scitbx::af::small<double, 6> params;
  void compose();
  scitbx::mat3<double> B_;
  scitbx::af::shared<scitbx::mat3<double>> dS_dp;
  SymmetrizeReduceEnlarge SRE;
};

/*class ModelState {
public:
  ModelState(dxtbx::model::Crystal &crystal,
             BaseParameterisation parameterisation,
             bool fix_orientation,
             bool fix_unit_cell
             bool fix_wavelength_spread,
             bool fix_mosaic_spread);

private:
  dxtbx::model::Crystal crystal;
  BaseParameterisation parameterisation;
  bool fix_orientation;
  bool fix_unit_cell;
  bool fix_wavelength_spread;
  bool fix_mosaic_spread;
};*/

using namespace boost::python;

namespace dials { namespace algorithms { namespace boost_python {

  BOOST_PYTHON_MODULE(dials_algorithms_profile_model_ellipsoid_parameterisation_ext) {
    class_<Simple1MosaicityParameterisation>("Simple1MosaicityParameterisation",
                                             no_init)
      .def(init<scitbx::af::shared<double>>())
      .def("get_params", &Simple1MosaicityParameterisation::get_params);

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