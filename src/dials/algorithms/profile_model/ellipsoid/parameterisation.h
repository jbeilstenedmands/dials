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
  virtual int num_parameters()=0;
  void set_params(scitbx::af::shared<double> parameters);
  bool is_angular();
  virtual scitbx::mat3<double> sigma()=0;
  virtual scitbx::af::shared<scitbx::mat3<double>> first_derivatives()=0;

private:
  scitbx::af::shared<double> parameters {};
};

class Simple1MosaicityParameterisation : public BaseParameterisation {
public:
  Simple1MosaicityParameterisation(scitbx::af::shared<double> parameters);
  int num_parameters();
  scitbx::mat3<double> sigma();
  scitbx::af::shared<scitbx::mat3<double>> first_derivatives();
  std::map<std::string, double> mosaicity();
};

class Simple6MosaicityParameterisation {
public:
  Simple6MosaicityParameterisation(scitbx::af::shared<double> parameters);
  int num_parameters();
  scitbx::mat3<double> sigma();
  scitbx::af::shared<scitbx::mat3<double>> first_derivatives();
  std::map<std::string, double> mosaicity();
  scitbx::af::shared<double> get_params();
  void set_params(scitbx::af::shared<double> parameters);
  bool is_angular();
private:
  scitbx::af::shared<double> parameters_ {};
};

class WavelengthSpreadParameterisation {
public:
  WavelengthSpreadParameterisation(double parameter);
  int num_parameters();
  double sigma();
  double first_derivative();
  double get_param();
  void set_param(double p);

private:
  double parameter_;
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
  scitbx::af::shared<double> get_params();
  void set_params(scitbx::af::shared<double>);
  scitbx::mat3<double> get_state();
  scitbx::af::shared<scitbx::mat3<double>> get_dS_dp();

private:
  scitbx::af::shared<double> params;
  void compose();
  scitbx::mat3<double> B_ {};
  scitbx::af::shared<scitbx::mat3<double>> dS_dp {};
  SymmetrizeReduceEnlarge SRE;
};

class ModelState {
public:
  ModelState(const dxtbx::model::Crystal &crystal,
             Simple6MosaicityParameterisation &m_parameterisation,
             WavelengthSpreadParameterisation &l_parameterisation,
             bool fix_orientation,
             bool fix_unit_cell,
             bool fix_wavelength_spread,
             bool fix_mosaic_spread);
  bool is_orientation_fixed();
  bool is_unit_cell_fixed();
  bool is_mosaic_spread_fixed();
  bool is_mosaic_spread_angular();
  bool is_wavelength_spread_fixed();
  scitbx::mat3<double> U_matrix();
  scitbx::mat3<double> B_matrix();
  scitbx::mat3<double> A_matrix();
  scitbx::mat3<double> mosaicity_covariance_matrix();
  double wavelength_spread();
  scitbx::af::shared<double> U_params();
  void set_U_params(scitbx::af::shared<double> p);
  scitbx::af::shared<double> B_params();
  void set_B_params(scitbx::af::shared<double> p);
  scitbx::af::shared<double> M_params();
  void set_M_params(scitbx::af::shared<double> p);
  double L_param();
  void set_L_param(double p);
  scitbx::af::shared<scitbx::mat3<double>> dU_dp();
  scitbx::af::shared<scitbx::mat3<double>> dB_dp();
  scitbx::af::shared<scitbx::mat3<double>> dM_dp();
  double dL_dp();
  scitbx::af::shared<double> active_parameters();
  void set_active_parameters(scitbx::af::shared<double> parameters);
  std::vector<std::string> parameter_labels();
  int n_active_parameters();

private:
  Simple6MosaicityParameterisation &M_parameterisation;
  WavelengthSpreadParameterisation &L_parameterisation;
  SimpleCellParameterisation B_parameterisation;
  SimpleUParameterisation U_parameterisation;
  bool fix_orientation {};
  bool fix_unit_cell {};
  bool fix_wavelength_spread {};
  bool fix_mosaic_spread {};
  int n_active_params {0};
};

class ReflectionModelState {
public:
  ReflectionModelState(ModelState &state,
                       scitbx::vec3<double> s0,
                       cctbx::miller::index<> h);
  void update();
  scitbx::mat3<double> mosaicity_covariance_matrix();
  scitbx::vec3<double> get_r();
  ModelState get_state();
  scitbx::af::shared<scitbx::mat3<double>> get_dS_dp();
  scitbx::af::shared<scitbx::vec3<double>> get_dr_dp();
  scitbx::af::shared<double> get_dL_dp();

private:
  scitbx::vec3<double> s0_ {};
  scitbx::vec3<double> r {};
  cctbx::miller::index<> h_ {};
  scitbx::vec3<double> norm_s0 {};
  ModelState &state_;
  scitbx::mat3<double> Q{0, 0, 0, 0, 0, 0, 0, 0, 0};
  scitbx::mat3<double> sigma{0, 0, 0, 0, 0, 0, 0, 0, 0};
  double sigma_lambda{0};
  scitbx::af::shared<scitbx::vec3<double>> dr_dp {};
  scitbx::af::shared<scitbx::mat3<double>> ds_dp {};
  scitbx::af::shared<double> dl_dp {};
  void recalc_sigma();
  void recalc_sigma_lambda();
};

using namespace boost::python;

namespace dials { namespace algorithms { namespace boost_python {

  BOOST_PYTHON_MODULE(dials_algorithms_profile_model_ellipsoid_parameterisation_ext) {
    class_<BaseParameterisation, boost::noncopyable>("BaseParameterisation", no_init);
    class_<Simple1MosaicityParameterisation, bases<BaseParameterisation>>(
      "Simple1MosaicityParameterisation", no_init)
      .def(init<scitbx::af::shared<double>>())
      .def("get_params", &Simple1MosaicityParameterisation::get_params)
      .def("sigma", &Simple1MosaicityParameterisation::sigma);
    class_<Simple6MosaicityParameterisation>(
      "Simple6MosaicityParameterisation", no_init)
      .def(init<scitbx::af::shared<double>>())
      .def("get_params", &Simple6MosaicityParameterisation::get_params)
      .def("sigma", &Simple6MosaicityParameterisation::sigma);
    class_<WavelengthSpreadParameterisation>("WavelengthSpreadParameterisation",
                                             no_init)
      .def(init<double>())
      .def("get_param", &WavelengthSpreadParameterisation::get_param)
      .def("sigma", &WavelengthSpreadParameterisation::sigma);

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

    class_<ModelState>("ModelState", no_init)
      .def(init<dxtbx::model::Crystal,
                Simple6MosaicityParameterisation &,
                WavelengthSpreadParameterisation &,
                bool,
                bool,
                bool,
                bool>())
      .def("is_orientation_fixed", &ModelState::is_orientation_fixed)
      .def("is_unit_cell_fixed", &ModelState::is_unit_cell_fixed)
      .def("is_mosaic_spread_fixed", &ModelState::is_mosaic_spread_fixed)
      .def("is_mosaic_spread_angular", &ModelState::is_mosaic_spread_angular)
      .def("is_wavelength_spread_fixed", &ModelState::is_wavelength_spread_fixed)
      .def("U_matrix", &ModelState::U_matrix)
      .def("B_matrix()", &ModelState::B_matrix)
      .def("A_matrix()", &ModelState::A_matrix)
      .def("active_parameters", &ModelState::active_parameters)
      .def("set_active_parameters", &ModelState::set_active_parameters)
      .def("n_active_parameters", &ModelState::n_active_parameters)
      .def("mosaicity_covariance_matrix", &ModelState::mosaicity_covariance_matrix);

    class_<ReflectionModelState>("ReflectionModelState", no_init)
      .def(init<ModelState &, scitbx::vec3<double>, cctbx::miller::index<>>())
      .def("update", &ReflectionModelState::update)
      .def("mosaicity_covariance_matrix",
           &ReflectionModelState::mosaicity_covariance_matrix)
      .def("get_r", &ReflectionModelState::get_r)
      .def("get_dS_dp", &ReflectionModelState::get_dS_dp)
      .def("get_dr_dp", &ReflectionModelState::get_dr_dp)
      .def("get_dL_dp", &ReflectionModelState::get_dL_dp);
  }
}}}  // namespace dials::algorithms::boost_python
