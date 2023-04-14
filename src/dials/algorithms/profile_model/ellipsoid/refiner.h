#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <scitbx/mat2.h>
#include <scitbx/mat3.h>
#include <dxtbx/model/panel.h>
#include <dials/array_family/reflection_table.h>
#include <dxtbx/model/experiment.h>
#include <dials/algorithms/profile_model/ellipsoid/parameterisation.h>

scitbx::vec2<double> rse(const std::vector<double> &R,
                         const std::vector<double> &mbar,
                         const std::vector<double> &xobs,
                         const double &norm_s0,
                         const dxtbx::model::Panel &panel);

void test_conditional(double norm_s0,
                      scitbx::vec3<double> mu,
                      scitbx::af::shared<scitbx::vec3<double>> dmu,
                      scitbx::mat3<double> S,
                      scitbx::af::shared<scitbx::mat3<double>> dS);

/* Define the Conditional Distribution class*/
class ConditionalDistribution2 {
public:
  ConditionalDistribution2(double norm_s0_,
                           scitbx::vec3<double> mu_,
                           scitbx::af::shared<scitbx::vec3<double>> dmu_,
                           scitbx::mat3<double> S_,
                           scitbx::af::shared<scitbx::mat3<double>> dS_);
  ConditionalDistribution2();
  scitbx::vec2<double> mean();
  scitbx::mat2<double> sigma();
  scitbx::af::shared<scitbx::vec2<double>> first_derivatives_of_mean();
  scitbx::af::shared<scitbx::mat2<double>> first_derivatives_of_sigma();

private:
  scitbx::vec3<double> mu;
  scitbx::af::shared<scitbx::vec3<double>> dmu;
  scitbx::mat3<double> S;
  scitbx::af::shared<scitbx::mat3<double>> dS;
  double epsilon;
  scitbx::vec2<double> mubar;
  scitbx::mat2<double> Sbar;
  scitbx::af::shared<scitbx::mat2<double>> dSbar{};
  scitbx::af::shared<scitbx::vec2<double>> dmbar{};
};

/*struct ReflectionSpotData {
    scitbx::vec3<double> s0;
    scitbx::vec3<double> sp;
    cctbx::miller<int> h;
    double ctot;
    scitbx::vec2<double> mobs;
    scitbx::mat2<double> Sobs;
}

class ReflectionLikelihood {
public:
  ReflectionLikelihood(model_,
                       ReflectionSpotData data_,
                       int panel_id_=0);
  void update();
  double log_likelihood();
  scitbx::af::shared<double> first_derivatives();
  scitbx::af::shared<double> fisher_information();

private:
  modelstate;
  ReflectionSpotData reflectiondata;
  double norm_s0;
  scitbx::mat2<double> R;
  int panel_id = 0;
  scitbx::vec3<double> mu;
  scitbx::mat3<double> S;
  scitbx::af::shared<scitbx::mat3<double>> dS;
  scitbx::af::shared<scitbx::vec3<double>> dmu;
  ConditionalDistribution2 conditional;
}*/

class RefinerData {
public:
  RefinerData(const dxtbx::model::Experiment &experiment,
              dials::af::reflection_table &reflections);
  RefinerData(scitbx::vec3<double> s0,
              scitbx::af::shared<scitbx::vec3<double>> sp,
              scitbx::af::shared<cctbx::miller::index<>> h,
              scitbx::af::shared<double> ctot,
              scitbx::af::shared<scitbx::vec2<double>> mobs,
              scitbx::af::shared<scitbx::mat2<double>> Sobs,
              scitbx::af::shared<size_t> panel_ids);
  scitbx::vec3<double> get_s0();
  scitbx::af::shared<scitbx::vec3<double>> get_sp_array();
  scitbx::af::shared<cctbx::miller::index<>> get_h_array();
  scitbx::af::shared<double> get_ctot_array();
  scitbx::af::shared<scitbx::vec2<double>> get_mobs_array();
  scitbx::af::shared<scitbx::mat2<double>> get_Sobs_array();
  scitbx::af::shared<size_t> get_panel_ids();

private:
  scitbx::vec3<double> s0;
  scitbx::af::shared<scitbx::vec3<double>> sp_array;
  scitbx::af::shared<cctbx::miller::index<>> h_array;
  scitbx::af::shared<double> ctot_array;
  scitbx::af::shared<scitbx::vec2<double>> mobs_array;
  scitbx::af::shared<scitbx::mat2<double>> Sobs_array;
  scitbx::af::shared<size_t> panel_ids;
};

class ReflectionLikelihood {
public:
  ReflectionLikelihood(ModelState &model,
                       scitbx::vec3<double> s0,
                       scitbx::vec3<double> sp,
                       cctbx::miller::index<> h,
                       double ctot,
                       scitbx::vec2<double> mobs,
                       scitbx::mat2<double> sobs,
                       size_t panel_id);
  void update();
  /*double log_likelihood();
  scitbx::vec3<double> first_derivatives();
  scitbx::vec3<double> fisher_information();*/

private:
  ReflectionModelState modelstate;
  scitbx::vec3<double> s0;
  double norm_s0;
  scitbx::vec3<double> sp;
  cctbx::miller::index<> h;
  double ctot;
  scitbx::vec2<double> mobs;
  scitbx::mat2<double> sobs;
  size_t panel_id;
  scitbx::mat3<double> R;
  scitbx::mat3<double> S;
  scitbx::af::shared<scitbx::mat3<double>> dS;
  scitbx::vec3<double> mu;
  scitbx::af::shared<scitbx::vec3<double>> dmu;
  ConditionalDistribution2 conditional;
};

class MLTarget {
public:
  MLTarget(ModelState &model, RefinerData &refinerdata);
  void update();

private:
  ModelState model;
  std::vector<ReflectionLikelihood> data{};
};

using namespace boost::python;

namespace dials { namespace algorithms { namespace boost_python {

  BOOST_PYTHON_MODULE(dials_algorithms_profile_model_ellipsoid_refiner_ext) {
    class_<ConditionalDistribution2>("ConditionalDistribution2", no_init)
      .def(init<double,
                scitbx::vec3<double>,
                scitbx::af::shared<scitbx::vec3<double>>,
                scitbx::mat3<double>,
                scitbx::af::shared<scitbx::mat3<double>>>());

    def("test_conditional", &test_conditional);

    class_<RefinerData>("RefinerData", no_init)
      .def(init<const dxtbx::model::Experiment &, dials::af::reflection_table &>())
      .def("get_s0", &RefinerData::get_s0)
      .def("get_sp_array", &RefinerData::get_sp_array)
      .def("get_h_array", &RefinerData::get_h_array)
      .def("get_ctot_array", &RefinerData::get_ctot_array)
      .def("get_mobs_array", &RefinerData::get_mobs_array)
      .def("get_panel_ids", &RefinerData::get_panel_ids)
      .def("get_Sobs_array", &RefinerData::get_Sobs_array);

    class_<ReflectionLikelihood>("ReflectionLikelihood", no_init)
      .def(init<ModelState &,
                scitbx::vec3<double>,
                scitbx::vec3<double>,
                cctbx::miller::index<>,
                double,
                scitbx::vec2<double>,
                scitbx::mat2<double>,
                size_t>());

    class_<MLTarget>("MLTarget", no_init)
      .def(init<ModelState &, RefinerData &>())
      .def("update", &MLTarget::update);
  }
}}}  // namespace dials::algorithms::boost_python