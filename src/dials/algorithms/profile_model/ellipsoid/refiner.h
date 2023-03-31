#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <scitbx/mat2.h>
#include <scitbx/mat3.h>
#include <dxtbx/model/panel.h>
#include <dials/array_family/reflection_table.h>
#include <dxtbx/model/experiment.h>

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
              scitbx::af::const_ref<cctbx::miller::index<>> h,
              scitbx::af::shared<double> ctot,
              scitbx::af::shared<scitbx::vec2<double>> mobs,
              scitbx::af::shared<scitbx::mat2<double>> Sobs,
              scitbx::af::shared<size_t> panel_ids);

private:
  scitbx::vec3<double> s0;
  scitbx::af::shared<scitbx::vec3<double>> sp_array;
  scitbx::af::const_ref<cctbx::miller::index<>> h_array;
  scitbx::af::shared<double> ctot_array;
  scitbx::af::shared<scitbx::vec2<double>> mobs_array;
  scitbx::af::shared<scitbx::mat2<double>> Sobs_array;
  scitbx::af::shared<size_t> panel_ids;
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
      .def(init<const dxtbx::model::Experiment &, dials::af::reflection_table &>());
  }
}}}  // namespace dials::algorithms::boost_python