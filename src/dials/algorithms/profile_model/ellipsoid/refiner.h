#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <scitbx/mat2.h>
#include <scitbx/mat3.h>
#include <dxtbx/model/panel.h>

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
  }
}}}  // namespace dials::algorithms::boost_python