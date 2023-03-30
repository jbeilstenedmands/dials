#include <iostream>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <dxtbx/model/panel.h>
#include <cmath>
#include <dials/algorithms/profile_model/ellipsoid/refiner.h>

/*
scitbx::vec2<double> rse(
    const std::vector<double> &R,
    const std::vector<double> &mbar,
    const std::vector<double> &xobs,
    const double &norm_s0,
    const Panel &panel
){
    scitbx::vec3<double> s1;
    scitbx::vec3<double> s3;
    s1[0] = (R[0] * mbar[0]) + (R[3] * mbar[1]) + (R[6] * norm_s0);
    s1[1] = (R[1] * mbar[0]) + (R[4] * mbar[1]) + (R[7] * norm_s0);
    s1[2] = (R[2] * mbar[0]) + (R[5] * mbar[1]) + (R[8] * norm_s0);
    s3[0] = (R[0] * xobs[0]) + (R[3] * xobs[1]) + (R[6] * norm_s0);
    s3[1] = (R[1] * xobs[0]) + (R[4] * xobs[1]) + (R[7] * norm_s0);
    s3[2] = (R[2] * xobs[0]) + (R[5] * xobs[1]) + (R[8] * norm_s0);
    scitbx::vec2<double> xyzcal = panel.get_ray_intersection_px(s1);
    scitbx::vec2<double> xyzobs = panel.get_ray_intersection_px(s3);
    double rx2 = pow(xyzcal[0] - xyzobs[0], 2);
    double ry2 = pow(xyzcal[1] - xyzobs[1], 2);
    scitbx::vec2<double> res{rx2, ry2};
    return res;
};*/

inline scitbx::vec2<double> compute_dmbar(const scitbx::mat3<double> &S,
                                          const scitbx::mat3<double> &dS,
                                          const scitbx::vec3<double> &dmu,
                                          double &epsilon) {
  scitbx::vec2<double> S12{S[2], S[5]};
  scitbx::vec2<double> A{dmu[0], dmu[1]};
  scitbx::vec2<double> B{dS[2] * epsilon / S[8], dS[5] * epsilon / S[8]};
  scitbx::vec2<double> C = S12 * -1.0 * dS[8] * epsilon / (S[8] * S[8]);
  scitbx::vec2<double> D = S12 * -1.0 * dmu[2] / S[8];
  return A + B + C + D;
}

inline scitbx::mat2<double> compute_dSbar(const scitbx::mat3<double> &S,
                                          const scitbx::mat3<double> &dS) {
  scitbx::mat2<double> A{dS[0], dS[1], dS[3], dS[4]};
  double B_mult = dS[8] / (S[8] * S[8]);
  scitbx::mat2<double> B{S[2] * S[6] * B_mult,
                         S[2] * S[7] * B_mult,
                         S[5] * S[6] * B_mult,
                         S[5] * S[7] * B_mult};
  scitbx::mat2<double> C{
    S[2] * dS[6] / S[8], S[2] * dS[7] / S[8], S[5] * dS[6] / S[8], S[5] * dS[7] / S[8]};
  scitbx::mat2<double> D{
    dS[2] * S[6] / S[8], dS[2] * S[7] / S[8], dS[5] * S[6] / S[8], dS[5] * S[7] / S[8]};
  return A + B - (C + D);
}

ConditionalDistribution2::ConditionalDistribution2(
  double norm_s0_,
  scitbx::vec3<double> mu_,
  scitbx::af::shared<scitbx::vec3<double>> dmu_,
  scitbx::mat3<double> S_,
  scitbx::af::shared<scitbx::mat3<double>> dS_)
    : mu(mu_), dmu(dmu_), S(S_), dS(dS_) {
  scitbx::mat2<double> S11(S[0], S[1], S[3], S[4]);
  scitbx::vec2<double> S12(S[2], S[5]);
  scitbx::vec2<double> S21(S[6], S[7]);
  scitbx::vec2<double> mu1(mu[0], mu[1]);

  epsilon = norm_s0_ - mu[2];
  mubar = mu1 + (S12 * (1.0 / S[8]) * epsilon);

  scitbx::mat2<double> outerprodS12S21(S12[0] * S21[0] / S[8],
                                       S12[0] * S21[1] / S[8],
                                       S12[1] * S21[0] / S[8],
                                       S12[1] * S21[1] / S[8]);
  Sbar = S11 - outerprodS12S21;
}

scitbx::vec2<double> ConditionalDistribution2::mean() {
  /*Return the conditional mean*/
  return mubar;
}

scitbx::mat2<double> ConditionalDistribution2::sigma() {
  /*Return the conditional sigma*/
  return Sbar;
}

scitbx::af::shared<scitbx::vec2<double>>
ConditionalDistribution2::first_derivatives_of_mean() {
  /* Return the conditional mu first derivatives*/
  if (dmbar.size() == 0) {
    /*#scitbx::af::shared<scitbx::vec2<double>> dmbar(dS.size()); /*initialise with
     * correct length*/
    for (int i = 0; i < dS.size(); ++i) {
      dmbar.push_back(compute_dmbar(S, dS[i], dmu[i], epsilon));
    }
  }
  return dmbar;
}

scitbx::af::shared<scitbx::mat2<double>>
ConditionalDistribution2::first_derivatives_of_sigma() {
  /* Return the conditional sigma first derivatives*/
  if (dSbar.size() == 0) {
    for (int i = 0; i < dS.size(); ++i) {
      dSbar.push_back(compute_dSbar(S, dS[i]));
    }
  }
  return dSbar;
}

void test_conditional(double norm_s0,
                      scitbx::vec3<double> mu,
                      scitbx::af::shared<scitbx::vec3<double>> dmu,
                      scitbx::mat3<double> S,
                      scitbx::af::shared<scitbx::mat3<double>> dS) {
  ConditionalDistribution2 cond = ConditionalDistribution2(norm_s0, mu, dmu, S, dS);
  scitbx::af::shared<scitbx::vec2<double>> dm = cond.first_derivatives_of_mean();
  scitbx::af::shared<scitbx::mat2<double>> dSs = cond.first_derivatives_of_sigma();
  scitbx::vec2<double> mean = cond.mean();
  std::cout << 'mean' << std::endl;
  std::cout << mean[0] << std::endl;
  std::cout << mean[1] << std::endl;
  scitbx::mat2<double> sigma = cond.sigma();
  std::cout << 'sigma' << std::endl;
  std::cout << sigma[0] << std::endl;
  std::cout << sigma[1] << std::endl;
  std::cout << sigma[2] << std::endl;
  std::cout << sigma[3] << std::endl;
  std::cout << 'dmean' << std::endl;
  for (int i = 0; i < dm.size(); ++i) {
    for (int j = 0; j < 2; ++j) {
      std::cout << dm[i][j] << std::endl;
    }
  }
  std::cout << 'dsigma' << std::endl;
  for (int i = 0; i < dSs.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      std::cout << dSs[i][j] << std::endl;
    }
  }
}
