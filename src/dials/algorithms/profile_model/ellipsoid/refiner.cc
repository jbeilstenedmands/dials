#include <iostream>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <dxtbx/model/panel.h>
#include <cmath>
#include <dials/algorithms/profile_model/ellipsoid/refiner.h>
#include <dials/array_family/reflection_table.h>
#include <dxtbx/model/experiment.h>
using dxtbx::model::Detector;
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
ConditionalDistribution2::ConditionalDistribution2() {}
ConditionalDistribution2::ConditionalDistribution2(
  double norm_s0_,
  scitbx::vec3<double> mu_,
  scitbx::af::shared<scitbx::vec3<double>> dmu_,
  scitbx::mat3<double> S_,
  scitbx::af::shared<scitbx::mat3<double>> dS_)
    : mu(mu_), dmu(dmu_), S(S_), dS(dS_) {
  scitbx::mat2<double> S11{S[0], S[1], S[3], S[4]};
  scitbx::vec2<double> S12{S[2], S[5]};
  scitbx::vec2<double> S21{S[6], S[7]};
  scitbx::vec2<double> mu1{mu[0], mu[1]};

  epsilon = norm_s0_ - mu[2];
  mubar = mu1 + (S12 * (1.0 / S[8]) * epsilon);

  scitbx::mat2<double> outerprodS12S21{S12[0] * S21[0] / S[8],
                                       S12[0] * S21[1] / S[8],
                                       S12[1] * S21[0] / S[8],
                                       S12[1] * S21[1] / S[8]};
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
  std::cout << mean[0] << std::endl;
  std::cout << mean[1] << std::endl;
  scitbx::mat2<double> sigma = cond.sigma();
  std::cout << sigma[0] << std::endl;
  std::cout << sigma[1] << std::endl;
  std::cout << sigma[2] << std::endl;
  std::cout << sigma[3] << std::endl;
  for (int i = 0; i < dm.size(); ++i) {
    for (int j = 0; j < 2; ++j) {
      std::cout << dm[i][j] << std::endl;
    }
  }
  for (int i = 0; i < dSs.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      std::cout << dSs[i][j] << std::endl;
    }
  }
}

scitbx::mat3<double> compute_change_of_basis_operation(scitbx::vec3<double> s0,
                                                       scitbx::vec3<double> s2) {
  const double TINY = 1e-7;
  DIALS_ASSERT((s2 - s0).length() > TINY);
  scitbx::vec3<double> e1 = s2.cross(s0).normalize();
  scitbx::vec3<double> e2 = s2.cross(e1).normalize();
  scitbx::vec3<double> e3 = s2.normalize();
  scitbx::mat3<double> R(e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2]);
  return R;
}

struct ReflectionStatistics {
  scitbx::vec3<double> sp;
  double ctot;
  scitbx::vec2<double> xbar;
  scitbx::mat2<double> Sobs;
};

ReflectionStatistics reflection_statistics(const dxtbx::model::Panel panel,
                                           const scitbx::vec3<double> xyzobs,
                                           const double s0_length,
                                           const scitbx::vec3<double> s0,
                                           const dials::af::Shoebox<> sbox) {
  typedef dials::af::Shoebox<>::float_type float_type;
  scitbx::vec2<double> p1(xyzobs[0], xyzobs[1]);
  scitbx::vec3<double> sp = panel.get_pixel_lab_coord(p1).normalize() * s0_length;
  scitbx::mat3<double> R = compute_change_of_basis_operation(s0, sp);

  const scitbx::af::versa<int, scitbx::af::c_grid<3>> mask = sbox.mask;
  const scitbx::af::const_ref<float_type, scitbx::af::c_grid<3>> data =
    sbox.data.const_ref();
  const scitbx::af::versa<float_type, scitbx::af::c_grid<3>> bgrd = sbox.background;
  int i0 = sbox.bbox[0];
  int j0 = sbox.bbox[2];
  int n1 = data.accessor()[1];
  int n2 = data.accessor()[2];

  scitbx::af::versa<float_type, scitbx::af::c_grid<3>> X(
    scitbx::af::c_grid<3>(n1, n2, 2));
  scitbx::af::versa<float_type, scitbx::af::c_grid<2>> C(scitbx::af::c_grid<2>(n1, n2));
  float_type ctot = 0;
  for (int j = 0; j < n1; ++j) {
    for (int i = 0; i < n2; ++i) {
      float_type c = data(0, j, i) - bgrd(0, j, i);
      if (c > 0) {
        if ((mask(0, j, i) & (1 | 4)) == (1 | 4)) {
          ctot += c;
          int ii = i + i0;
          int jj = j + j0;
          scitbx::vec2<double> sc(ii + 0.5, jj + 0.5);
          scitbx::vec3<double> s =
            panel.get_pixel_lab_coord(sc).normalize() * s0_length;
          scitbx::vec3<double> e = R * s;
          X(j, i, 0) = e[0];
          X(j, i, 1) = e[1];
          C(j, i) = c;
        }
      }
    }
  }
  // check ctot > 0

  // now sum to get Xbar
  scitbx::vec2<double> xbar(0, 0);
  for (int j = 0; j < n1; ++j) {
    for (int i = 0; i < n2; ++i) {
      xbar[0] += X(j, i, 0) * C(j, i);
      xbar[1] += X(j, i, 1) * C(j, i);
    }
  }
  xbar[0] = xbar[0] / ctot;
  xbar[1] = xbar[1] / ctot;

  scitbx::mat2<double> Sobs(0, 0, 0, 0);
  for (int j = 0; j < n1; ++j) {
    for (int i = 0; i < n2; ++i) {
      float_type c_i = C(j, i);
      scitbx::vec2<double> x_i(X(j, i, 0) - xbar[0], X(j, i, 1) - xbar[1]);
      Sobs[0] += pow(x_i[0], 2) * c_i;
      Sobs[1] += x_i[0] * x_i[1] * c_i;
      Sobs[2] += x_i[0] * x_i[1] * c_i;
      Sobs[3] += pow(x_i[1], 2) * c_i;
    }
  }
  Sobs[0] = Sobs[0] / ctot;
  Sobs[1] = Sobs[1] / ctot;
  Sobs[2] = Sobs[2] / ctot;
  Sobs[3] = Sobs[3] / ctot;

  ReflectionStatistics result = {sp, ctot, xbar, Sobs};
  return result;
}

RefinerData::RefinerData(const dxtbx::model::Experiment &experiment,
                         dials::af::reflection_table &reflections)
    : s0(experiment.get_beam()->get_s0()),
      sp_array(reflections.size()),
      h_array(reflections["miller_index"]),
      ctot_array(reflections.size()),
      mobs_array(reflections.size()),
      Sobs_array(reflections.size()),
      panel_ids(reflections["panel"]),
      detector(*experiment.get_detector()) {
  double s0_length = s0.length();
  scitbx::af::const_ref<scitbx::vec3<double>> xyzobs = reflections["xyzobs.px.value"];
  // std::shared_ptr<dxtbx::model::Detector> detector = experiment.get_detector();
  scitbx::af::shared<dials::af::Shoebox<>> sbox = reflections["shoebox"];
  for (size_t i = 0; i < reflections.size(); ++i) {
    size_t panel_id = panel_ids[i];
    dxtbx::model::Panel &panel = (detector)[panel_id];  // get panel obj
    ReflectionStatistics result =
      reflection_statistics(panel, xyzobs[i], s0_length, s0, sbox[i]);
    sp_array[i] = result.sp;
    ctot_array[i] = result.ctot;
    mobs_array[i] = result.xbar;
    Sobs_array[i] = result.Sobs;
  }

  // now need to dampen outliers in ctot list as these are used as weights - don't want
  // an enormous intensity to dominate the refinement.
  scitbx::af::shared<double> ctot_copy(ctot_array.begin(), ctot_array.end());
  int n = ctot_array.size();
  std::sort(ctot_copy.begin(), ctot_copy.end(), [](int a, int b) { return a < b; });
  double Q1 = ctot_copy[n / 4];
  double Q3 = ctot_copy[3 * n / 4];
  assert(Q3 >= Q1);
  double IQR = Q3 - Q1;
  double T = Q3 + (1.5 * IQR);
  for (size_t i = 0; i < n; ++i) {
    if (ctot_array[i] > T) {
      ctot_array[i] = T;
    }
  }
};

RefinerData::RefinerData(scitbx::vec3<double> s0_,
                         Detector &detector_,
                         scitbx::af::shared<scitbx::vec3<double>> sp_,
                         scitbx::af::shared<cctbx::miller::index<>> h_,
                         scitbx::af::shared<double> ctot_,
                         scitbx::af::shared<scitbx::vec2<double>> mobs_,
                         scitbx::af::shared<scitbx::mat2<double>> Sobs_,
                         scitbx::af::shared<size_t> panel_ids_)
    : s0(s0_),
      sp_array(sp_),
      h_array(h_),
      ctot_array(ctot_),
      mobs_array(mobs_),
      Sobs_array(Sobs_),
      panel_ids(panel_ids_),
      detector(detector_) {}

scitbx::vec3<double> RefinerData::get_s0() {
  return s0;
}
scitbx::af::shared<scitbx::vec3<double>> RefinerData::get_sp_array() {
  return sp_array;
}
scitbx::af::shared<cctbx::miller::index<>> RefinerData::get_h_array() {
  return h_array;
}
scitbx::af::shared<double> RefinerData::get_ctot_array() {
  return ctot_array;
}
scitbx::af::shared<scitbx::vec2<double>> RefinerData::get_mobs_array() {
  return mobs_array;
}
scitbx::af::shared<scitbx::mat2<double>> RefinerData::get_Sobs_array() {
  return Sobs_array;
}
scitbx::af::shared<size_t> RefinerData::get_panel_ids() {
  return panel_ids;
}
Detector &RefinerData::get_detector() {
  return detector;
}

ReflectionLikelihood::ReflectionLikelihood(ModelState &model,
                                           Detector &detector,
                                           scitbx::vec3<double> s0,
                                           scitbx::vec3<double> sp,
                                           cctbx::miller::index<> h,
                                           double ctot,
                                           scitbx::vec2<double> mobs,
                                           scitbx::mat2<double> sobs,
                                           size_t panel_id)
    : modelstate(ReflectionModelState(model, s0, h)),
      detector(detector),
      s0(s0),
      sp(sp),
      h(h),
      ctot(ctot),
      mobs(mobs),
      sobs(sobs),
      panel_id(panel_id) {
  norm_s0 = s0.length();
  R = compute_change_of_basis_operation(s0, sp);
  scitbx::vec3<double> s2 = s0 + modelstate.get_r();
  mu = R * s2;
  scitbx::mat3<double> RT = R.transpose();
  S = (R * model.mosaicity_covariance_matrix()) * RT;
  scitbx::af::shared<scitbx::mat3<double>> dS_dp = modelstate.get_dS_dp();
  dS.resize(dS_dp.size(), {0, 0, 0, 0, 0, 0, 0, 0, 0});
  for (size_t i = 0; i < dS_dp.size(); ++i) {
    dS[i] = (R * dS_dp[i]) * RT;
  }
  scitbx::af::shared<scitbx::vec3<double>> dr_dp = modelstate.get_dr_dp();
  dmu.resize(dr_dp.size(), {0, 0, 0});
  for (size_t i = 0; i < dr_dp.size(); ++i) {
    dmu[i] = R * dr_dp[i];
  }
  this->conditional = ConditionalDistribution2(norm_s0, mu, dmu, S, dS);
}

void ReflectionLikelihood::update() {
  modelstate.update();
  scitbx::vec3<double> s2 = s0 + modelstate.get_r();
  mu = R * s2;
  ModelState state = modelstate.get_state();
  scitbx::mat3<double> RT = R.transpose();
  if (!state.is_mosaic_spread_fixed()) {
    S = (R * modelstate.mosaicity_covariance_matrix()) * RT;
    scitbx::af::shared<scitbx::mat3<double>> dS_dp = modelstate.get_dS_dp();
    for (size_t i = 0; i < dS_dp.size(); ++i) {
      dS[i] = (R * dS_dp[i]) * RT;
    }
  }
  if (!state.is_unit_cell_fixed() || !state.is_orientation_fixed()) {
    scitbx::af::shared<scitbx::vec3<double>> dr_dp = modelstate.get_dr_dp();
    for (size_t i = 0; i < dr_dp.size(); ++i) {
      dmu[i] = R * dr_dp[i];
    }
  }
  this->conditional = ConditionalDistribution2(norm_s0, mu, dmu, S, dS);
}

double ReflectionLikelihood::log_likelihood() {
  double S22_inv = 1.0 / S[8];
  scitbx::mat2<double> Sbar = conditional.sigma();
  scitbx::vec2<double> mubar = conditional.mean();
  scitbx::mat2<double> Sbar_inv = Sbar.inverse();
  double Sbar_det = Sbar.determinant();
  // marginal likelihood
  double m_d = norm_s0 - mu[2];
  double m_lnL = ctot * (std::log(S[8]) + (S22_inv * pow(m_d, 2)));
  // conditional likelihood
  scitbx::vec2<double> c_d = mobs - mubar;
  scitbx::mat2<double> cdcdT{
    pow(c_d[0], 2), c_d[0] * c_d[1], c_d[0] * c_d[1], pow(c_d[1], 2)};
  scitbx::mat2<double> y = Sbar_inv * (sobs + cdcdT);
  double c_lnL = ctot * (std::log(Sbar_det) + y[0] + y[3]);
  // return the joint likelihood
  return -0.5 * (m_lnL + c_lnL);
}

scitbx::af::shared<double> ReflectionLikelihood::first_derivatives() {
  scitbx::af::shared<scitbx::mat2<double>> dSbar =
    conditional.first_derivatives_of_sigma();
  scitbx::af::shared<scitbx::vec2<double>> dmBar =
    conditional.first_derivatives_of_mean();
  scitbx::mat2<double> Sbar = conditional.sigma();
  scitbx::vec2<double> mubar = conditional.mean();
  int n_param = dSbar.size();
  double S22_inv = 1.0 / S[8];
  scitbx::mat2<double> Sbar_inv = Sbar.inverse();
  double epsilon = norm_s0 - mu[2];
  scitbx::vec2<double> c_d = mobs - mubar;
  scitbx::mat2<double> I(1.0, 0.0, 0.0, 1.0);
  scitbx::mat2<double> cdcdT(
    pow(c_d[0], 2), c_d[0] * c_d[1], c_d[0] * c_d[1], pow(c_d[1], 2));
  scitbx::mat2<double> V2 = I - (Sbar_inv * (sobs + cdcdT));
  scitbx::af::shared<double> V_vec(n_param, 0.0);
  for (int i = 0; i < n_param; ++i) {
    scitbx::mat2<double> Vvec = Sbar_inv * dSbar[i];
    V_vec[i] =
      ctot * (Vvec[0] * V2[0] + Vvec[1] * V2[2] + Vvec[2] * V2[1] + Vvec[3] * V2[3]);
  }
  for (int i = 0; i < n_param; ++i) {
    V_vec[i] += ctot
                * (S22_inv * dS[i][8] * (1.0 - (S22_inv * pow(epsilon, 2)))
                   + (2 * S22_inv * epsilon * -1.0 * dmu[i][2]));
  }
  for (int i = 0; i < n_param; ++i) {
    V_vec[i] -=
      2.0 * ctot
      * ((c_d[0] * dmBar[i][0] * Sbar_inv[0]) + (c_d[0] * dmBar[i][1] * Sbar_inv[2])
         + (c_d[1] * dmBar[i][0] * Sbar_inv[1]) + (c_d[1] * dmBar[i][1] * Sbar_inv[3]));
  }
  for (int i = 0; i < V_vec.size(); ++i) {
    V_vec[i] *= -0.5;
  }
  return V_vec;
}

scitbx::af::versa<double, scitbx::af::c_grid<2>>
ReflectionLikelihood::fisher_information() {
  scitbx::af::shared<scitbx::mat2<double>> dSbar =
    conditional.first_derivatives_of_sigma();
  scitbx::af::shared<scitbx::vec2<double>> dmBar =
    conditional.first_derivatives_of_mean();
  scitbx::mat2<double> Sbar = conditional.sigma();
  int n1 = dS.size();
  double S22_inv = 1 / S[8];
  scitbx::mat2<double> Sbar_inv = Sbar.inverse();
  scitbx::af::versa<double, scitbx::af::c_grid<2>> I(scitbx::af::c_grid<2>(n1, n1), 0);
  for (int j = 0; j < n1; ++j) {
    for (int i = 0; i < n1; ++i) {
      double U = pow(S22_inv, 2) * dS[j][8] * dS[i][8];
      double V = (((Sbar_inv * dSbar[j]) * Sbar_inv) * dSbar[i]).trace();
      scitbx::vec2<double> Y = Sbar_inv * dmBar[i];
      double W = 2.0 * ((Y[0] * dmBar[j][0]) + (Y[1] * dmBar[j][1]));
      double X = 2.0 * dmu[i][2] * S22_inv * dmu[j][2];
      I(j, i) = 0.5 * ctot * (V + W + U + X);
    }
  }
  return I;
}

double ReflectionLikelihood::square_error() {
  scitbx::vec2<double> mubar = conditional.mean();
  return pow(mubar[0] - mobs[0], 2) + pow(mubar[1] - mobs[1], 2);
}

scitbx::vec2<double> ReflectionLikelihood::rse() {
  scitbx::vec3<double> s1{};
  scitbx::vec3<double> s3{};
  scitbx::vec2<double> mbar = conditional.mean();
  s1[0] = (R[0] * mbar[0]) + (R[3] * mbar[1]) + (R[6] * norm_s0);
  s1[1] = (R[1] * mbar[0]) + (R[4] * mbar[1]) + (R[7] * norm_s0);
  s1[2] = (R[2] * mbar[0]) + (R[5] * mbar[1]) + (R[8] * norm_s0);
  s3[0] = (R[0] * mobs[0]) + (R[3] * mobs[1]) + (R[6] * norm_s0);
  s3[1] = (R[1] * mobs[0]) + (R[4] * mobs[1]) + (R[7] * norm_s0);
  s3[2] = (R[2] * mobs[0]) + (R[5] * mobs[1]) + (R[8] * norm_s0);
  scitbx::vec2<double> xyzcal = (detector)[0].get_ray_intersection_px(s1);
  scitbx::vec2<double> xyzobs = (detector)[0].get_ray_intersection_px(s3);
  double rx2 = pow(xyzcal[0] - xyzobs[0], 2);
  double ry2 = pow(xyzcal[1] - xyzobs[1], 2);
  scitbx::vec2<double> rse{rx2, ry2};
  return rse;
}

MLTarget::MLTarget(ModelState &model_, RefinerData &refinerdata) : model(model_) {
  scitbx::af::shared<cctbx::miller::index<>> h_list = refinerdata.get_h_array();
  scitbx::vec3<double> s0 = refinerdata.get_s0();
  scitbx::af::shared<scitbx::vec3<double>> sp_list = refinerdata.get_sp_array();
  scitbx::af::shared<double> ctot_list = refinerdata.get_ctot_array();
  scitbx::af::shared<scitbx::vec2<double>> mobs_list = refinerdata.get_mobs_array();
  scitbx::af::shared<size_t> panel_ids = refinerdata.get_panel_ids();
  scitbx::af::shared<scitbx::mat2<double>> sobs_list = refinerdata.get_Sobs_array();
  for (size_t i = 0; i < h_list.size(); ++i) {
    data.push_back(ReflectionLikelihood(model,
                                        refinerdata.get_detector(),
                                        s0,
                                        sp_list[i],
                                        h_list[i],
                                        ctot_list[i],
                                        mobs_list[i],
                                        sobs_list[i],
                                        panel_ids[i]));
  }
}

void MLTarget::update() {
  for (ReflectionLikelihood d : data) {
    d.update();
  }
}

double MLTarget::log_likelihood() {
  double l = 0.0;
  for (ReflectionLikelihood d : data) {
    l += d.log_likelihood();
  }
  return l;
}

double MLTarget::mse() {
  double mse = 0.0;
  for (ReflectionLikelihood d : data) {
    mse += d.square_error();
  }
  mse /= data.size();
  return mse;
}

scitbx::vec2<double> MLTarget::rmsd() {
  scitbx::vec2<double> rmsd = {0.0, 0.0};
  for (ReflectionLikelihood d : data) {
    rmsd += d.rse();
  }
  rmsd[0] /= data.size();
  rmsd[1] /= data.size();
  rmsd[0] = pow(rmsd[0], 0.5);
  rmsd[1] = pow(rmsd[1], 0.5);
  return rmsd;
}

scitbx::af::shared<double> MLTarget::first_derivatives() {
  int n1 = model.n_active_parameters();
  scitbx::af::shared<double> derivatives(n1, 0);
  for (ReflectionLikelihood d : data) {
    scitbx::af::shared<double> di = d.first_derivatives();
    for (size_t i = 0; i < di.size(); ++i) {
      derivatives[i] += di[i];
    }
  }
  return derivatives;
}

scitbx::af::versa<double, scitbx::af::c_grid<2>> MLTarget::fisher_information() {
  int n1 = model.n_active_parameters();
  scitbx::af::versa<double, scitbx::af::c_grid<2>> joint_f(
    scitbx::af::c_grid<2>(n1, n1), 0);
  for (ReflectionLikelihood d : data) {
    scitbx::af::versa<double, scitbx::af::c_grid<2>> fi = d.fisher_information();
    for (size_t i = 0; i < fi.size(); ++i) {
      joint_f[i] += fi[i];
    }
    /*for (size_t i=0;i<fi.accessor()[0];++i){
      for (size_t j=0;j<fi.accessor()[1];++j){
        joint_f += fi(i, j);
      }
    }*/
  }
  return joint_f;
}
