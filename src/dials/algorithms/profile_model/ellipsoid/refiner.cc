#include <iostream>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <dxtbx/model/panel.h>
#include <cmath>
#include <dials/algorithms/profile_model/ellipsoid/refiner.h>
#include <dials/array_family/reflection_table.h>
#include <dxtbx/model/experiment.h>
#include <tuple>
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
      panel_ids(reflections["panel"]) {
  double s0_length = s0.length();
  scitbx::af::const_ref<scitbx::vec3<double>> xyzobs = reflections["xyzobs.px.value"];
  std::shared_ptr<dxtbx::model::Detector> detector = experiment.get_detector();
  scitbx::af::shared<dials::af::Shoebox<>> sbox = reflections["shoebox"];
  for (size_t i = 0; i < reflections.size(); ++i) {
    size_t panel_id = panel_ids[i];
    dxtbx::model::Panel &panel = (*detector)[panel_id];  // get panel obj
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
                         scitbx::af::shared<scitbx::vec3<double>> sp_,
                         scitbx::af::const_ref<cctbx::miller::index<>> h_,
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
      panel_ids(panel_ids_) {}