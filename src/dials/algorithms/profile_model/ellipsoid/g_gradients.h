#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <cmath>

scitbx::af::shared<scitbx::mat3<double>> dB_dp(
  rstbx::symmetry::AG Bconverter,
  cctbx::sgtbx::tensor_rank_2::constraints<double> constraints) {
  scitbx::af::shared<scitbx::mat3<double>> dB_dg(6);
  scitbx::sym_mat3<double> g = Bconverter.G;
  assert(g.size() == 6);
  double f = Bconverter.phi;
  double p = Bconverter.psi;
  double t = Bconverter.theta;

  double cosf = std::cos(f);
  double sinf = std::sin(f);
  double cosp = std::cos(p);
  double sinp = std::sin(p);
  double cost = std::cos(t);
  double sint = std::sin(t);

  double trig1 = cosf * cost * sinp - sinf * sint;
  double trig2 = cost * sinf + cosf * sinp * sint;
  double trig3 = -cost * sinf * sinp - cosf * sint;
  double trig4 = cosf * cost - sinf * sinp * sint;

  double g0 = g[0];
  double g1 = g[1];
  double g2 = g[2];
  double g3 = g[3];
  double g4 = g[4];
  double g5 = g[5];

  double rad3 =
    g0 - ((g2 * g3 * g3 + g1 * g4 * g4 - 2 * g3 * g4 * g5) / (g1 * g2 - g5 * g5));
  double sqrt_rad3 = std::sqrt(rad3);

  dB_dg[0] = scitbx::mat3<double>(0.5 * trig4 / sqrt_rad3,
                                  0,
                                  0,
                                  0.5 * cosp * sint / sqrt_rad3,
                                  0,
                                  0,
                                  0.5 * trig2 / sqrt_rad3,
                                  0,
                                  0);

  double fac4 = g2 * g3 - g4 * g5;
  double rad1 = g1 - g5 * g5 / g2;
  double rad1_three_half = std::sqrt(rad1 * rad1 * rad1);
  double fac3 = g5 * g5 - g1 * g2;
  double rad2 =
    -(g2 * g3 * g3 + g4 * (g1 * g4 - 2 * g3 * g5) + g0 * fac3) / (g1 * g2 - g5 * g5);
  double factor_dg1 = fac4 * fac4 / (fac3 * fac3 * std::sqrt(rad2));
  double fac5 = g3 - (g4 * g5 / g2);

  dB_dg[1] = scitbx::mat3<double>(
    -0.5 * fac5 * trig3 / rad1_three_half + 0.5 * factor_dg1 * trig4,
    0.5 * trig3 / std::sqrt(rad1),
    0,
    -0.5 * fac5 * cosp * cost / rad1_three_half + 0.5 * factor_dg1 * cosp * sint,
    0.5 * cosp * cost / std::sqrt(rad1),
    0,
    -0.5 * fac5 * trig1 / rad1_three_half + 0.5 * factor_dg1 * trig2,
    0.5 * trig1 / std::sqrt(rad1),
    0);

  double rat5_22 = g5 / (g2 * g2);
  double fac1 = g5 * (g3 - g4 * g5 / g2);
  double fac2 = (g1 * g4 - g3 * g5);
  double fac2sq = fac2 * fac2;

  dB_dg[2] = scitbx::mat3<double>(
    -0.5 * rat5_22 * fac1 * trig3 / rad1_three_half
      + g4 * rat5_22 * trig3 / std::sqrt(rad1)
      + 0.5 * fac2sq * trig4 / (fac3 * fac3 * std::sqrt(rad2))
      + 0.5 * g4 * cosp * sinf / pow(g2, 1.5),
    0.5 * rat5_22 * (g5 * trig3 / std::sqrt(rad1) + sqrt(g2) * cosp * sinf),
    -0.5 * cosp * sinf / std::sqrt(g2),

    -0.5 * rat5_22 * fac1 * cosp * cost / rad1_three_half
      + g4 * rat5_22 * cosp * cost / std::sqrt(rad1) + 0.5 * g4 * sinp / pow(g2, 1.5)
      + 0.5 * (fac2sq / fac3) * cosp * sint / (fac3 * std::sqrt(rad2)),
    0.5 * rat5_22 * (g5 * cosp * cost / std::sqrt(rad1) + sqrt(g2) * sinp),
    -0.5 * sinp / std::sqrt(g2),

    -0.5 * rat5_22 * fac1 * trig1 / rad1_three_half
      + g4 * rat5_22 * trig1 / std::sqrt(rad1)
      + 0.5 * fac2sq * trig2 / (fac3 * fac3 * std::sqrt(rad2))
      - 0.5 * g4 * cosf * cosp / pow(g2, 1.5),
    0.5 * rat5_22 * (g5 * trig1 / std::sqrt(rad1) - sqrt(g2) * cosf * cosp),
    0.5 * cosf * cosp / std::sqrt(g2));

  dB_dg[3] = scitbx::mat3<double>(
    trig3 / std::sqrt(rad1) + fac4 * trig4 / (fac3 * std::sqrt(rad2)),
    0,
    0,
    cosp * cost / std::sqrt(rad1) + fac4 * cosp * sint / (fac3 * std::sqrt(rad2)),
    0,
    0,
    trig1 / std::sqrt(rad1) + fac4 * trig2 / (fac3 * std::sqrt(rad2)),
    0,
    0);

  dB_dg[4] = scitbx::mat3<double>(
    -g5 * trig3 / (g2 * std::sqrt(rad1)) + fac2 * trig4 / (fac3 * std::sqrt(rad2))
      - cosp * sinf / std::sqrt(g2),
    0,
    0,
    -g5 * cosp * cost / (g2 * std::sqrt(rad1)) - sinp / std::sqrt(g2)
      + fac2 * cosp * sint / (fac3 * std::sqrt(rad2)),
    0,
    0,
    -g5 * trig1 / (g2 * std::sqrt(rad1)) + fac2 * trig2 / (fac3 * std::sqrt(rad2))
      + cosf * cosp / std::sqrt(g2),
    0,
    0);

  double better_ratio = (fac2 / fac3) * (fac4 / fac3);

  dB_dg[5] = scitbx::mat3<double>(
    fac1 * trig3 / (g2 * rad1_three_half) - g4 * trig3 / (g2 * std::sqrt(rad1))
      + better_ratio * trig4 / std::sqrt(rad2),
    -g5 * trig3 / (g2 * std::sqrt(rad1)) - cosp * sinf / std::sqrt(g2),
    0,
    fac1 * cosp * cost / (g2 * rad1_three_half)
      - g4 * cosp * cost / (g2 * std::sqrt(rad1))
      + better_ratio * cosp * sint / std::sqrt(rad2),
    -g5 * cosp * cost / (g2 * std::sqrt(rad1)) - sinp / std::sqrt(g2),
    0,
    fac1 * trig1 / (g2 * rad1_three_half) - g4 * trig1 / (g2 * std::sqrt(rad1))
      + better_ratio * trig2 / std::sqrt(rad2),
    -g5 * trig1 / (g2 * std::sqrt(rad1)) + cosf * cosp / std::sqrt(g2),
    0);

  int Nindep = constraints.n_independent_params();
  scitbx::af::shared<scitbx::mat3<double>> values(Nindep);
  for (int i = 0; i < 9; ++i) {
    scitbx::sym_mat3<double> all_grads{
      dB_dg[0][i], dB_dg[1][i], dB_dg[2][i], dB_dg[3][i], dB_dg[4][i], dB_dg[5][i]};
    scitbx::af::small<double, 6> g_indep = constraints.independent_gradients(all_grads);
    for (int j = 0; j < Nindep; ++j) {
      values[j][i] = g_indep[j];
    }
  }
  return values;
}
