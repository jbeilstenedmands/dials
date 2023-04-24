#include <iostream>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <dxtbx/model/panel.h>
#include <cmath>
#include <dials/algorithms/profile_model/ellipsoid/parameterisation.h>
#include <dials/algorithms/profile_model/ellipsoid/g_gradients.h>
#include <dials/algorithms/refinement/parameterisation/parameterisation_helpers.h>
#include <dials/array_family/reflection_table.h>
#include <dxtbx/model/experiment.h>
#include <map>
#include <string>
#include <cctbx/sgtbx/space_group.h>
#include <cctbx/sgtbx/tensor_rank_2.h>
#include <cctbx/crystal_orientation.h>

WavelengthSpreadParameterisation::WavelengthSpreadParameterisation(double parameter)
    : parameter_(parameter) {}
double WavelengthSpreadParameterisation::sigma() {
  return parameter_;
}
int WavelengthSpreadParameterisation::num_parameters() {
  return 1;
}
double WavelengthSpreadParameterisation::first_derivative() {
  return 2.0 * parameter_;
}
double WavelengthSpreadParameterisation::get_param() {
  return parameter_;
}
void WavelengthSpreadParameterisation::set_param(double p) {
  parameter_ = p;
}

BaseParameterisation::BaseParameterisation(scitbx::af::shared<double> parameters_)
    : parameters(parameters_) {}

scitbx::af::shared<double> BaseParameterisation::get_params() {
  return parameters;
}

int BaseParameterisation::num_parameters() {
  return parameters.size();
}

void BaseParameterisation::set_params(scitbx::af::shared<double> parameters_) {
  parameters = parameters_;
}
bool BaseParameterisation::is_angular() {
  return false;
}
scitbx::af::shared<scitbx::mat3<double>> BaseParameterisation::first_derivatives() {
  scitbx::af::shared<scitbx::mat3<double>> derivatives;
  return derivatives;
}

Simple1MosaicityParameterisation::Simple1MosaicityParameterisation(
  scitbx::af::shared<double> parameters_)
    : BaseParameterisation(parameters_) {}

int Simple1MosaicityParameterisation::num_parameters() {
  return 1;
}
scitbx::mat3<double> Simple6MosaicityParameterisation::sigma() {
  scitbx::af::shared<double> params = this->get_params();
  scitbx::mat3<double> M{
    params[0], 0, 0, params[1], params[2], 0, params[3], params[4], params[5]};
  scitbx::mat3<double> MMT = M * M.transpose();
  return MMT;
}
scitbx::af::shared<scitbx::mat3<double>>
Simple6MosaicityParameterisation::first_derivatives() {
  scitbx::af::shared<double> p = this->get_params();
  double p1(p[0]), p2(p[1]), p3(p[2]), p4(p[3]), p5(p[4]), p6(p[5]);
  scitbx::af::shared<scitbx::mat3<double>> derivatives(6);
  derivatives[0] = scitbx::mat3<double>(2 * p1, p2, p4, p2, 0, 0, p4, 0, 0);
  derivatives[1] = scitbx::mat3<double>(0, p1, 0, p1, 2 * p2, p4, 0, p4, 0);
  derivatives[2] = scitbx::mat3<double>(0, 0, 0, 0, 2 * p3, p5, 0, p5, 0);
  derivatives[3] = scitbx::mat3<double>(0, 0, p1, 0, 0, p2, p1, p2, 2 * p4);
  derivatives[4] = scitbx::mat3<double>(0, 0, 0, 0, 0, p3, 0, p3, 2 * p5);
  derivatives[5] = scitbx::mat3<double>(0, 0, 0, 0, 0, 0, 0, 0, 2 * p6);
  return derivatives;
}
std::map<std::string, double> Simple6MosaicityParameterisation::mosaicity() {
  std::map<std::string, double> mosaicity;
  mosaicity.insert(std::pair<std::string, double>("firsttest", this->get_params()[0]));
  return mosaicity;
}

Simple6MosaicityParameterisation::Simple6MosaicityParameterisation(
  scitbx::af::shared<double> parameters)
    : parameters_(parameters) {}

void Simple6MosaicityParameterisation::set_params(
  scitbx::af::shared<double> parameters) {
  parameters_ = parameters;
}
scitbx::af::shared<double> Simple6MosaicityParameterisation::get_params() {
  return parameters_;
}
bool Simple6MosaicityParameterisation::is_angular() {
  return false;
}

int Simple6MosaicityParameterisation::num_parameters() {
  return 6;
}
scitbx::mat3<double> Simple1MosaicityParameterisation::sigma() {
  double psq = pow(this->get_params()[0], 2);
  return scitbx::mat3<double>(psq, 0, 0, 0, psq, 0, 0, 0, psq);
}
scitbx::af::shared<scitbx::mat3<double>>
Simple1MosaicityParameterisation::first_derivatives() {
  double p2 = this->get_params()[0] * 2.0;
  scitbx::af::shared<scitbx::mat3<double>> derivatives(1);
  derivatives[0] = scitbx::mat3<double>(p2, 0, 0, 0, p2, 0, 0, 0, p2);
  return derivatives;
}
std::map<std::string, double> Simple1MosaicityParameterisation::mosaicity() {
  std::map<std::string, double> mosaicity;
  mosaicity.insert(std::pair<std::string, double>("spherical", this->get_params()[0]));
  return mosaicity;
}

void SimpleUParameterisation::compose() {
  dials::refinement::CrystalOrientationCompose coc(
    istate, params[0], axes[0], params[1], axes[1], params[2], axes[2]);
  U_ = coc.U();
  dS_dp[0] = coc.dU_dphi1();
  dS_dp[1] = coc.dU_dphi2();
  dS_dp[2] = coc.dU_dphi3();
}

SimpleUParameterisation::SimpleUParameterisation(const dxtbx::model::Crystal &crystal) {
  istate = crystal.get_U();
  axes[1] = {0.0, 1.0, 0.0};
  axes[2] = {0.0, 0.0, 1.0};
  compose();
}

scitbx::af::shared<double> SimpleUParameterisation::get_params() {
  return params;
}
scitbx::mat3<double> SimpleUParameterisation::get_state() {
  return U_;
}
void SimpleUParameterisation::set_params(scitbx::af::shared<double> p) {
  assert(p.size() == 3);
  params = p;
  compose();
}
scitbx::af::shared<scitbx::mat3<double>> SimpleUParameterisation::get_dS_dp() {
  return dS_dp;
}

SymmetrizeReduceEnlarge::SymmetrizeReduceEnlarge(cctbx::sgtbx::space_group space_group)
    : space_group_(space_group), constraints_(space_group, true), Bconverter() {}

void SymmetrizeReduceEnlarge::set_orientation(scitbx::mat3<double> B) {
  orientation_ = cctbx::crystal_orientation(B, true);
}

scitbx::af::small<double, 6> SymmetrizeReduceEnlarge::forward_independent_parameters() {
  Bconverter.forward(orientation_);
  return constraints_.independent_params(Bconverter.G);
}

cctbx::crystal_orientation SymmetrizeReduceEnlarge::backward_orientation(
  scitbx::af::small<double, 6> independent) {
  scitbx::sym_mat3<double> ustar = constraints_.all_params(independent);
  Bconverter.validate_and_setG(ustar);
  orientation_ = Bconverter.back_as_orientation();
  return orientation_;
}

scitbx::af::shared<scitbx::mat3<double>> SymmetrizeReduceEnlarge::forward_gradients() {
  return dB_dp(Bconverter, constraints_);
}

void SimpleCellParameterisation::compose() {
  scitbx::af::small<double, 6> vals(params.size());
  for (int i = 0; i < params.size(); ++i) {
    vals[i] = 1E-5 * params[i];
  }
  SRE.set_orientation(B_);
  SRE.forward_independent_parameters();
  B_ = SRE.backward_orientation(vals).reciprocal_matrix();
  dS_dp = SRE.forward_gradients();
  for (int i = 0; i < dS_dp.size(); ++i) {
    for (int j = 0; j < 9; ++j) {
      dS_dp[i][j] *= 1E-5;
    }
  }
}

SimpleCellParameterisation::SimpleCellParameterisation(
  const dxtbx::model::Crystal &crystal)
    : B_(crystal.get_B()), SRE(crystal.get_space_group()) {
  // first get params
  SRE.set_orientation(B_);
  scitbx::af::small<double, 6> X = SRE.forward_independent_parameters();
  params = scitbx::af::shared<double>(X.size());
  for (int i = 0; i < X.size(); ++i) {
    params[i] = 1E5 * X[i];
  }
  compose();
}

scitbx::af::shared<double> SimpleCellParameterisation::get_params() {
  return params;
}
scitbx::mat3<double> SimpleCellParameterisation::get_state() {
  return B_;
}
void SimpleCellParameterisation::set_params(scitbx::af::shared<double> p) {
  params = p;
  compose();
}
scitbx::af::shared<scitbx::mat3<double>> SimpleCellParameterisation::get_dS_dp() {
  return dS_dp;
}

ModelState::ModelState(const dxtbx::model::Crystal &crystal,
                       Simple6MosaicityParameterisation &m_parameterisation,
                       WavelengthSpreadParameterisation &l_parameterisation,
                       bool fix_orientation,
                       bool fix_unit_cell,
                       bool fix_wavelength_spread,
                       bool fix_mosaic_spread)
    : M_parameterisation(m_parameterisation),
      L_parameterisation(l_parameterisation),
      B_parameterisation(SimpleCellParameterisation(crystal)),
      U_parameterisation(SimpleUParameterisation(crystal)),
      fix_orientation(fix_orientation),
      fix_unit_cell(fix_unit_cell),
      fix_wavelength_spread(fix_wavelength_spread),
      fix_mosaic_spread(fix_mosaic_spread),
      n_active_params(0) {
  if (!fix_orientation) {
    scitbx::af::shared<double> p = U_parameterisation.get_params();
    n_active_params += p.size();
  }
  if (!fix_unit_cell) {
    scitbx::af::shared<double> p = B_parameterisation.get_params();
    n_active_params += p.size();
  }
  if (!fix_mosaic_spread) {
    scitbx::af::shared<double> p = M_parameterisation.get_params();
    n_active_params += p.size();
  }
  if (!fix_wavelength_spread) {
    n_active_params += 1;
  }
}

int ModelState::n_active_parameters() {
  return n_active_params;
}

scitbx::mat3<double> ModelState::mosaicity_covariance_matrix() {
  return M_parameterisation.sigma();
}
double ModelState::wavelength_spread() {
  return L_parameterisation.sigma();
}

scitbx::mat3<double> ModelState::U_matrix() {
  return U_parameterisation.get_state();
}
scitbx::mat3<double> ModelState::B_matrix() {
  return B_parameterisation.get_state();
}
scitbx::mat3<double> ModelState::A_matrix() {
  return U_parameterisation.get_state() * B_parameterisation.get_state();
}
scitbx::af::shared<double> ModelState::U_params() {
  return U_parameterisation.get_params();
}
void ModelState::set_U_params(scitbx::af::shared<double> p) {
  U_parameterisation.set_params(p);
}
scitbx::af::shared<double> ModelState::B_params() {
  return B_parameterisation.get_params();
}
void ModelState::set_B_params(scitbx::af::shared<double> p) {
  B_parameterisation.set_params(p);
}
scitbx::af::shared<double> ModelState::M_params() {
  return M_parameterisation.get_params();
}
void ModelState::set_M_params(scitbx::af::shared<double> p) {
  M_parameterisation.set_params(p);
}
double ModelState::L_param() {
  return L_parameterisation.get_param();
}
void ModelState::set_L_param(double p) {
  L_parameterisation.set_param(p);
}

bool ModelState::is_orientation_fixed() {
  return fix_orientation;
}
bool ModelState::is_unit_cell_fixed() {
  return fix_unit_cell;
}
bool ModelState::is_mosaic_spread_fixed() {
  return fix_mosaic_spread;
}
bool ModelState::is_mosaic_spread_angular() {
  return M_parameterisation.is_angular();
}
bool ModelState::is_wavelength_spread_fixed() {
  return fix_wavelength_spread;
}

scitbx::af::shared<scitbx::mat3<double>> ModelState::dU_dp() {
  return U_parameterisation.get_dS_dp();
}
scitbx::af::shared<scitbx::mat3<double>> ModelState::dB_dp() {
  return B_parameterisation.get_dS_dp();
}
scitbx::af::shared<scitbx::mat3<double>> ModelState::dM_dp() {
  return M_parameterisation.first_derivatives();
}
double ModelState::dL_dp() {
  return L_parameterisation.first_derivative();
}
scitbx::af::shared<double> ModelState::active_parameters() {
  scitbx::af::shared<double> active_params = {};
  if (!fix_orientation) {
    scitbx::af::shared<double> p = U_parameterisation.get_params();
    for (size_t i = 0; i < p.size(); ++i) {
      active_params.push_back(p[i]);
    }
  }
  if (!fix_unit_cell) {
    scitbx::af::shared<double> p = B_parameterisation.get_params();
    for (size_t i = 0; i < p.size(); ++i) {
      active_params.push_back(p[i]);
    }
  }
  if (!fix_mosaic_spread) {
    scitbx::af::shared<double> p = M_parameterisation.get_params();
    for (size_t i = 0; i < p.size(); ++i) {
      active_params.push_back(p[i]);
    }
  }
  if (!fix_wavelength_spread) {
    active_params.push_back(L_parameterisation.get_param());
  }
  return active_params;
}
void ModelState::set_active_parameters(scitbx::af::shared<double> parameters) {
  size_t n_param = 0;
  DIALS_ASSERT(parameters.size() == n_active_parameters());
  if (!fix_orientation) {
    size_t nU = U_parameterisation.get_params().size();
    scitbx::af::shared<double> new_U(nU, 0.0);
    for (size_t i = 0; i < nU; ++i) {
      new_U[i] = parameters[i + n_param];
    }
    U_parameterisation.set_params(new_U);
    n_param += nU;
  }
  if (!fix_unit_cell) {
    size_t nB = B_parameterisation.get_params().size();
    scitbx::af::shared<double> new_B(nB, 0.0);
    for (size_t i = 0; i < nB; ++i) {
      new_B[i] = parameters[i + n_param];
    }
    B_parameterisation.set_params(new_B);
    n_param += nB;
  }
  if (!fix_mosaic_spread) {
    size_t nM = M_parameterisation.get_params().size();
    scitbx::af::shared<double> new_M(nM, 0.0);
    for (size_t i = 0; i < nM; ++i) {
      new_M[i] = parameters[i + n_param];
    }
    M_parameterisation.set_params(new_M);
    n_param += nM;
  }
  if (!fix_wavelength_spread) {
    size_t nL = 1;
    L_parameterisation.set_param(parameters[n_param]);
    n_param += nL;
  }
}

std::vector<std::string> ModelState::parameter_labels() {
  std::vector<std::string> labels(1);
  labels[0] = "Test";
  return labels;
}

ReflectionModelState::ReflectionModelState(ModelState &state,
                                           scitbx::vec3<double> s0,
                                           cctbx::miller::index<> h)
    : s0_(s0), h_(h), state_(state) {
  r = state_.A_matrix() * h_;
  norm_s0 = s0_.normalize();
  size_t n_params = 0;
  if (!state_.is_orientation_fixed()) {
    n_params += state_.U_params().size();
  }
  if (!state_.is_unit_cell_fixed()) {
    n_params += state_.B_params().size();
  }
  if (!state_.is_mosaic_spread_fixed()) {
    n_params += state_.M_params().size();
  }
  if (!state_.is_wavelength_spread_fixed()) {
    n_params += 1;
  }
  dr_dp.resize(n_params, {0, 0, 0});
  ds_dp.resize(n_params, {0, 0, 0, 0, 0, 0, 0, 0, 0});
  dl_dp.resize(n_params, 0);
  if (state_.is_mosaic_spread_angular()) {
    scitbx::vec3<double> norm_r = r.normalize();
    scitbx::vec3<double> q1(norm_r.cross(norm_s0).normalize());
    scitbx::vec3<double> q2(norm_r.cross(q1).normalize());
    Q[0] = q1[0];
    Q[1] = q1[1];
    Q[2] = q1[2];
    Q[3] = q2[0];
    Q[4] = q2[1];
    Q[5] = q2[2];
    Q[6] = norm_r[0];
    Q[7] = norm_r[1];
    Q[8] = norm_r[2];
  }
  recalc_sigma();
  recalc_sigma_lambda();
  update();
}

ModelState ReflectionModelState::get_state() {
  return state_;
}

void ReflectionModelState::recalc_sigma() {
  scitbx::mat3<double> M = state_.mosaicity_covariance_matrix();
  if (state_.is_mosaic_spread_angular()) {
    /* First check if r has actually been updated */
    if (!state_.is_orientation_fixed() || !state_.is_unit_cell_fixed()) {
      scitbx::vec3<double> norm_r = r.normalize();
      scitbx::vec3<double> q1(norm_r.cross(norm_s0).normalize());
      scitbx::vec3<double> q2(norm_r.cross(q1).normalize());
      Q[0] = q1[0];
      Q[1] = q1[1];
      Q[2] = q1[2];
      Q[3] = q2[0];
      Q[4] = q2[1];
      Q[5] = q2[2];
      Q[6] = norm_r[0];
      Q[7] = norm_r[1];
      Q[8] = norm_r[2];
    }
    sigma = (Q.transpose() * M) * Q;
  } else {
    sigma = M;
  }
}

void ReflectionModelState::recalc_sigma_lambda() {
  sigma_lambda = state_.wavelength_spread();
}

void ReflectionModelState::update() {
  if (!state_.is_orientation_fixed() || !state_.is_unit_cell_fixed()) {
    r = state_.A_matrix() * h_;
  }
  if (!state_.is_wavelength_spread_fixed()) {
    recalc_sigma_lambda();
  }
  if (!state_.is_mosaic_spread_fixed()) {
    recalc_sigma();
  }
  size_t ntot = 0;
  if (!state_.is_orientation_fixed()) {
    scitbx::af::shared<scitbx::mat3<double>> dU_dp = state_.dU_dp();
    scitbx::mat3<double> B = state_.B_matrix();
    for (size_t i = 0; i < dU_dp.size(); ++i) {
      dr_dp[i] = (dU_dp[i] * B) * h_;
    }
    ntot += dU_dp.size();
  }
  if (!state_.is_unit_cell_fixed()) {
    scitbx::af::shared<scitbx::mat3<double>> dB_dp = state_.dB_dp();
    scitbx::mat3<double> U = state_.U_matrix();
    for (size_t i = 0; i < dB_dp.size(); ++i) {
      dr_dp[i + ntot] = (U * dB_dp[i]) * h_;
    }
    ntot += dB_dp.size();
  }
  if (!state_.is_mosaic_spread_fixed()) {
    scitbx::af::shared<scitbx::mat3<double>> dM_dp = state_.dM_dp();
    if (state_.is_mosaic_spread_angular()) {
      scitbx::mat3<double> QT = Q.transpose();
      for (size_t i = 0; i < dM_dp.size(); ++i) {
        ds_dp[i + ntot] = (QT * dM_dp[i]) * Q;
      }
    } else {
      for (size_t i = 0; i < dM_dp.size(); ++i) {
        ds_dp[i + ntot] = dM_dp[i];
      }
    }
    ntot += dM_dp.size();
  }
  if (!state_.is_wavelength_spread_fixed()) {
    dl_dp[ntot] = state_.dL_dp();
  }
}

scitbx::mat3<double> ReflectionModelState::mosaicity_covariance_matrix() {
  return sigma;
}
scitbx::vec3<double> ReflectionModelState::get_r() {
  return r;
}
scitbx::af::shared<scitbx::mat3<double>> ReflectionModelState::get_dS_dp() {
  return ds_dp;
}
scitbx::af::shared<scitbx::vec3<double>> ReflectionModelState::get_dr_dp() {
  return dr_dp;
}
scitbx::af::shared<double> ReflectionModelState::get_dL_dp() {
  return dl_dp;
}