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

Simple1MosaicityParameterisation::Simple1MosaicityParameterisation(
  scitbx::af::shared<double> parameters_)
    : BaseParameterisation(parameters_) {}

bool Simple1MosaicityParameterisation::is_angular() {
  return false;
}
int Simple1MosaicityParameterisation::num_parameters() {
  return 1;
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
  params = scitbx::af::small<double, 6>(X.size());
  for (int i = 0; i < X.size(); ++i) {
    params[i] = 1E5 * X[i];
  }
  compose();
}

scitbx::af::small<double, 6> SimpleCellParameterisation::get_params() {
  return params;
}
scitbx::mat3<double> SimpleCellParameterisation::get_state() {
  return B_;
}
void SimpleCellParameterisation::set_params(scitbx::af::small<double, 6> p) {
  params = p;
  compose();
}
scitbx::af::shared<scitbx::mat3<double>> SimpleCellParameterisation::get_dS_dp() {
  return dS_dp;
}