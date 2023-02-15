from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from numpy.linalg import norm

from scitbx import linalg, matrix

from dials.algorithms.profile_model.ellipsoid import (
    calc_dr_dp,
    calc_ds_dp,
    mosaicity_from_eigen_decomposition,
)
from dials.algorithms.refinement.parameterisation.crystal_parameters import (
    CrystalOrientationParameterisation,
    CrystalUnitCellParameterisation,
)
from dials.array_family import flex


class BaseParameterisation(ABC):
    def __init__(self, params: Optional[np.array] = None) -> None:
        """
        Initialise with the parameters

        """
        if params is not None:
            assert len(params) == self.num_parameters()
            self.params = params
        else:
            self.params = np.array([0.0] * self.num_parameters(), dtype=np.float64)

    @abstractmethod
    def num_parameters(self):
        pass

    @property
    def parameters(self) -> np.array:
        """
        Return the parameters

        """
        return self.params

    @parameters.setter
    def parameters(self, params: np.array) -> None:
        assert len(params) == self.num_parameters()
        self.params = params


class Simple1MosaicityParameterisation(BaseParameterisation):
    """
    A simple mosaicity parameterisation that uses 1 parameter to describe a
    multivariate normal reciprocal lattice profile. Sigma is enforced as positive
    definite by parameterising using the cholesky decomposition.

    M = | b1 0  0  |
        |  0 b1 0  |
        |  0  0 b1 |

    S = M*M^T

    """

    @staticmethod
    def is_angular() -> bool:
        return False

    @staticmethod
    def num_parameters() -> int:
        return 1

    def sigma(self) -> np.array:
        """
        Compute the covariance matrix of the MVN from the parameters

        """
        psq = self.params[0] ** 2
        return matrix.sqr((psq, 0, 0, 0, psq, 0, 0, 0, psq))

    def first_derivatives(self) -> np.array:
        """
        Compute the first derivatives of Sigma w.r.t the parameters

        """
        b1 = self.params[0]
        ds = flex.double([2.0 * b1, 0, 0, 0, 2.0 * b1, 0, 0, 0, 2.0 * b1])
        ds.reshape(flex.grid(1, 3, 3))
        return ds

    def mosaicity(self) -> Dict:
        """One value for mosaicity for Simple1"""
        decomp = linalg.eigensystem.real_symmetric(self.sigma().as_flex_double_matrix())
        v = list(mosaicity_from_eigen_decomposition(decomp.values()))
        return {"spherical": v[0]}


class Simple6MosaicityParameterisation(BaseParameterisation):
    """
    A simple mosaicity parameterisation that uses 6 parameters to describe a
    multivariate normal reciprocal lattice profile. Sigma is enforced as positive
    definite by parameterising using the cholesky decomposition.

    M = | b1 0  0  |
        | b2 b3 0  |
        | b4 b5 b6 |

    S = M*M^T

    """

    @staticmethod
    def is_angular() -> bool:
        return False

    @staticmethod
    def num_parameters() -> int:
        return 6

    def sigma(self) -> np.array:
        """
        Compute the covariance matrix of the MVN from the parameters

        """
        M = matrix.sqr(
            (
                self.params[0],
                0,
                0,
                self.params[1],
                self.params[2],
                0,
                self.params[3],
                self.params[4],
                self.params[5],
            )
        )
        return M * M.transpose()

    def mosaicity(self) -> Dict:
        """Three components for mosaicity for Simple6"""
        decomp = linalg.eigensystem.real_symmetric(self.sigma().as_flex_double_matrix())
        vals = list(mosaicity_from_eigen_decomposition(decomp.values()))
        min_m = min(vals)
        max_m = max(vals)
        vals.remove(min_m)
        vals.remove(max_m)
        return {"min": min_m, "mid": vals[0], "max": max_m}

    def first_derivatives(self) -> np.array:
        """
        Compute the first derivatives of Sigma w.r.t the parameters

        """
        b1, b2, b3, b4, b5, b6 = self.params
        ds = flex.double(
            [
                2 * b1,
                b2,
                b4,
                b2,
                0,
                0,
                b4,
                0,
                0,
                0,
                b1,
                0,
                b1,
                2 * b2,
                b4,
                0,
                b4,
                0,
                0,
                0,
                0,
                0,
                2 * b3,
                b5,
                0,
                b5,
                0,
                0,
                0,
                b1,
                0,
                0,
                b2,
                1,
                b2,
                2 * b4,
                0,
                0,
                0,
                0,
                0,
                b3,
                0,
                b3,
                2 * b5,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2 * b6,
            ]
        )
        ds.reshape(flex.grid(6, 3, 3))

        return ds


class WavelengthSpreadParameterisation(BaseParameterisation):
    """
    A simple wavelength parameterisation that uses 1 parameter to describe a
    multivariate normal wavelength spread. Sigma is enforced as positive
    definite by parameterising using the cholesky decomposition.

    L = | 0 0 0  |
        | 0 0 0  |
        | 0 0 l1 |

    S = L*L^T

    """

    @staticmethod
    def num_parameters() -> int:
        return 1

    def sigma(self) -> flex.double:
        """
        The normal distribution sigma

        """
        return flex.double([self.params[0]])

    def first_derivatives(self) -> flex.double:
        """
        Compute the first derivatives of Sigma w.r.t the parameters

        """
        return flex.double([2 * self.params[0]])


class Angular2MosaicityParameterisation(BaseParameterisation):
    """
    A simple mosaicity parameterisation that uses 2 parameters to describe a
    multivariate normal angular mosaic spread. Sigma is enforced as positive
    definite by parameterising using the cholesky decomposition.
    W = | w1 0  0  |
        | 0 w1  0  |
        | 0  0 w2 |
    S = W*W^T
    """

    @staticmethod
    def is_angular() -> bool:
        return True

    @staticmethod
    def num_parameters() -> int:
        return 2

    def sigma(self) -> matrix:
        """
        Compute the covariance matrix of the MVN from the parameters
        """
        p1sq = self.params[0] ** 2
        p2sq = self.params[1] ** 2
        return matrix.sqr((p1sq, 0, 0, 0, p1sq, 0, 0, 0, p2sq))

    def mosaicity(self) -> Dict:
        """Two unique components of mosaicity"""
        decomp = linalg.eigensystem.real_symmetric(self.sigma().as_flex_double_matrix())
        m = mosaicity_from_eigen_decomposition(decomp.values())
        v = decomp.vectors()
        mosaicities = {"radial": 0, "angular": 0}
        # two values must be same, could have accidental degeneracy where all 3 same:
        unique_ = list(set(m))
        if len(unique_) == 1:
            return {"angular": unique_[0], "radial": unique_[0]}
        else:
            assert len(unique_) == 2
            for i in range(3):
                vec = (v[i * 3], v[(i * 3) + 1], v[(i * 3) + 2])
                if vec == (0, 0, 1):
                    mosaicities["radial"] = m[i]
                else:
                    mosaicities["angular"] = m[i]
        return mosaicities

    def first_derivatives(self) -> np.array:
        """
        Compute the first derivatives of Sigma w.r.t the parameters
        """
        b1, b2 = self.params
        ds = flex.double(
            [2 * b1, 0, 0, 0, 2 * b1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * b2]
        )
        ds.reshape(flex.grid(2, 3, 3))
        return ds


class Angular4MosaicityParameterisation(BaseParameterisation):
    """
    A simple mosaicity parameterisation that uses 4 parameters to describe a
    multivariate normal angular mosaic spread. Sigma is enforced as positive
    definite by parameterising using the cholesky decomposition.
    W = | w1  0  0  |
        | w2 w3  0  |
        | 0   0 w4 |
    S = W*W^T
    """

    @staticmethod
    def is_angular() -> bool:
        return True

    @staticmethod
    def num_parameters() -> int:
        return 4

    def sigma(self) -> np.array:
        """
        Compute the covariance matrix of the MVN from the parameters
        """
        ab = self.params[0] * self.params[1]
        aa = self.params[0] ** 2
        bcsq = self.params[1] ** 2 + self.params[2] ** 2
        dd = self.params[3] ** 2
        return matrix.sqr((aa, ab, 0.0, ab, bcsq, 0, 0, 0, dd))

    def mosaicity(self) -> Dict:
        """Three components of mosaicity"""
        decomp = linalg.eigensystem.real_symmetric(self.sigma().as_flex_double_matrix())
        m = mosaicity_from_eigen_decomposition(decomp.values())
        v = decomp.vectors()
        mosaicities = {"radial": 0, "angular_0": 0, "angular_1": 0}
        n_angular = 0
        for i in range(3):
            vec = (v[i * 3], v[(i * 3) + 1], v[(i * 3) + 2])
            if vec == (0, 0, 1):
                mosaicities["radial"] = m[i]
            else:
                mosaicities["angular_" + str(n_angular)] = m[i]
                n_angular += 1
        return mosaicities

    def first_derivatives(self) -> np.array:
        """
        Compute the first derivatives of Sigma w.r.t the parameters
        """
        b1, b2, b3, b4 = self.params
        ds = flex.double(
            [
                2 * b1,
                b2,
                0,
                b2,
                0,
                0,
                0,
                0,
                0,
                0,
                b1,
                0,
                b1,
                2 * b2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2 * b3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2 * b4,
            ]
        )
        ds.reshape(flex.grid(4, 3, 3))
        return ds


class ModelState(object):
    """
    A class to keep track of the model state

    """

    def __init__(
        self,
        experiment,
        mosaicity_parameterisation,
        wavelength_parameterisation=None,
        fix_mosaic_spread=False,
        fix_wavelength_spread=True,
        fix_unit_cell=False,
        fix_orientation=False,
    ):
        """
        Initialise the model state

        """

        # Save the crystal model
        self.experiment = experiment
        self.crystal = experiment.crystal

        # The U and P parameterisation
        self._U_parameterisation = CrystalOrientationParameterisation(self.crystal)
        self._B_parameterisation = CrystalUnitCellParameterisation(self.crystal)

        # The M and L parameterisations
        self._M_parameterisation = mosaicity_parameterisation
        self._L_parameterisation = wavelength_parameterisation

        # Set the flags to fix parameters
        self._is_mosaic_spread_fixed = fix_mosaic_spread
        self._is_wavelength_spread_fixed = fix_wavelength_spread
        self._is_unit_cell_fixed = fix_unit_cell
        self._is_orientation_fixed = fix_orientation

        # Check wavelength parameterisation
        if not self.is_wavelength_spread_fixed:
            assert self._L_parameterisation is not None

    @property
    def is_orientation_fixed(self) -> bool:
        return self._is_orientation_fixed

    @property
    def is_unit_cell_fixed(self) -> bool:
        return self._is_unit_cell_fixed

    @property
    def is_mosaic_spread_fixed(self) -> bool:
        return self._is_mosaic_spread_fixed

    @property
    def is_mosaic_spread_angular(self) -> bool:
        return self._M_parameterisation.is_angular()

    @property
    def is_wavelength_spread_fixed(self) -> bool:
        return self._is_wavelength_spread_fixed

    @property
    def unit_cell(self):
        return self.crystal.get_unit_cell()

    @property
    def U_matrix(self) -> np.array:
        return self.crystal.get_U()

    @property
    def B_matrix(self) -> np.array:
        return self.crystal.get_B()

    @property
    def A_matrix(self) -> np.array:
        return matrix.sqr(self.crystal.get_A())

    @property
    def mosaicity_covariance_matrix(self) -> np.array:
        return self._M_parameterisation.sigma()

    @property
    def wavelength_spread(self) -> flex.double:
        if self._L_parameterisation is not None:
            return self._L_parameterisation.sigma()
        return flex.double()

    @property
    def U_params(self) -> np.array:
        """Get the parameters of the orientation parameterisation"""
        return np.array(self._U_parameterisation.get_param_vals(), dtype=np.float64)

    @U_params.setter
    def U_params(self, params) -> None:
        self._U_parameterisation.set_param_vals(tuple(float(i) for i in params))

    @property
    def B_params(self) -> np.array:
        """Get the parameters of the orientation parameterisation"""
        return np.array(self._B_parameterisation.get_param_vals(), dtype=np.float64)

    @B_params.setter
    def B_params(self, params) -> None:
        self._B_parameterisation.set_param_vals(tuple(float(i) for i in params))

    @property
    def M_params(self) -> np.array:
        "Parameters of the mosaicity parameterisation"
        return self._M_parameterisation.parameters

    @M_params.setter
    def M_params(self, params) -> None:
        self._M_parameterisation.parameters = params

    @property
    def L_params(self) -> np.array:
        "Parameters of the Lambda (wavelength) parameterisation"
        if self._L_parameterisation is not None:
            return np.array(self._L_parameterisation.parameters, dtype=np.float64)
        return np.array([])

    @L_params.setter
    def L_params(self, params: flex.double) -> None:
        if self._L_parameterisation is not None:
            self._L_parameterisation.parameters = params

    @property
    def dU_dp(self) -> np.array:
        """
        Get the first derivatives of U w.r.t its parameters

        """
        ds_dp = self._U_parameterisation.get_ds_dp()
        return flex.mat3_double(ds_dp)

    @property
    def dB_dp(self) -> np.array:
        """
        Get the first derivatives of B w.r.t its parameters

        """
        ds_dp = self._B_parameterisation.get_ds_dp()
        return flex.mat3_double(ds_dp)

    @property
    def dM_dp(self) -> np.array:
        """
        Get the first derivatives of M w.r.t its parameters

        """
        return self._M_parameterisation.first_derivatives()

    @property
    def dL_dp(self) -> flex.double:
        """
        Get the first derivatives of L w.r.t its parameters

        """
        if self._L_parameterisation is not None:
            return self._L_parameterisation.first_derivatives()
        return flex.double()

    @property
    def active_parameters(self) -> np.array:
        """
        The active parameters in order: U, B, M, L, W
        """
        active_params = []
        if not self.is_orientation_fixed:
            active_params.append(self.U_params)
        if not self.is_unit_cell_fixed:
            active_params.append(self.B_params)
        if not self.is_mosaic_spread_fixed:
            active_params.append(self.M_params)
        if not self.is_wavelength_spread_fixed:
            active_params.append(self.L_params)
        active_params = np.concatenate(active_params)
        assert len(active_params) > 0
        return active_params

    @active_parameters.setter
    def active_parameters(self, params) -> None:
        """
        Set the active parameters in order: U, B, M, L, W
        """
        if not self.is_orientation_fixed:
            n_U_params = len(self.U_params)
            temp = params[:n_U_params]
            params = params[n_U_params:]
            self.U_params = temp
        if not self.is_unit_cell_fixed:
            n_B_params = len(self.B_params)
            temp = params[:n_B_params]
            params = params[n_B_params:]
            self.B_params = temp
        if not self.is_mosaic_spread_fixed:
            n_M_params = self.M_params.size
            temp = params[:n_M_params]
            params = params[n_M_params:]
            self.M_params = np.array(temp)
        if not self.is_wavelength_spread_fixed:
            n_L_params = self.L_params.size
            temp = params[:n_L_params]
            self.L_params = temp

    @property
    def parameter_labels(self) -> List[str]:
        """
        Get the parameter labels

        """
        labels = []
        if not self.is_orientation_fixed:
            labels += [f"Crystal_U_{i}" for i in range(self.U_params.size)]
        if not self.is_unit_cell_fixed:
            labels += [f"Crystal_B_{i}" for i in range(self.B_params.size)]
        if not self.is_mosaic_spread_fixed:
            labels += [f"Mosaicity_{i}" for i in range(self.M_params.size)]
        if not self.is_wavelength_spread_fixed:
            labels.append("Wavelength_Spread")
        assert len(labels) > 0
        return labels


class ReflectionModelState(object):
    """
    Class to compute basic derivatives of Sigma and r w.r.t parameters

    """

    def __init__(self, state, s0, h):
        """
        Initialise with the state and compute derivatives

        """

        # Compute the reciprocal lattice vector
        self._h = h
        self._r = state.A_matrix * self._h
        self._s0 = np.array(s0, dtype=np.float64)
        self._norm_s0 = (self._s0 / norm(self._s0)).flatten()
        self._norm_s0_cctbx = matrix.col(
            (self._norm_s0[0], self._norm_s0[1], self._norm_s0[2])
        )

        self.state = state
        self._Q_cctbx = matrix.sqr((0, 0, 0, 0, 0, 0, 0, 0, 0))

        n_params = 0
        if not self.state.is_orientation_fixed:
            n_params += len(self.state.U_params)
        if not self.state.is_unit_cell_fixed:
            n_params += len(self.state.B_params)
        if not self.state.is_mosaic_spread_fixed:
            n_params += len(self.state.M_params)
        if not self.state.is_wavelength_spread_fixed:
            n_params += len(self.state.L_params)

        # The array of derivatives
        self._dr_dp = flex.vec3_double(n_params, (0, 0, 0))
        self._ds_dp = flex.double(flex.grid(3, 3, n_params), 0)
        self._dl_dp = flex.double(n_params)

        if self.state.is_mosaic_spread_angular:
            r = np.array(self._r)
            norm_r = r / norm(r)
            q1 = np.cross(norm_r, self._norm_s0)
            q1 /= norm(q1)
            q2 = np.cross(norm_r, q1)
            q2 /= norm(q2)
            self._Q_cctbx = matrix.sqr(
                (
                    q1[0],
                    q1[1],
                    q1[2],
                    q2[0],
                    q2[1],
                    q2[2],
                    norm_r[0],
                    norm_r[1],
                    norm_r[2],
                )
            )
            self._Q_cctbx_T = matrix.sqr(self._Q_cctbx.transpose())
        self._recalc_sigma()
        self._recalc_sigma_lambda()
        self.update()

    def _recalc_sigma(self):
        # Compute the covariance matrix
        M = self.state.mosaicity_covariance_matrix
        if self.state.is_mosaic_spread_angular:
            # Define rotation for W sigma components
            # check if r has actually been updated
            if (not self.state.is_orientation_fixed) or (
                not self.state.is_unit_cell_fixed
            ):
                r = np.array(self._r)
                norm_r = r / norm(r)
                q1 = np.cross(norm_r, self._norm_s0)
                q1 /= norm(q1)
                q2 = np.cross(norm_r, q1)
                q2 /= norm(q2)
                self._Q_cctbx = matrix.sqr(
                    (
                        q1[0],
                        q1[1],
                        q1[2],
                        q2[0],
                        q2[1],
                        q2[2],
                        norm_r[0],
                        norm_r[1],
                        norm_r[2],
                    )
                )
                self._Q_cctbx_T = matrix.sqr(self._Q_cctbx.transpose())
            self._sigma = (self._Q_cctbx_T * M) * self._Q_cctbx
        else:
            self._sigma = M

    def _recalc_sigma_lambda(self):
        # Get the wavelength spread

        assert len(self.state.wavelength_spread) == len(self.state.dL_dp)
        if len(self.state.wavelength_spread) == 0:
            self._sigma_lambda = 0
        else:
            assert len(self.state.wavelength_spread) == 1
            self._sigma_lambda = self.state.wavelength_spread[0]

    def update(self):
        "Updates r, sigma, derivatives based on latest model state"

        # Set the reciprocal lattice vector
        if (not self.state.is_orientation_fixed) or (not self.state.is_unit_cell_fixed):
            self._r = self.state.A_matrix * self._h
        if not self.state.is_wavelength_spread_fixed:
            self._recalc_sigma_lambda()
        if not self.state.is_mosaic_spread_fixed:
            self._recalc_sigma()

        # Compute derivatives w.r.t U parameters
        n_tot = 0
        state = self.state
        if (not state.is_orientation_fixed) or (not state.is_unit_cell_fixed):
            self._dr_dp = calc_dr_dp(
                self.state.dU_dp,
                self.state.dB_dp,
                self._h,
                state.B_matrix,
                state.U_matrix,
                state.is_orientation_fixed,
                state.is_unit_cell_fixed,
                self._dr_dp.size(),
            )
        if not state.is_orientation_fixed:
            n_tot += self.state.dU_dp.all()[0]
        if not state.is_unit_cell_fixed:
            n_tot += self.state.dB_dp.all()[0]

        # Compute derivatives w.r.t M parameters
        if not state.is_mosaic_spread_fixed:
            dM_dp = self.state.dM_dp
            n_M_params = dM_dp.all()[0]
            self._ds_dp = calc_ds_dp(
                dM_dp,
                self._Q_cctbx,
                n_tot,
                self._ds_dp.all()[2],
                state.is_mosaic_spread_angular,
            )
            n_tot += n_M_params
        # Compute derivatives   w.r.t L parameters
        if not state.is_wavelength_spread_fixed:
            self._dl_dp[n_tot] = self.state.dL_dp[0]

    @property
    def mosaicity_covariance_matrix(self) -> np.array:
        return self._sigma

    def get_r(self) -> np.array:
        """
        Return the reciprocal lattice vector

        """
        return self._r  # .reshape(3, 1)

    def get_dS_dp(self) -> np.array:
        """
        Return the derivatives of the covariance matrix (an array of size 3x3xn)

        """
        return self._ds_dp

    def get_dr_dp(self) -> np.array:
        """
        Return the derivatives of the reciprocal lattice vector ( an array of size 3xn)

        """
        return self._dr_dp

    @property
    def wavelength_spread(self) -> float:
        return self._sigma_lambda

    def get_dL_dp(self) -> flex.double:
        """
        Return the derivatives of the wavelength spread

        """
        return self._dl_dp
