from __future__ import annotations

import logging
import random
import textwrap
from math import pi, sqrt
from typing import List

import numpy as np
from numpy.linalg import norm

from dxtbx import flumpy
from scitbx import linalg, matrix

from dials.algorithms.profile_model.ellipsoid import (
    cpp_compute_dmbar,
    cpp_compute_dSbar,
    cpp_first_derivatives,
    cpp_fisher_information,
    cpp_log_likelihood,
    cpp_rotate_mat3_double,
    cpp_rotate_vec3_double,
    mosaicity_from_eigen_decomposition,
    reflection_statistics,
    rse,
)
from dials.algorithms.profile_model.ellipsoid.model import (
    compute_change_of_basis_operation,
)
from dials.algorithms.profile_model.ellipsoid.parameterisation import (
    ReflectionModelState,
)
from dials.array_family import flex
from dials.util import tabulate

logger = logging.getLogger("dials")

flex.set_random_seed(0)
random.seed(0)


class BadSpotForIntegrationException(Exception):
    pass


class ConditionalDistribution(object):
    """
    A class to compute useful stuff about the conditional distribution

    """

    def __init__(self, norm_s0, mu, dmu, S, dS):
        # norm_s0 is a float i.e. norm(s0)
        self._mu = mu  # 3x1 array
        self._dmu = dmu  #
        self._S = S
        self._dS = dS

        S22 = S[8]

        S12 = matrix.col((S[2], S[5]))
        S11 = matrix.sqr((S[0], S[1], S[3], S[4]))
        S21 = matrix.col((S[6], S[7]))

        # The partitioned mean vector
        mu2 = mu[2]

        # The epsilon
        self.epsilon = norm_s0 - mu2

        # Compute the conditional mean
        self._mubar = (
            mu[0] + (S[2] * self.epsilon / S22),
            mu[1] + (S[5] * self.epsilon / S22),
        )

        # Compute the conditional covariance matrix
        self._Sbar = S11 - (
            matrix.sqr(
                (S12[0] * S21[0], S12[0] * S21[1], S12[1] * S21[0], S12[1] * S21[1])
            )
            / S22
        )

        # Set to None and compute on demand
        self.dSbar = None
        self.dmbar = None
        self.d2Sbar = None
        self.d2mbar = None

    def mean(self) -> np.array:
        """
        Return the conditional mean (a 2x1 array)

        """
        return self._mubar

    def sigma(self) -> np.array:
        """
        Return the conditional sigma (a 2x2 array)

        """
        return self._Sbar

    def first_derivatives_of_sigma(self) -> List[np.array]:
        """
        Return the marginal first derivatives (as a list of 2x2 arrays)

        """
        if self.dSbar is None:
            self.dSbar = cpp_compute_dSbar(self._S, self._dS)

        return self.dSbar

    def first_derivatives_of_mean(self) -> List[np.array]:
        """
        Return the marginal first derivatives (a list of 2x1 arrays)

        """
        if self.dmbar is None:
            self.dmbar = cpp_compute_dmbar(self._S, self._dS, self._dmu, self.epsilon)

        return self.dmbar


def rotate_vec3_double(R, A):
    """
    Helper function to rotate an array of matrices

    """
    return np.einsum("ij,jk->ik", R, A)


def rotate_mat3_double(R, A):
    """
    Helper function to rotate an array of matrices

    """
    return np.einsum("ij,jkv,kl->ilv", R, A, R.T)


class ReflectionLikelihood(object):
    def __init__(self, model, s0, sp, h, ctot, mobs, sobs):

        # Save stuff
        modelstate = ReflectionModelState(model, s0, h)
        self.modelstate = modelstate
        self.s0_orig = matrix.col((s0[0, 0], s0[1, 0], s0[2, 0]))
        self.s0 = s0.reshape(3, 1)
        self.norm_s0 = float(norm(s0))
        self.sp = sp.reshape(3, 1)
        self.h = np.array([h], dtype=np.float64).reshape(3, 1)
        self.ctot = ctot
        self.mobs = mobs
        self.sobs = sobs

        # Compute the change of basis
        self.R = compute_change_of_basis_operation(self.s0, self.sp)  # const
        s2 = matrix.col(
            (
                self.s0_orig[0] + self.modelstate.get_r()[0],
                self.s0_orig[1] + self.modelstate.get_r()[1],
                self.s0_orig[2] + self.modelstate.get_r()[2],
            )
        )
        self.R_cctbx = matrix.sqr(self.R.flatten())
        self.R_cctbx_T = self.R_cctbx.transpose()
        # Rotate the mean vector
        self.mu = self.R_cctbx * s2

        self.S = (
            self.R_cctbx * modelstate.mosaicity_covariance_matrix
        ) * self.R_cctbx_T

        self.dS = cpp_rotate_mat3_double(
            self.R_cctbx, modelstate.get_dS_dp()
        )  # const when not refining mosaicity?
        self.dmu = cpp_rotate_vec3_double(
            self.R_cctbx, modelstate.get_dr_dp()
        )  # const when not refining uc/orientation?
        # Construct the conditional distribution
        self.conditional = ConditionalDistribution(
            self.norm_s0, self.mu, self.dmu, self.S, self.dS
        )

    def update(self):

        # The s2 vector
        r = self.modelstate.get_r()
        s2 = (self.s0_orig[0] + r[0], self.s0_orig[1] + r[1], self.s0_orig[2] + r[2])
        # Rotate the mean vector
        self.mu = self.R_cctbx * s2

        # Rotate the covariance matrix
        if not self.modelstate.state.is_mosaic_spread_fixed:
            self.S = (
                self.R_cctbx * self.modelstate.mosaicity_covariance_matrix
            ) * self.R_cctbx_T

        # Rotate the first derivative matrices
        if not self.modelstate.state.is_mosaic_spread_fixed:
            self.dS = cpp_rotate_mat3_double(
                self.R_cctbx, self.modelstate.get_dS_dp()
            )  # const when not refining mosaicity?

        # Rotate the first derivative of s2
        if (not self.modelstate.state.is_unit_cell_fixed) or not (
            self.modelstate.state.is_orientation_fixed
        ):
            self.dmu = cpp_rotate_vec3_double(
                self.R_cctbx, self.modelstate.get_dr_dp()
            )  # const when not refining uc/orientation?

        # Construct the conditional distribution
        self.conditional = ConditionalDistribution(
            self.norm_s0, self.mu, self.dmu, self.S, self.dS
        )

    def log_likelihood(self):
        """
        Compute the log likelihood for the reflection

        """

        # Get data
        ctot = self.ctot
        mobs = self.mobs
        Sobs = self.sobs

        # Get info about the marginal
        S22 = self.S[8]
        mu2 = self.mu[2]

        # Get info about the conditional
        Sbar = self.conditional.sigma()
        mubar = self.conditional.mean()

        return cpp_log_likelihood(
            ctot,
            mobs,
            Sobs,
            Sbar,
            mubar,
            mu2,
            S22,
            self.norm_s0,
        )

    def first_derivatives(self):
        """
        Compute the first derivatives

        """
        # Get data
        ctot = self.ctot
        mobs = self.mobs  # 2x1 array
        Sobs = self.sobs

        # Get info about marginal distribution
        S22 = self.S[8]
        dS22 = flex.double([self.dS[2, 2, i] for i in range(self.dS.all()[2])])
        mu2 = self.mu[2]

        # Get info about conditional distribution
        Sbar = self.conditional.sigma()  # 3x3 array
        mubar = self.conditional.mean()  # 2x1 array
        dSbar_flex = self.conditional.first_derivatives_of_sigma()  # list of 2x2 arrays
        dmbar_flex = self.conditional.first_derivatives_of_mean()  # list of 2x1 arrays

        # The distance from the ewald sphere
        dep = -1.0 * self.dmu.parts()[2]

        V_CPP = cpp_first_derivatives(
            ctot,
            mobs,
            Sobs,
            S22,
            dS22,
            mu2,
            self.norm_s0,
            Sbar,
            mubar,
            dSbar_flex,
            dmbar_flex,
            dep,
        )
        return -0.5 * V_CPP

    def fisher_information(self):
        """
        Compute the fisher information

        """
        ctot = self.ctot

        # Get info about marginal distribution
        S22 = self.S[8]
        dS22 = flex.double([self.dS[2, 2, i] for i in range(self.dS.all()[2])])

        # Get info about conditional distribution
        Sbar = self.conditional.sigma()  # 2x2 array
        dSbar_flex = self.conditional.first_derivatives_of_sigma()  # list of 2x2 arrays
        dmbar_flex = self.conditional.first_derivatives_of_mean()  # list of 2x1 arrays
        dmu2 = self.dmu.parts()[2]
        dS22 = flex.double(dS22)

        return cpp_fisher_information(
            ctot, S22, dS22, Sbar, dmu2, dSbar_flex, dmbar_flex
        )


class MaximumLikelihoodTarget(object):
    def __init__(self, model, s0, sp_list, h_list, ctot_list, mobs_list, sobs_list):

        # Check input
        assert len(h_list) == sp_list.shape[-1]
        assert len(h_list) == ctot_list.shape[-1]
        assert len(h_list) == len(mobs_list)
        assert len(h_list) == len(sobs_list)

        # Save the model
        self.model = model

        # Compute the change of basis for each reflection
        self.data = []

        for i in range(len(h_list)):
            self.data.append(
                ReflectionLikelihood(
                    model,
                    s0,
                    sp_list[:, i],
                    h_list[i],
                    ctot_list[i],
                    mobs_list[i],
                    sobs_list[i],
                )
            )

    def update(self):
        for d in self.data:
            d.modelstate.update()  # update the ReflectionModelState
            d.update()  # update the ReflectionLikelihood

    def mse(self):
        """
        The MSE in local reflection coordinates

        """
        mse = 0
        for i in range(len(self.data)):
            mbar = self.data[i].conditional.mean()
            xobs = self.data[i].mobs
            mse += (xobs[0] - mbar[0]) ** 2 + (xobs[1] - mbar[1]) ** 2
            # mse += np.dot((xobs - mbar).T, xobs - mbar)
        mse /= len(self.data)
        return mse

    def rmsd(self):
        """
        The RMSD in pixels

        """
        mse_x = 0.0
        mse_y = 0.0
        for i in range(len(self.data)):
            R = self.data[
                i
            ].R_cctbx  # matrix.sqr(flex.double(self.data[i].R.tolist()))  # (3x3 numpy array)
            mbar = self.data[i].conditional.mean()  # 2x1 array
            xobs = self.data[i].mobs  # # 2x1 array
            norm_s0 = self.data[i].norm_s0
            # xobs = (xobs[0, 0], xobs[1, 0])
            # mbar = (mbar[0, 0], mbar[1, 0])
            rse_i = rse(R, mbar, xobs, norm_s0, self.model.experiment.detector)
            mse_x += rse_i[0]
            mse_y += rse_i[1]
        mse_x /= len(self.data)
        mse_y /= len(self.data)
        return np.sqrt(np.array([mse_x, mse_y]))

    def log_likelihood(self):
        """
        The joint log likelihood

        """
        return sum(d.log_likelihood() for d in self.data)

    def jacobian(self):
        """
        Return the Jacobian

        """
        return flex.double([list(d.first_derivatives()) for d in self.data])

    def first_derivatives(self):
        """
        The joint first derivatives

        """
        dL = 0
        for d in self.data:
            dL += d.first_derivatives()
        return dL

    def fisher_information(self):
        """
        The joint fisher information

        """
        return sum(d.fisher_information() for d in self.data)


def line_search(func, x, p, tau=0.5, delta=1.0, tolerance=1e-7):
    """
    Perform a line search
    :param func The function to minimize
    :param x The initial position
    :param p The direction to search
    :param tau: The backtracking parameter
    :param delta: The initial step
    :param tolerance: The algorithm tolerance
    :return: The amount to move in the given direction

    """
    fa = func(x)
    if p.length() < 1:
        min_delta = tolerance
    else:
        min_delta = tolerance / p.length()

    while delta > min_delta:
        try:
            fb = func(x + delta * p)
            if fb <= fa:
                return delta
        except Exception:
            pass
        delta *= tau
    return 0


def gradient_descent(f, df, x0, max_iter=1000, tolerance=1e-10):
    """
    Find the minimum using gradient descent and a line search
    :param f The function to minimize
    :param df The function to compute derivatives
    :param x0 The initial position
    :param max_iter: The maximum number of iterations
    :param tolerance: The algorithm tolerance
    :return: The amount to move in the given direction

    """
    delta = 0.5
    for it in range(max_iter):
        p = -matrix.col(df(x0))
        delta = line_search(f, x0, p, delta=min(1.0, delta * 2), tolerance=tolerance)
        x = x0 + delta * p
        assert f(x) <= f(x0)
        if (x - x0).length() < tolerance:
            break
        x0 = x
    return x


class FisherScoringMaximumLikelihoodBase(object):
    """
    A class to solve maximum likelihood equations using fisher scoring

    """

    def __init__(self, x0, max_iter=1000, tolerance=1e-7):
        """
        Configure the algorithm

        :param x0: The initial parameter estimates
        :param max_iter: The maximum number of iterations
        :param tolerance: The parameter tolerance

        """
        self.x0 = matrix.col(x0)
        self.max_iter = max_iter
        self.tolerance = tolerance

    def solve(self):
        """
        Find the maximum likelihood estimate

        """
        x0 = self.x0

        # Loop through the maximum number of iterations
        for it in range(self.max_iter):
            # Compute the derivative and fisher information at x0
            S, I = self.score_and_fisher_information(x0)

            # Solve the update equation to get direction
            p = matrix.col(self.solve_update_equation(S, I))

            # Perform a line search to ensure that each step results in an increase the
            # in log likelihood. In the rare case where the update does not result in an
            # increase in the likelihood (only observed for absurdly small samples
            # e.g. 2 reflections or when 1 parameter approaches zero) do an iteration
            # of gradient descent
            delta = self.line_search(x0, p)
            if delta > 0:
                x = x0 + delta * p
            else:
                x = self.gradient_search(x0)

            # Call an update
            self.callback(x)
            # Break the loop if the parameters change less than the tolerance
            if (x - x0).length() < self.tolerance:
                break

            # Update the parameter
            x0 = x

        # Save the parameters
        self.num_iter = it + 1
        self.parameters = x

    def solve_update_equation(self, S, I):
        """
        Solve the update equation using cholesky decomposition
        :param S: The score
        :param I: The fisher information
        :return: The parameter delta

        """

        # Construct triangular matrix
        LL = flex.double()
        for j in range(len(S)):
            for i in range(j + 1):
                LL.append(I[j * len(S) + i])

        # Perform the decomposition
        ll = linalg.l_l_transpose_cholesky_decomposition_in_place(LL)
        p = flex.double(S)
        return ll.solve(p)

    def line_search(self, x, p, tau=0.5, delta=1.0, tolerance=1e-7):
        """
        Perform a line search
        :param x The initial position
        :param p The direction to search
        :return: The amount to move in the given direction

        """

        def f(x):
            return -self.log_likelihood(x)

        return line_search(f, x, p, tolerance=self.tolerance)

    def gradient_search(self, x0):
        """
        Find the minimum using gradient descent and a line search
        :param x0 The initial position
        :return: The amount to move in the given direction

        """

        def f(x):
            return -self.log_likelihood(x)

        def df(x):
            return -self.score(x)

        return gradient_descent(f, df, x0, max_iter=1, tolerance=self.tolerance)


class FisherScoringMaximumLikelihood(FisherScoringMaximumLikelihoodBase):
    """
    A class to solve the maximum likelihood equations

    """

    def __init__(
        self,
        model,
        s0,
        sp_list,
        h_list,
        ctot_list,
        mobs_list,
        sobs_list,
        max_iter=1000,
        tolerance=1e-7,
    ):
        """
        Initialise the algorithm:

        """
        # Initialise the super class
        super(FisherScoringMaximumLikelihood, self).__init__(
            model.active_parameters, max_iter=max_iter, tolerance=tolerance
        )

        # Save the parameterisation
        self.model = model

        # Save some stuff
        self.s0 = s0
        self.sp_list = sp_list
        self.h_list = h_list
        self.ctot_list = ctot_list
        self.mobs_list = mobs_list
        self.sobs_list = sobs_list

        # Store the parameter history
        self.history = []

        self._ml_target = MaximumLikelihoodTarget(
            self.model,
            self.s0,
            self.sp_list,
            self.h_list,
            self.ctot_list,
            self.mobs_list,
            self.sobs_list,
        )

        # Print initial
        self.callback(self.model.active_parameters)

    def log_likelihood(self, x):
        """
        :param x: The parameter estimate
        :return: The log likelihood at x

        """
        self.model.active_parameters = x
        self._ml_target.update()
        return self._ml_target.log_likelihood()

    def score(self, x):
        """
        :param x: The parameter estimate
        :return: The score at x

        """
        self.model.active_parameters = x
        self._ml_target.update()
        return flumpy.from_numpy(self._ml_target.first_derivatives())

    def score_and_fisher_information(self, x):
        """
        :param x: The parameter estimate
        :return: The score and fisher information at x

        """
        self.model.active_parameters = x
        self._ml_target.update()
        S = flumpy.from_numpy(self._ml_target.first_derivatives())
        I = self._ml_target.fisher_information()
        return S, I

    def mse(self, x):
        """
        :param x: The parameter estimate
        :return: The MSE at x

        """
        return self._ml_target.mse()

    def rmsd(self, x):
        """
        :param x: The parameter estimate
        :return: The RMSD at x

        """
        return self._ml_target.rmsd()

    def jacobian(self, x):
        """
        :param x: The parameter estimate
        :return: The Jacobian at x

        """
        return self._ml_target.jacobian()

    def condition_number(self, x):
        """
        The condition number of the Jacobian

        """
        from scitbx.linalg.svd import real as svd_real

        svd = svd_real(self.jacobian(x), False, False)
        return max(svd.sigma) / min(svd.sigma)

    def correlation(self, x):
        """
        The correlation of the Jacobian

        """
        J = self.jacobian(x)
        C = flex.double(flex.grid(J.all()[1], J.all()[1]))
        for j in range(C.all()[0]):
            for i in range(C.all()[1]):
                a = J[:, i : i + 1].as_1d()
                b = J[:, j : j + 1].as_1d()
                C[j, i] = flex.linear_correlation(a, b).coefficient()
        return C

    def callback(self, x):
        """
        Handle and update in parameter values

        """
        self.model.active_parameters = x
        self._ml_target.update()
        lnL = self._ml_target.log_likelihood()
        mse = self._ml_target.mse()
        rmsd = self._ml_target.rmsd()

        # Get the unit cell
        unit_cell = self.model.unit_cell.parameters()

        # Get some matrices
        U = list(self.model.U_matrix)
        M = list(self.model.mosaicity_covariance_matrix)

        # Print some information
        format_string1 = "  Unit cell: (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)"
        format_string2 = "  | % .6f % .6f % .6f |"
        format_string3 = "  | % .2e % .2e % .2e |"

        logger.info(f"\nIteration: {len(self.history)}")
        if not self.model.is_unit_cell_fixed:
            logger.info("\n" + format_string1 % unit_cell)
        if not self.model.is_orientation_fixed:
            logger.info(
                "\n".join(
                    [
                        "",
                        "  U matrix (orientation)",
                        format_string2 % tuple(U[0:3]),
                        format_string2 % tuple(U[3:6]),
                        format_string2 % tuple(U[6:9]),
                    ]
                )
            )
        if not self.model.is_mosaic_spread_fixed:
            logger.info(
                "\n".join(
                    [
                        "",
                        "  Sigma M",
                        format_string3 % tuple(M[0:3]),
                        format_string3 % tuple(M[3:6]),
                        format_string3 % tuple(M[6:9]),
                    ]
                )
            )

        logger.info(
            "\n".join(
                [
                    "",
                    "  ln(L) = %f" % lnL,
                    "",
                    "  R.M.S.D (local) = %.5g" % sqrt(mse),
                    "",
                    "  R.M.S.D (pixel): X = %.3f, Y = %.3f" % tuple(rmsd),
                ]
            )
        )

        # Append the parameters to the history
        self.history.append(
            {
                "parameters": list(x),
                "likelihood": lnL,
                "unit_cell": unit_cell,
                "orientation": list(U),
                "rlp_mosaicity": list(M),
                "rmsd": tuple(rmsd),
            }
        )


class Refiner(object):
    """
    High level profile refiner class that handles book keeping etc

    """

    def __init__(self, state, data):
        """
        Set the data and initial parameters

        """
        self.s0 = data.s0
        self.h_list = data.h_list
        self.sp_list = data.sp_list
        self.ctot_list = data.ctot_list
        self.mobs_list = data.mobs_list
        self.sobs_list = data.sobs_list
        self.state = state
        self.history = []

    def refine(self):
        """
        Perform the profile refinement

        """
        self.refine_fisher_scoring()

    def refine_fisher_scoring(self):
        """
        Perform the profile refinement

        """

        # Print information
        logger.info("\nComponents to refine:")
        logger.info(" Orientation:       %s" % (not self.state.is_orientation_fixed))
        logger.info(" Unit cell:         %s" % (not self.state.is_unit_cell_fixed))
        logger.info(" RLP mosaicity:     %s" % (not self.state.is_mosaic_spread_fixed))
        logger.info(
            " Wavelength spread: %s\n" % (not self.state.is_wavelength_spread_fixed)
        )

        # Initialise the algorithm
        self.ml = FisherScoringMaximumLikelihood(
            self.state,
            self.s0,
            self.sp_list,
            self.h_list,
            self.ctot_list,
            self.mobs_list,
            self.sobs_list,
        )

        # Solve the maximum likelihood equations
        self.ml.solve()

        # Get the parameters
        self.parameters = flex.double(self.ml.parameters)

        # set the parameters
        self.state.active_parameters = self.parameters

        # Print summary table of refinement.
        rows = []
        headers = ["Iteration", "likelihood", "RMSD (pixel) X,Y"]
        for i, h in enumerate(self.ml.history):
            l = h["likelihood"]
            rmsd = h["rmsd"]
            rows.append([str(i), f"{l:.4f}", f"{rmsd[0]:.3f}, {rmsd[1]:.3f}"])
        logger.info(
            "\nRefinement steps:\n\n" + textwrap.indent(tabulate(rows, headers), " ")
        )

        # Print the eigen values and vectors of sigma_m
        if not self.state.is_mosaic_spread_fixed:
            logger.info("\nDecomposition of Sigma_M:")
            print_eigen_values_and_vectors(self.state.mosaicity_covariance_matrix)

        # Save the history
        self.history = self.ml.history

        # Return the optimizer
        return self.ml

    def correlation(self):
        """
        Return the correlation matrix between parameters

        """
        return self.ml.correlation(self.state.active_parameters)

    def labels(self):
        """
        Return parameter labels

        """
        return self.state.parameter_labels


"""def calc_values(panel, xyzobs, s0_length, s0, sbox):
    # The vector to the pixel centroid
    sp = np.array(panel.get_pixel_lab_coord(xyzobs[0:2]), dtype=np.float64).reshape(
        3, 1
    )
    sp *= s0_length / norm(sp)

    # Compute change of basis
    R = compute_change_of_basis_operation(s0, sp)

    # Get data and compute total counts
    data = sbox.data
    mask = sbox.mask
    bgrd = sbox.background

    # Get array of vectors
    i0 = sbox.bbox[0]
    j0 = sbox.bbox[2]
    assert data.all()[0] == 1
    X = np.zeros(shape=(data.all()[1:]) + (2,), dtype=np.float64)
    C = np.zeros(shape=data.all()[1:], dtype=np.float64)
    ctot = 0
    for j in range(data.all()[1]):
        for i in range(data.all()[2]):
            c = data[0, j, i] - bgrd[0, j, i]
            # print(data[0, j, i], bgrd[0, j, i])
            if mask[0, j, i] & (1 | 4) == (1 | 4) and c > 0:
                ctot += c
                ii = i + i0
                jj = j + j0
                s = np.array(
                    panel.get_pixel_lab_coord((ii + 0.5, jj + 0.5)),
                    dtype=np.float64,
                ).reshape(3, 1)
                s *= s0_length / norm(s)
                e = np.matmul(R, s)
                X[j, i, :] = e[0:2, 0]
                C[j, i] = c

    # Check we have a sensible number of counts
    if ctot <= 0:
        raise BadSpotForIntegrationException(
            "Strong spot found with <= 0 counts! Check spotfinding results"
        )

    # Compute the mean vector
    C = np.expand_dims(C, axis=2)
    xbar = C * X
    xbar = xbar.reshape(-1, xbar.shape[-1])
    xbar = xbar.sum(axis=0)
    xbar /= ctot

    xbar = xbar.reshape(2, 1)

    # Compute the covariance matrix
    Sobs = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            x = np.array(X[j, i], dtype=np.float64).reshape(2, 1)
            Sobs += np.matmul(x - xbar, (x - xbar).T) * C[j, i, 0]

    Sobs /= ctot
    if (Sobs[0, 0] <= 0) or (Sobs[1, 1] <= 0):
        raise BadSpotForIntegrationException(
            "Strong spot variance <= 0. Check spotfinding results"
        )
    return sp, ctot, xbar, Sobs"""


class RefinerData(object):
    """
    A class for holding the data needed for the profile refinement

    """

    def __init__(self, s0, sp_list, h_list, ctot_list, mobs_list, sobs_list):
        """
        Init the data

        ctot_list is a list of total counts per reflection

        """
        self.s0 = s0
        self.sp_list = sp_list
        self.h_list = h_list
        self.ctot_list = ctot_list
        self.mobs_list = mobs_list
        self.sobs_list = sobs_list

    @classmethod
    def from_reflections(self, experiment, reflections):
        """
        Generate the required data from the reflections

        """

        # Get the beam vector
        s0 = np.array([experiment.beam.get_s0()], dtype=np.float64).reshape(3, 1)

        # Get the reciprocal lattice vector
        h_list = reflections["miller_index"]

        # Initialise the list of observed intensities and covariances
        sp_list = np.zeros(shape=(3, len(h_list)))
        ctot_list = np.zeros(shape=(len(h_list)))
        mobs_list = flex.vec2_double(len(h_list))  # np.zeros(shape=(2, len(h_list)))
        Sobs_list = [0] * len(h_list)  # np.zeros(shape=(2, 2, len(h_list)))

        logger.info(
            "Computing observed covariance for %d reflections" % len(reflections)
        )
        s0_length = float(norm(s0))
        s0 = experiment.beam.get_s0()
        assert len(experiment.detector) == 1
        panel = experiment.detector[0]
        sbox = reflections["shoebox"]
        xyzobs = reflections["xyzobs.px.value"]
        for r in range(len(reflections)):

            sp, ctot, xbar, Sobs = reflection_statistics(
                panel, xyzobs[r], s0_length, s0, sbox[r]
            )
            # print(sp, ctot, xbar, Sobs)
            # s0 = np.array([experiment.beam.get_s0()], dtype=np.float64).reshape(3, 1)
            # sp, ctot, xbar, Sobs = calc_values(panel, xyzobs[r], s0_length, s0, sbox[r])
            # print(sp, ctot, xbar, Sobs)
            # assert 0
            # Add to the lists
            sp_list[:, r] = sp  # [:, 0]
            ctot_list[r] = ctot
            mobs_list[r] = xbar
            Sobs_list[r] = Sobs  # [:, 0]
            # print(type(Sobs))
            # print(type(sp))
            # print(type(ctot))
            # print(type(xbar))
            """Sobs_list[0, 0, r] = Sobs[0]
            Sobs_list[0, 1, r] = Sobs[1]
            Sobs_list[1, 0, r] = Sobs[2]
            Sobs_list[1, 1, r] = Sobs[3]"""

        # Print some information
        logger.info("")
        logger.info(
            "I_min = %.2f, I_max = %.2f" % (np.min(ctot_list), np.max(ctot_list))
        )
        s0 = np.array([experiment.beam.get_s0()], dtype=np.float64).reshape(3, 1)

        # Sometimes a single reflection might have an enormouse intensity for
        # whatever reason and since we weight by intensity, this can cause the
        # refinement to be dominated by these reflections. Therefore, if the
        # intensity is greater than some value, damp the weighting accordingly
        def damp_outlier_intensity_weights(ctot_list):
            n = ctot_list.size
            sorted_ctot = np.sort(ctot_list)
            Q1 = sorted_ctot[n // 4]
            Q2 = sorted_ctot[n // 2]
            Q3 = sorted_ctot[3 * n // 4]
            IQR = Q3 - Q1
            T = Q3 + 1.5 * IQR
            logger.info(f"Median I = {Q2:.2f}\nQ1/Q3 I = {Q1:.2f}, {Q3:.2f}")
            logger.info(f"Damping effect of intensities > {T:.2f}")
            ndamped = 0
            for i, ctot in enumerate(ctot_list):
                if ctot > T:
                    logger.debug(f"Damping {ctot:.2f}")
                    ctot_list[i] = T
                    ndamped += 1
            logger.info(f"Damped {ndamped}/{n} reflections")
            return ctot_list

        ctot_list = damp_outlier_intensity_weights(ctot_list)

        # Print the mean covariance
        """Smean = np.mean(Sobs_list, axis=2)
        logger.info("")
        logger.info("Mean observed covariance:")
        logger.info(print_matrix_np(Smean))
        print_eigen_values_and_vectors_of_observed_covariance(Smean, s0)"""

        # Compute the distance from the Ewald sphere
        epsilon = flex.double(
            s0_length - matrix.col(s).length() for s in reflections["s2"]
        )
        mv = flex.mean_and_variance(epsilon)
        logger.info("")
        logger.info("Mean distance from Ewald sphere: %.3g" % mv.mean())
        logger.info(
            "Variance in distance from Ewald sphere: %.3g"
            % mv.unweighted_sample_variance()
        )

        # Return the profile refiner data
        return RefinerData(s0, sp_list, h_list, ctot_list, mobs_list, Sobs_list)


def print_eigen_values_and_vectors_of_observed_covariance(A, s0):
    """
    Print the eigen values and vectors of a matrix

    """

    # Compute the eigen decomposition of the covariance matrix
    A = matrix.sqr(flumpy.from_numpy(A))
    s0 = matrix.col(flumpy.from_numpy(s0))
    eigen_decomposition = linalg.eigensystem.real_symmetric(A.as_flex_double_matrix())
    Q = matrix.sqr(eigen_decomposition.vectors())
    L = matrix.diag(eigen_decomposition.values())

    # Print the matrix eigen values
    logger.info(f"\nEigen Values:\n{print_matrix(L, indent=2)}\n")
    logger.info(f"\nEigen Vectors:\n{print_matrix(Q, indent=2)}\n")

    logger.info("Observed covariance in degrees equivalent units")
    logger.info("C1: %.5f degrees" % (sqrt(L[0]) * (180.0 / pi) / s0.length()))
    logger.info("C2: %.5f degrees" % (sqrt(L[3]) * (180.0 / pi) / s0.length()))


def print_eigen_values_and_vectors(A):
    """
    Print the eigen values and vectors of a matrix

    """

    # Compute the eigen decomposition of the covariance matrix
    eigen_decomposition = linalg.eigensystem.real_symmetric(A.as_flex_double_matrix())
    eigen_values = eigen_decomposition.values()

    # Print the matrix eigen values
    logger.info(
        f"\n Eigen Values:\n{print_matrix(matrix.diag(eigen_values), indent=2)}\n"
    )
    logger.info(
        f"\n Eigen Vectors:\n{print_matrix(matrix.sqr(eigen_decomposition.vectors()), indent=2)}\n"
    )

    mosaicity = mosaicity_from_eigen_decomposition(eigen_values)
    logger.info(
        f"""
 Mosaicity in degrees equivalent units:
 M1 : {mosaicity[0]:.5f} degrees
 M2 : {mosaicity[1]:.5f} degrees
 M3 : {mosaicity[2]:.5f} degrees
"""
    )


def print_matrix_np(A, fmt="%.3g", indent=0):
    """
    Pretty print matrix

    """
    t = [fmt % a for a in A.flatten()]
    l = [len(tt) for tt in t]
    max_l = max(l)
    fmt = "%" + ("%d" % (max_l + 1)) + "s"
    prefix = " " * indent
    lines = []
    for j in range(A.shape[0]):
        line = ""
        for i in range(A.shape[1]):
            line += fmt % t[i + j * A.shape[1]]
        lines.append("%s|%s|" % (prefix, line))
    return "\n".join(lines)


def print_matrix(A, fmt="%.3g", indent=0):
    """
    Pretty print matrix

    """
    t = [fmt % a for a in A]
    l = [len(tt) for tt in t]
    max_l = max(l)
    fmt = "%" + ("%d" % (max_l + 1)) + "s"
    prefix = " " * indent
    lines = []
    for j in range(A.n[0]):
        line = ""
        for i in range(A.n[1]):
            line += fmt % t[i + j * A.n[1]]
        lines.append("%s|%s|" % (prefix, line))
    return "\n".join(lines)
