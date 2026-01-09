"""Methods for symmetry determination from partial datasets.

This module implements the methods of `Gildea, R. J. & Winter, G. (2018).
Acta Cryst. D74, 405-410 <https://doi.org/10.1107/S2059798318002978>`_ for
determination of Patterson group symmetry from sparse multi-crystal data sets in
the presence of an indexing ambiguity.
"""

from __future__ import annotations

import json
import logging
import math
from typing import List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

import iotbx.phil
from cctbx import miller, sgtbx
from cctbx.sgtbx.lattice_symmetry import metric_subgroups
from libtbx import Auto
from scitbx import matrix
from scitbx.array_family import flex

import dials.util
import dials.util.system
from dials.algorithms.indexing.symmetry import find_matching_symmetry
from dials.algorithms.symmetry import median_unit_cell, symmetry_base
from dials.algorithms.symmetry.cosym import engine as cosym_engine
from dials.algorithms.symmetry.cosym import target
from dials.algorithms.symmetry.laue_group import ScoreCorrelationCoefficient
from dials.util.observer import Subject
from dials.util.reference import intensities_from_reference_file

logger = logging.getLogger(__name__)

# these parameters are only required if running the cosym procedure in symmetry analysis mode.
symmetry_analysis_phil = """
lattice_group = None
  .type = space_group
  .short_caption = "Lattice group"

space_group = None
  .type = space_group
  .short_caption = "Space group"

lattice_symmetry_max_delta = 5.0
  .type = float(value_min=0)
  .short_caption = "Lattice symmetry max δ"

best_monoclinic_beta = True
  .type = bool
  .help = "If True, then for monoclinic centered cells, I2 will be preferred over C2 if"
          "it gives a less oblique cell (i.e. smaller beta angle)."
  .short_caption = "Best monoclinic β"
"""

# these parameters are required for the core cosym procedure for e.g. non-symmetry based isomorphism analysis
cosym_scope = """
min_reflections = 10
  .type = int(value_min=0)
  .help = "The minimum number of merged reflections per experiment required to perform cosym analysis."

seed = 230
  .type = int(value_min=0)

normalisation = kernel quasi *ml_iso ml_aniso
  .type = choice

d_min = Auto
  .type = float(value_min=0)

min_i_mean_over_sigma_mean = 4
  .type = float(value_min=0)
  .short_caption = "Minimum <I>/<σ>"

min_cc_half = 0.6
  .type = float(value_min=0, value_max=1)
  .short_caption = "Minimum CC½"

dimensions = Auto
  .type = int(value_min=2)
  .short_caption = "Dimensions"

use_curvatures = True
  .type = bool
  .short_caption = "Use curvatures"

weights = *count standard_error
  .type = choice
  .short_caption = "Weights"
  .help = "If not None, a weights matrix is used in the cosym procedure."
          "weights=count uses the number of reflections used to calculate a pairwise correlation coefficient as its weight"
          "weights=standard_error uses the reciprocal of the standard error as the weight. The standard error is given by"
          "(1-CC*2)/sqrt(N), where N=(n-2) or N=(neff-1) depending on the cc_weights option."
cc_weights = None sigma
  .type = choice
  .help = "If not None, a weighted cc-half formula is used for calculating pairwise correlation coefficients and degrees of"
          "freedom in the cosym procedure."
          "weights=sigma uses the intensity uncertainties to perform inverse variance weighting during the cc calculation."

min_pairs = 3
  .type = int(value_min=1)
  .help = 'Minimum number of pairs for inclusion of correlation coefficient in calculation of Rij matrix.'
  .short_caption = "Minimum number of pairs"

minimization
  .short_caption = "Minimization"
{
  engine = *scitbx scipy
    .type = choice
    .short_caption = "Engine"
  max_iterations = 100
    .type = int(value_min=0)
    .short_caption = "Maximum number of iterations"
  max_calls = None
    .type = int(value_min=0)
    .short_caption = "Maximum number of calls"
}

nproc = Auto
  .type = int(value_min=1)
  .help = "Number of processes"
"""


phil_scope = iotbx.phil.parse(
    f"""\
{cosym_scope}
{symmetry_analysis_phil}
""",
    process_includes=True,
)


def piecewise_constant_bic(y: np.ndarray) -> Tuple[int, float]:
    """
    Single-changepoint detection by Bayesian Information Criterion (BIC) for a piecewise-constant model.

    Returns:
      - k_best (int, 0-based): index of the first point in the second segment (change between k-1 and k)
      - delta_bic (float): BIC(no-change) - BIC(two-segment); larger => stronger evidence

    Notes:
      - SSE-based Gaussian-likelihood approximation with BIC penalty.
    """
    n = len(y)
    if n < 3:
        return 1, 0.0

    # One-segment fit (global mean)
    mu = y.mean()
    sse0 = ((y - mu) ** 2).sum()
    bic1 = n * np.log(sse0 / n if sse0 > 0 else 1e-12) + 1 * np.log(n)

    best_k = 1
    best_bic = float("inf")
    logger.debug(f"Testing dimension {n}")
    # Try every split k = 1..n-1
    for k in range(1, n):
        mu1 = y[:k].mean()
        mu2 = y[k:].mean()
        sse = ((y[:k] - mu1) ** 2).sum() + ((y[k:] - mu2) ** 2).sum()
        bic2 = n * np.log(sse / n if sse > 0 else 1e-12) + 2 * np.log(n)
        logger.debug(f"K: {k} BIC1: {bic1} BIC2: {bic2}")
        if bic2 < best_bic:
            best_bic = bic2
            best_k = k

    # Return the best split and the evidence strength
    delta_bic = bic1 - best_bic
    logger.info(f"Best k: {best_k}")
    logger.info(f"delta BIC:  {delta_bic}")
    return best_k, float(delta_bic)


'''def broken_stick_thresholds(p: int) -> np.ndarray:
    """Return Broken–Stick thresholds b_k for k=1..p."""
    # b_k = (1/p) * sum_{i=k}^p (1/i)
    H = np.array([np.sum(1.0/np.arange(k, p+1)) for k in range(1, p+1)], dtype=float)
    return H / p

def broken_stick_mask(vr: np.ndarray) -> np.ndarray:
    """Return a boolean mask of which components exceed the Broken–Stick threshold."""
    vr = np.asarray(vr, dtype=float)
    bs = broken_stick_thresholds(len(vr))
    return vr > bs


def broken_stick_keep_count(vr: np.ndarray) -> int:
    """Return the number of components retained by the Broken–Stick rule."""
    return int(np.sum(broken_stick_mask(vr)))'''

def elbow_point(dimensions, functional) -> int:
    """
    Return the 1-based index of the elbow using a geometric method
    """
    x = np.array(dimensions)
    y = np.array(functional)

    n = len(x)
    if n < 3:
        return int(x[-1])  # trivial fallback
    
    # slopes from i to last point
    dx = (x[-1] - x[:-1])
    dy = (y[-1] - y[:-1])
    # avoid /0; if dx==0, set slope to +inf
    slopes = np.where(np.abs(dx) > 0, dy / dx, np.sign(dy) * np.inf)
    p_m = int(np.argmin(slopes))  # steepest descent


    # line through P1 -> P2
    P1 = np.array([x[p_m], y[p_m]], dtype=float)
    P2 = np.array([x[-1],   y[-1]], dtype=float)
    v = P2 - P1
    if np.allclose(v, 0):
        return int(x[p_m])
    
    # unit normal to v
    nrm = np.array([ v[1], -v[0] ], dtype=float)
    nrm /= np.linalg.norm(nrm)

    # distances from points p_m..end to the line (signed projection onto normal)
    Xi = np.column_stack([x[p_m:], y[p_m:]])
    R = P1 - Xi
    dists = np.abs(R @ nrm)
    j = int(np.argmax(dists))
    elbow = int(x[p_m + j])

    return elbow


class CorrelationChangeDetector:
    """
    Decide the number of significant components for a correlation-matrix PCA
    using the Broken–Stick rule.

    """
    def __init__(
        self,
        consensus_snapshots: int = 2,  # require same assessed dimension across last S snapshots
    ):
        self._functional_values = []
        self._dim_count: int = 0  # how many dimensions processed so far
        self.consensus_snapshots = consensus_snapshots

        # History of calculated values for consensus
        self._history_broken_stick: List[Optional[int]] = []
        self._history_elbow: List[Optional[int]] = []

    '''def _detect_on_snapshot(self, variance_ratios: np.ndarray) -> Optional[int]:
        """
        Run BIC detection + tail stability on the current variance ratio list.
        Returns 1-based index of first point after the drop, or None.
        """
        n = len(variance_ratios)
        if n < 4:
            return None

        keeps = broken_stick_mask(variance_ratios.astype(float))

        ## For the correlation matrix use case, we need to add one to the number of dimensions,
        ## as the last dimension is always expected to have low variance (for good data) because it
        ## is the component of the main signal, i.e. the direction from the origin to the cloud of
        ## points at a distance of ~1.
        n_keep = broken_stick_keep_count(keeps) + 1 

        return n_keep'''
    

    def _coverage_k(self, vr: np.ndarray) -> Optional[int]:
        vr = np.asarray(vr, dtype=float)
        # Normalize defensively if sum is close to 1
        s = vr.sum()
        if abs(s - 1.0) <= 0.05:
            vr = vr / s
        # Ensure non-increasing
        if not np.all(vr[:-1] >= vr[1:]):
            vr = np.sort(vr)[::-1]
        # Small p guard
        if len(vr) == 0:
            return None
        cum = np.cumsum(vr)
        k_idx = np.searchsorted(cum, 0.95, side='left')+1
        if k_idx >= len(vr):
            return len(vr)
        return int(k_idx + 1)  # return 1-based count


    def update(
        self, functional_current: float, variance_ratios_current: np.ndarray
    ) -> Optional[int]:
        """
        Update with the current dimension's 'a' (scalar) and the refreshed 'b' list (numpy array).
        Returns the 1-based dimension index when consensus is achieved, else None.
        """
        # Track dimension count and the first two 'a' values
        self._functional_values.append(functional_current)
        self._dim_count += 1

        # Run snapshot detection on current variance ratio list
        '''nkeeps = self._detect_on_snapshot(
            np.asarray(variance_ratios_current, dtype=float)
        )'''
        k_cov = self._coverage_k(np.asarray(variance_ratios_current, dtype=float))
        self._history_broken_stick.append(k_cov)
        if self._dim_count > 3:
            elbow = elbow_point(list(range(1,self._dim_count+1)), self._functional_values)
            logger.info(f"Current elbow point : {elbow}")
            self._history_elbow.append(int(elbow))

        # Keep last few entries for minimal memory
        keep = max(3, self.consensus_snapshots + 1)
        if len(self._history_broken_stick) > keep:
            self._history_broken_stick = self._history_broken_stick[-keep:]

        # Consensus check
        if self.consensus_snapshots <= 1:
            return k_cov

        tail = self._history_broken_stick[-self.consensus_snapshots :]
        tail_elbow = self._history_elbow[-self.consensus_snapshots :]
        logger.info(tail)
        logger.info(tail_elbow)
        if (
            len(tail) == self.consensus_snapshots
            and all(t is not None for t in tail)
            and len(set(tail)) == 1
        ):
            if (
                len(tail_elbow) == self.consensus_snapshots
                and all(t is not None for t in tail_elbow)
                and len(set(tail_elbow)) == 1
            ):
                return max(tail[-1], tail_elbow[-1])

        return None


class ChangeDetector:
    """
    Snapshot-wise drop detector for refreshed variance-ratio lists, using:
      - Bayesian Information Criterion (BIC) single-changepoint model on the current variance-ratio list
      - Tail stability check
      - Initial-step gate using functional values (only when the best change is at n_dims=2)
      - Consensus across recent snapshots to stabilize final decision.

    Call the update method after analysis at each dimension - returns the dimension number (i.e. 1-based index)
    (first point after the drop) when consensus is achieved, otherwise returns None. Minimum possible
    returned dimension is 2 (i.e., a drop immediately after the first element). Only the first two
    functional values are used for the initial-step gate.
    """

    def __init__(
        self,
        delta_bic_min: float = 10.0,  # strength of evidence required
        consensus_snapshots: int = 2,  # require same assessed dimension across last S snapshots
    ):
        self.delta_bic_min = delta_bic_min
        self.consensus_snapshots = consensus_snapshots

        self._functional_values = []
        self._dim_count: int = 0  # how many dimensions processed so far

        # History of calculated values for consensus
        self._history_bic: List[Optional[int]] = []
        self._history_elbow: List[Optional[int]] = []

    def _detect_on_snapshot(self, variance_ratios: np.ndarray) -> Optional[int]:
        """
        Run BIC detection + tail stability on the current variance ratio list.
        Returns 1-based index of first point after the drop, or None.
        """
        n = len(variance_ratios)
        if n < 4:
            return None, 0

        k0, dBIC = piecewise_constant_bic(
            variance_ratios.astype(float)
        )

        return max(2, k0), dBIC

    def update(
        self, functional_current: float, variance_ratios_current: np.ndarray
    ) -> Optional[int]:
        """
        Update with the current dimension's 'a' (scalar) and the refreshed 'b' list (numpy array).
        Returns the 1-based dimension index when consensus is achieved, else None.
        """
        # Track dimension count and the first two 'a' values
        self._functional_values.append(functional_current)
        self._dim_count += 1

        # Run snapshot detection on current variance ratio list
        k_snapshot, dBic = self._detect_on_snapshot(
            np.asarray(variance_ratios_current, dtype=float)
        )
        self._history_bic.append(k_snapshot)
        if self._dim_count > 3:
            elbow = elbow_point(list(range(1,self._dim_count+1)), self._functional_values)
            logger.info(f"Current elbow point : {elbow}")
            self._history_elbow.append(int(elbow))

        # Keep last few entries for minimal memory
        keep = max(3, self.consensus_snapshots + 1)
        if len(self._history_bic) > keep:
            self._history_bic = self._history_bic[-keep:]

        # Consensus check
        if self.consensus_snapshots <= 1:
            return k_snapshot

        tail = self._history_bic[-self.consensus_snapshots :]
        tail_elbow = self._history_elbow[-self.consensus_snapshots :]
        logger.info(tail)
        logger.info(tail_elbow)
        if (
            len(tail) == self.consensus_snapshots
            and all(t is not None for t in tail)
            and len(set(tail)) == 1
            and dBic > self.delta_bic_min
        ):
            if (
                len(tail_elbow) == self.consensus_snapshots
                and all(t is not None for t in tail_elbow)
                and len(set(tail_elbow)) == 1
            ):
                if (abs(tail_elbow[-1] - tail[-1]) <= 1):
                    # Having one too few dimensions is much worse than one too
                    # many, so go with the higher.
                    return max(tail[-1], tail_elbow[-1])

        return None



class CosymAnalysis(symmetry_base, Subject):
    """Perform cosym analysis.

    Perform cosym analysis on the input intensities using the methods of
    `Gildea, R. J. & Winter, G. (2018). Acta Cryst. D74, 405-410
    <https://doi.org/10.1107/S2059798318002978>`_ for
    determination of Patterson group symmetry from sparse multi-crystal data sets in
    the presence of an indexing ambiguity.
    """

    def __init__(
        self,
        intensities,
        params,
        seed_dataset: int | None = None,
        apply_sigma_correction=True,
    ):
        """Initialise a CosymAnalysis object.

        Args:
          intensities (cctbx.miller.array): The intensities on which to perform
            cosym analysis.
          params (libtbx.phil.scope_extract): Parameters for the analysis.
          seed_dataset (int): The index into the intensities list to use when
            choosing a seed dataset for the reindexing analysis (the x,y,z reindex
            mode will be used for this dataset).
            If None, a high density cluster point is chosen.
        """
        self.seed_dataset = seed_dataset
        if self.seed_dataset:
            self.seed_dataset = int(self.seed_dataset)
            assert 0 <= seed_dataset < len(intensities), (
                "cosym_analysis: seed_dataset parameter must be an integer that can be used to index the intensities list"
            )

        max_id = len(intensities) - 1
        super().__init__(
            intensities,
            normalisation=params.normalisation,
            lattice_symmetry_max_delta=params.lattice_symmetry_max_delta,
            d_min=params.d_min,
            min_i_mean_over_sigma_mean=params.min_i_mean_over_sigma_mean,
            min_cc_half=params.min_cc_half,
            relative_length_tolerance=None,
            absolute_angle_tolerance=None,
            best_monoclinic_beta=params.best_monoclinic_beta,
            apply_sigma_correction=apply_sigma_correction,
        )
        Subject.__init__(
            self, events=["optimised", "analysed_symmetry", "analysed_clusters"]
        )

        # remove those with less than min_reflections after setup.
        if params.min_reflections:
            to_remove = []
            min_id = 0
            histy = flex.histogram(
                self.dataset_ids.as_double(),
                min_id - 0.5,
                max_id + 0.5,
                n_slots=max_id + 1 - min_id,
            )
            vals = histy.slots()
            for i, _ in enumerate(range(min_id, max_id + 1)):
                n = vals[i]
                if n < params.min_reflections:
                    to_remove.append(i)
            if to_remove:
                logger.info(
                    f"Removing datasets {', '.join(str(i) for i in to_remove)} with < {params.min_reflections} reflections"
                )
                sel = flex.bool(self.intensities.size(), True)
                for i in to_remove:
                    sel.set_selected(self.dataset_ids == i, False)
                self.intensities = self.intensities.select(sel)
                self.dataset_ids = self.dataset_ids.select(sel)

        self.params = params

        def _map_space_group_to_input_cell(intensities, space_group):
            from cctbx.sgtbx.bravais_types import bravais_lattice

            best_subgroup = find_matching_symmetry(
                intensities.unit_cell(),
                space_group,
                best_monoclinic_beta=str(bravais_lattice(group=space_group)) == "mI",
            )
            cb_op_inp_best = best_subgroup["cb_op_inp_best"]
            best_subsym = best_subgroup["best_subsym"]
            cb_op_best_primitive = best_subsym.change_of_basis_op_to_primitive_setting()

            sg_cb_op_inp_primitive = (
                space_group.info().change_of_basis_op_to_primitive_setting()
            )
            sg_primitive = space_group.change_basis(sg_cb_op_inp_primitive)
            sg_best = sg_primitive.change_basis(cb_op_best_primitive.inverse())
            # best_subgroup above is the bravais type, so create thin copy here with the
            # user-input space group instead
            best_subsym = best_subsym.customized_copy(space_group_info=sg_best.info())
            best_subgroup = {
                "subsym": best_subsym.change_basis(cb_op_inp_best.inverse()),
                "best_subsym": best_subsym,
                "cb_op_inp_best": cb_op_inp_best,
            }

            intensities = intensities.customized_copy(
                space_group_info=sg_best.change_basis(cb_op_inp_best.inverse()).info()
            )
            return intensities, best_subgroup

        if self.params.space_group is not None:
            self.intensities, self.best_subgroup = _map_space_group_to_input_cell(
                self.intensities, self.params.space_group.group()
            )
            self.best_subgroup["cb_op_inp_best"] = (
                self.best_subgroup["cb_op_inp_best"] * self.cb_op_inp_min
            )
            self.input_space_group = self.intensities.space_group()

            # ensure still unique after mapping - merge equivalents in the higher symmetry
            new_intensities = None
            new_dataset_ids = flex.int([])
            for d in set(self.dataset_ids):
                sel = self.dataset_ids == d
                these_i = self.intensities.select(sel)
                these_merged = these_i.merge_equivalents().array()
                if not new_intensities:
                    new_intensities = these_merged
                else:
                    new_intensities = new_intensities.concatenate(these_merged)
                new_dataset_ids.extend(flex.int(these_merged.size(), d))
            self.intensities = new_intensities
            self.dataset_ids = new_dataset_ids

        else:
            self.input_space_group = None

        if self.params.lattice_group is not None:
            tmp_intensities, _ = _map_space_group_to_input_cell(
                self.intensities, self.params.lattice_group.group()
            )
            self.params.lattice_group = tmp_intensities.space_group_info()
        # N.B. currently only multiprocessing used if cc_weights=sigma
        if self.params.nproc is Auto:
            params.nproc = dials.util.system.CPU_COUNT
            logger.info(f"Setting nproc={params.nproc}")

    def _intialise_target(self):
        if self.params.dimensions is Auto:
            dimensions = None
        else:
            dimensions = self.params.dimensions
        if self.params.lattice_group is not None:
            self.lattice_group = (
                self.params.lattice_group.group()
                .build_derived_patterson_group()
                .info()
                .primitive_setting()
                .group()
            )
        self.target = target.Target(
            self.intensities,
            self.dataset_ids.as_numpy_array(),
            min_pairs=self.params.min_pairs,
            lattice_group=self.lattice_group,
            dimensions=dimensions,
            weights=self.params.weights,
            cc_weights=self.params.cc_weights,
            nproc=self.params.nproc,
        )

    def _determine_dimensions(self, dims_to_test, outlier_rejection=False, pca_variance_model="BIC"):
        logger.info("=" * 80)
        logger.info("\nAutomatic determination of number of dimensions for analysis")
        dimensions = []
        functional = []

        if pca_variance_model == "BIC":
            det = ChangeDetector(
                delta_bic_min=10.0,
                consensus_snapshots=2,
            )
        else:
            det = CorrelationChangeDetector(consensus_snapshots=2)

        for dim in range(1, dims_to_test + 1):
            logger.info(f"Testing dimension: {dim}/{dims_to_test}")
            self.target.set_dimensions(dim)
            max_calls = self.params.minimization.max_calls
            self._optimise(
                self.params.minimization.engine,
                max_iterations=self.params.minimization.max_iterations,
                max_calls=min(20, max_calls) if max_calls else max_calls,
            )
            self._principal_component_analysis()
            dimensions.append(dim)
            functional.append(
                self.target.compute_functional_score_for_dimension_assessment(
                    self.minimizer.x, outlier_rejection
                )
            )
            logger.info(f"Functional: {functional[-1]}")
            decision = det.update(
                functional[-1], np.array(self.explained_variance_ratio)
            )

            if decision is not None:
                logger.info(f"Step change detected at dimension {decision}")
                self.target.set_dimensions(decision)
                logger.info("Using %i dimensions for analysis", self.target.dim)
                return dimensions, functional

        # Find the elbow point of the curve, in the same manner as that used by
        # distl spotfinder for resolution method 1 (Zhang et al 2006).
        # See also dials/algorithms/spot_finding/per_image_analysis.py
        x_g = elbow_point(dimensions, functional)

        logger.info(
            dials.util.tabulate(
                zip(dimensions, functional), headers=("Dimensions", "Functional")
            )
        )

        logger.info("Best number of dimensions: %i", x_g)
        if int(x_g) < 2:
            logger.info(
                "As a minimum of 2-dimensions is required, dimensions have been set to 2."
            )
            self.target.set_dimensions(2)
        else:
            self.target.set_dimensions(int(x_g))
        logger.info("Using %i dimensions for analysis", self.target.dim)

        return dimensions, functional

    def run(self):
        self._intialise_target()
        if self.params.dimensions is Auto and self.target.dim != 2:
            self._determine_dimensions(self.target.dim)
        self._optimise(
            self.params.minimization.engine,
            max_iterations=self.params.minimization.max_iterations,
            max_calls=self.params.minimization.max_calls,
        )
        self._principal_component_analysis()

        self._analyse_symmetry()

    @Subject.notify_event(event="optimised")
    def _optimise(self, engine, max_iterations=None, max_calls=None):
        np.random.seed(self.params.seed)
        NN = len(set(self.dataset_ids))
        n_sym_ops = len(self.target.sym_ops)

        coords = np.random.rand(NN * n_sym_ops * self.target.dim)
        if engine == "scitbx":
            self.minimizer = cosym_engine.minimize_scitbx_lbfgs(
                self.target,
                coords,
                use_curvatures=self.params.use_curvatures,
                max_iterations=max_iterations,
                max_calls=max_calls,
            )
        else:
            self.minimizer = cosym_engine.minimize_scipy(
                self.target,
                coords,
                method="L-BFGS-B",
                max_iterations=max_iterations,
                max_calls=max_calls,
            )

        self.coords = self.minimizer.x.reshape(
            self.target.dim, NN * n_sym_ops
        ).transpose()

    def _principal_component_analysis(self, cluster=False):
        # Perform PCA
        from sklearn.decomposition import PCA

        if cluster:
            logger.info(f"Dimensions used for clustering PCA: {self.target.dim}")
            pca = PCA(n_components=self.target.dim).fit(self.coords)
        else:
            pca = PCA().fit(self.coords)
        logger.info("Principal component analysis:")
        logger.info(
            "Explained variance: "
            + ", ".join(["%.2g" % v for v in pca.explained_variance_])
        )
        logger.info(
            "Explained variance ratio: "
            + ", ".join(["%.4g" % v for v in pca.explained_variance_ratio_])
        )
        self.explained_variance = pca.explained_variance_
        self.explained_variance_ratio = pca.explained_variance_ratio_
        if self.target.dim > 3 and not cluster:
            pca.n_components = 3
        self.coords_reduced = pca.fit_transform(self.coords)
        self.pca_components = pca.components_

    @Subject.notify_event(event="analysed_symmetry")
    def _analyse_symmetry(self):
        sym_ops = [sgtbx.rt_mx(s).new_denominators(1, 12) for s in self.target.sym_ops]

        if not self.input_space_group:
            self._symmetry_analysis = SymmetryAnalysis(
                self.coords, sym_ops, self.subgroups, self.cb_op_inp_min
            )
            logger.info(str(self._symmetry_analysis))
            self.best_solution = self._symmetry_analysis.best_solution
            self.best_subgroup = self.best_solution.subgroup
        else:
            self.best_solution = None
            self._symmetry_analysis = None

        cosets = sgtbx.cosets.left_decomposition(
            self.target._lattice_group,
            self.best_subgroup["subsym"].space_group().build_derived_acentric_group(),
        )
        self.reindexing_ops = self._reindexing_ops(self.coords, sym_ops, cosets)

    def _reindexing_ops(
        self,
        coords: np.ndarray,
        sym_ops: list[sgtbx.rt_mx],
        cosets: sgtbx.cosets.left_decomposition,
    ) -> list[sgtbx.change_of_basis_op]:
        """Identify the reindexing operator for each dataset.

        Args:
          coords (np.ndarray):
            A flattened list of the N-dimensional vectors, i.e. coordinates in
            the first dimension are stored first, followed by the coordinates in
            the second dimension, etc.
          sym_ops (List[sgtbx.rt_mx]): List of cctbx.sgtbx.rt_mx used for the cosym
            symmetry analysis
          cosets (sgtbx.cosets.left_decomposition): The left coset decomposition of the
            lattice group with respect to the proposed Patterson group

        Returns:
          List[sgtbx.change_of_basis_op]: A list of reindexing operators corresponding
            to each dataset.
        """
        reindexing_ops = []
        n_datasets = len(set(self.dataset_ids))
        n_sym_ops = len(sym_ops)
        coord_ids = np.arange(n_datasets * n_sym_ops)
        dataset_ids = coord_ids % n_datasets

        if self.seed_dataset is not None:
            # if seed dataset was specified, use the reindexing op xyz as seed
            sel = np.where(dataset_ids == self.seed_dataset)
            xis = np.array([coords[sel][0]])
        else:
            # choose a high density point as seed
            X = coords
            nbrs = NearestNeighbors(
                n_neighbors=min(11, len(X)), algorithm="brute", metric="cosine"
            ).fit(X)
            distances, indices = nbrs.kneighbors(X)
            average_distance = np.array([dist[1:].mean() for dist in distances])
            i = average_distance.argmin()
            xis = np.array([X[i]])
        coordstr = ",".join(str(round(i, 4)) for i in xis[0])
        logger.debug(f"Coordinate of cluster seed dataset: {coordstr}")

        for j in range(n_datasets):
            sel = np.where(dataset_ids == j)
            X = coords[sel]
            # Find nearest neighbour in cosine-space to the current cluster centroid
            nbrs = NearestNeighbors(
                n_neighbors=min(1, len(X)), algorithm="brute", metric="cosine"
            ).fit(X)
            distances, indices = nbrs.kneighbors([xis.mean(axis=0)])
            k = indices[0][0]
            xis = np.append(xis, [X[k]], axis=0)
            for partition in cosets.partitions:
                if sym_ops[k] in partition:
                    cb_op = sgtbx.change_of_basis_op(partition[0]).new_denominators(
                        self.cb_op_inp_min
                    )
                    reindexing_ops.append(
                        (
                            self.cb_op_inp_min.inverse() * cb_op * self.cb_op_inp_min
                        ).as_xyz()
                    )
                    break

        return reindexing_ops

    def as_dict(self):
        """Return a dictionary representation of the results.

        Returns:
          dict
        """
        d = {
            "input_symmetry": {
                "hall_symbol": self.input_intensities[0]
                .space_group()
                .type()
                .hall_symbol(),
                "unit_cell": self.median_unit_cell.parameters(),
            },
            "cb_op_inp_min": self.cb_op_inp_min.as_xyz(),
            "min_cell_symmetry": {
                "hall_symbol": self.intensities.space_group().type().hall_symbol(),
                "unit_cell": self.intensities.unit_cell().parameters(),
            },
            "lattice_point_group": self.lattice_group.type().hall_symbol(),
        }

        if self._symmetry_analysis is not None:
            d.update(self._symmetry_analysis.as_dict())
        return d

    def as_json(self, filename=None, indent=2):
        """Return a json representation of the results.

        Args:
          filename (str): Optional filename to export the json representation of
            the results.
          indent (int): The indent level for pretty-printing of the json. If ``None``
            is the most compact representation.

        Returns:
          str:
        """
        d = self.as_dict()

        json_str = json.dumps(d, indent=indent)
        if filename:
            with open(filename, "w") as f:
                f.write(json_str)
        return json.dumps(d, indent=indent)


class SymmetryAnalysis:
    def __init__(self, coords, sym_ops, subgroups, cb_op_inp_min):
        import scipy.spatial.distance as ssd

        self.subgroups = subgroups
        self.cb_op_inp_min = cb_op_inp_min
        n_datasets = coords.shape[0] // len(sym_ops)
        dist_mat = ssd.pdist(coords, metric="cosine")
        cos_angle = 1 - ssd.squareform(dist_mat)

        self._sym_ops_cos_angle = {}
        for dataset_id in range(n_datasets):
            for ref_sym_op_id in range(len(sym_ops)):
                ref_idx = n_datasets * ref_sym_op_id + dataset_id
                for sym_op_id in range(ref_sym_op_id + 1, len(sym_ops)):
                    op = sym_ops[ref_sym_op_id].inverse().multiply(sym_ops[sym_op_id])
                    op = op.new_denominators(1, 12)
                    comp_idx = n_datasets * sym_op_id + dataset_id
                    self._sym_ops_cos_angle.setdefault(op, [])
                    self._sym_ops_cos_angle[op].append(cos_angle[ref_idx, comp_idx])

        self._score_symmetry_elements()
        self._score_laue_groups()

    def _score_symmetry_elements(self):
        self.sym_op_scores = {}
        for op, cos_angle in self._sym_ops_cos_angle.items():
            cc_true = 1
            cc = np.mean(cos_angle)
            score = ScoreSymmetryElement(cc, sigma_cc=0.1, cc_true=cc_true)
            score.sym_op = op
            self.sym_op_scores[op] = score

    def _score_laue_groups(self):
        subgroup_scores = [
            ScoreSubGroup(subgrp, list(self.sym_op_scores.values()))
            for subgrp in self.subgroups.result_groups
        ]
        total_likelihood = sum(score.likelihood for score in subgroup_scores)
        for score in subgroup_scores:
            score.likelihood /= total_likelihood
        self.subgroup_scores = sorted(
            subgroup_scores, key=lambda score: score.likelihood, reverse=True
        )

        # The 'confidence' scores are derived from the total probability of the best
        # solution p_best and that for the next best solution p_next:
        #   confidence = [p_best * (p_best - p_next)]^1/2.

        for i, score in enumerate(self.subgroup_scores[:-1]):
            next_score = self.subgroup_scores[i + 1]
            if score.likelihood > 0 and next_score.likelihood > 0:
                lgc = score.likelihood * (score.likelihood - next_score.likelihood)
                confidence = abs(lgc) ** 0.5
                if lgc < 0:
                    confidence = -confidence
                score.confidence = confidence

        self.best_solution = self.subgroup_scores[0]

    @staticmethod
    def sym_ops_table(d):
        header = ("likelihood", "Z-CC", "CC", "", "Operator")
        rows = [header]
        for score in d["sym_op_scores"]:
            rows.append(
                (
                    f"{score['likelihood']:.3f}",
                    f"{score['z_cc']:.2f}",
                    f"{score['cc']:.2f}",
                    score["stars"],
                    str(sgtbx.rt_mx(str(score["operator"])).r().info()),
                )
            )
        return rows

    @staticmethod
    def subgroups_table(d):
        header = (
            "Patterson group",
            "",
            "Likelihood",
            "NetZcc",
            "Zcc+",
            "Zcc-",
            "delta",
            "Reindex operator",
        )
        rows = [header]
        for score in d["subgroup_scores"]:
            rows.append(
                (
                    str(
                        sgtbx.space_group(
                            hall_symbol=str(score["patterson_group"])
                        ).info()
                    ),
                    score["stars"],
                    f"{score['likelihood']:.3f}",
                    f"{score['z_cc_net']: .2f}",
                    f"{score['z_cc_for']: .2f}",
                    f"{score['z_cc_against']: .2f}",
                    f"{score['max_angular_difference']:.1f}",
                    str(sgtbx.change_of_basis_op(str(score["cb_op"]))),
                )
            )
        return rows

    @staticmethod
    def summary_table(d):
        best_subgroup = d["subgroup_scores"][0]
        cell = ", ".join(f"{i:.3f}" for i in best_subgroup["unit_cell"])
        return (
            (
                "Best solution",
                str(
                    sgtbx.space_group(
                        hall_symbol=str(best_subgroup["patterson_group"])
                    ).info()
                ),
            ),
            ("Unit cell", cell),
            ("Reindex operator", best_subgroup["cb_op"]),
            ("Laue group probability", f"{best_subgroup['likelihood']:.3f}"),
            ("Laue group confidence", f"{best_subgroup['confidence']:.3f}"),
        )

    def __str__(self):
        """Return a string representation of the results.

        Returns:
          str:
        """
        output = []
        output.append("Scoring individual symmetry elements")
        d = self.as_dict()
        output.append(dials.util.tabulate(self.sym_ops_table(d), headers="firstrow"))

        output.append("Scoring all possible sub-groups")
        output.append(dials.util.tabulate(self.subgroups_table(d), headers="firstrow"))

        output.append(
            "Best solution: %s"
            % self.best_solution.subgroup["best_subsym"].space_group_info()
        )
        cell = ", ".join(
            f"{i:.3f}"
            for i in self.best_solution.subgroup["best_subsym"].unit_cell().parameters()
        )
        output.append(f"Unit cell: {cell}")
        output.append(
            "Reindex operator: %s"
            % (self.best_solution.subgroup["cb_op_inp_best"] * self.cb_op_inp_min)
        )
        output.append(f"Laue group probability: {self.best_solution.likelihood:.3f}")
        output.append(f"Laue group confidence: {self.best_solution.confidence:.3f}")
        return "\n".join(output)

    def as_dict(self):
        """Return a dictionary representation of the results.

        Returns:
          dict
        """
        d = {"cb_op_inp_min": self.cb_op_inp_min.as_xyz()}

        d["sym_op_scores"] = []
        for rt_mx, score in self.sym_op_scores.items():
            dd = score.as_dict()
            dd["operator"] = rt_mx.as_xyz()
            d["sym_op_scores"].append(dd)

        d["subgroup_scores"] = []
        for score in self.subgroup_scores:
            dd = score.as_dict()
            dd["cb_op"] = (
                sgtbx.change_of_basis_op(dd["cb_op"]) * self.cb_op_inp_min
            ).as_xyz()
            d["subgroup_scores"].append(dd)

        return d


class ScoreSymmetryElement:
    """Analyse intensities for presence of a given symmetry operation.

    1) Calculate the probability of observing this CC if the sym op is present,
       p(CC; S), modelled by a Cauchy distribution centred on cc_true and width
       gamma = sigma_cc.

    2) Calculate the probability of observing this CC if the sym op is
       NOT present, p(CC; !S).

    3) Calculate the likelihood of symmetry element being present,
       p(S; CC) = p(CC; S) / (p(CC; S) + p(CC; !S))

    See appendix A1 of `Evans, P. R. (2011). Acta Cryst. D67, 282-292.
    <https://doi.org/10.1107/S090744491003982X>`_
    """

    def __init__(self, cc, sigma_cc, cc_true):
        """Initialise a ScoreSymmetryElement object.

        Args:
          cc (float): the correlation coefficient for this symmetry element
          sigma_cc (float): the estimated error in the correlation coefficient
          cc_true (float): the expected value of CC if the symmetry element is present,
            E(CC; S)
        """

        self.cc = cc
        self.sigma_cc = sigma_cc
        self.z_cc = self.cc / self.sigma_cc
        score_cc = ScoreCorrelationCoefficient(self.cc, self.sigma_cc, cc_true)
        self.p_cc_given_s = score_cc.p_cc_given_s
        self.p_cc_given_not_s = score_cc.p_cc_given_not_s
        self.likelihood = score_cc.p_s_given_cc

    @property
    def stars(self):
        # define stars attribute - used mainly for output
        if self.likelihood > 0.9:
            stars = "***"
        elif self.likelihood > 0.7:
            stars = "**"
        elif self.likelihood > 0.5:
            stars = "*"
        else:
            stars = ""
        return stars

    def as_dict(self):
        """Return a dictionary representation of the symmetry element scoring.

        The dictionary will contain the following keys:
          - likelihood: The likelihood of the symmetry element being present
          - z_cc: The Z-score for the correlation coefficient
          - cc: The correlation coefficient for the symmetry element
          - operator: The xyz representation of the symmetry element

        Returns:
          dict:
        """

        return {
            "likelihood": self.likelihood,
            "z_cc": self.z_cc,
            "cc": self.cc,
            "stars": self.stars,
        }


class ScoreSubGroup:
    """Score the probability of a given subgroup being the true subgroup.

    1) Calculates overall Zcc scores for symmetry elements present/absent from
       the subgroup.

    2) Calculates the overall likelihood for this subgroup.

    See appendix A2 of `Evans, P. R. (2011). Acta Cryst. D67, 282-292.
    <https://doi.org/10.1107/S090744491003982X>`_
    """

    def __init__(self, subgroup, sym_op_scores):
        """Initialise a ScoreSubGroup object.

        Args:
          subgroup (dict): A dictionary describing the subgroup as generated by
            :class:`cctbx.sgtbx.lattice_symmetry.metric_subgroups`.
          sym_op_scores (list): A list of :class:`ScoreSymmetryElement` objects for each
            symmetry element possibly in the lattice symmetry.
        """
        # Combined correlation coefficients for symmetry operations
        # present/absent from subgroup
        self.subgroup = subgroup
        patterson_group = subgroup["subsym"].space_group()

        # Overall Zcc scores for symmetry elements present/absent from subgroup
        self.z_cc_for = 0
        self.z_cc_against = 0
        n_for = 0
        n_against = 0
        PL_for = 0
        PL_against = 0
        power = 2
        for score in sym_op_scores:
            if score.sym_op in patterson_group:
                self.z_cc_for += score.z_cc**power
                n_for += 1
                PL_for += math.log(score.p_cc_given_s)
            else:
                self.z_cc_against += score.z_cc**power
                n_against += 1
                PL_against += math.log(score.p_cc_given_not_s)

        # Overall likelihood for this subgroup
        self.likelihood = math.exp(PL_for + PL_against)

        if n_against > 0:
            self.z_cc_against = (self.z_cc_against / n_against) ** (1 / power)
        if n_for > 0:
            self.z_cc_for = (self.z_cc_for / n_for) ** (1 / power)
        self.z_cc_net = self.z_cc_for - self.z_cc_against
        self.confidence = 0

    def __str__(self):
        """Return a string representation of the subgroup scores.

        Returns:
          str:
        """
        return "{} {:.3f} {:.2f} {:.2f} {:.2f}".format(
            self.subgroup["best_subsym"].space_group_info(),
            self.likelihood,
            self.z_cc_net,
            self.z_cc_for,
            self.z_cc_against,
        )

    @property
    def stars(self):
        if self.likelihood > 0.8:
            stars = "***"
        elif self.likelihood > 0.6:
            stars = "**"
        elif self.likelihood > 0.4:
            stars = "*"
        else:
            stars = ""
        return stars

    def as_dict(self):
        """Return a dictionary representation of the subgroup scoring.

        The dictionary will contain the following keys:
          - patterson_group: The current subgroup
          - likelihood: The likelihood of the subgroup being correct
          - confidence: The confidence of the subgroup being correct
          - z_cc_for: The combined Z-scores for all symmetry elements present in the
            subgroup
          - z_cc_against: The combined Z-scores for all symmetry elements present in
            the lattice group but not in the subgroup
          - z_cc_net: The net Z-score, i.e. z_cc_for - z_cc_against
          - max_angular_difference: The maximum angular difference between the
            symmetrised unit cell and the P1 unit cell.
          - cb_op: The change of basis operation from the input unit cell to the
            'best' unit cell.

        Returns:
          dict:
        """
        return {
            "patterson_group": self.subgroup["best_subsym"]
            .space_group()
            .type()
            .hall_symbol(),
            "unit_cell": self.subgroup["best_subsym"].unit_cell().parameters(),
            "likelihood": self.likelihood,
            "confidence": self.confidence,
            "z_cc_net": self.z_cc_net,
            "z_cc_for": self.z_cc_for,
            "z_cc_against": self.z_cc_against,
            "max_angular_difference": self.subgroup["max_angular_difference"],
            "cb_op": f"{self.subgroup['cb_op_inp_best']}",
            "stars": self.stars,
        }


def extract_reference_intensities(
    params: iotbx.phil.scope_extract, wavelength: float
) -> miller.array:
    # Extract/calculate a set of intensities from a reference.
    if params.d_min not in {Auto, None}:
        reference_intensities = intensities_from_reference_file(
            params.reference,
            d_min=params.d_min,
            wavelength=wavelength,
            k_sol=params.reference_model.k_sol,
            b_sol=params.reference_model.b_sol,
        )
    else:
        reference_intensities = intensities_from_reference_file(
            params.reference,
            wavelength=wavelength,
            k_sol=params.reference_model.k_sol,
            b_sol=params.reference_model.b_sol,
        )
    initial_space_group_info = reference_intensities.space_group_info()
    group = metric_subgroups(
        reference_intensities.crystal_symmetry(),
        params.lattice_symmetry_max_delta,
        enforce_max_delta_for_generated_two_folds=True,
    ).result_groups[0]
    ref_cb_op = (
        group["best_subsym"].change_of_basis_op_to_minimum_cell()
        * group["cb_op_inp_best"]
    )

    reference_intensities = (
        reference_intensities.change_basis(
            ref_cb_op,
        )
        .expand_to_p1()
        .as_non_anomalous_array()
        .merge_equivalents()
        .array()
    )
    if not reference_intensities.sigmas():
        reference_intensities.set_sigmas(reference_intensities.data() ** 0.5)
    return reference_intensities, initial_space_group_info


def change_of_basis_op_to_best_cell(
    experiments,
    max_delta,
    relative_length_tolerance,
    absolute_angle_tolerance,
    best_subgroup,
):
    """
    Compute change of basis op to map experiments from P1 cell to the best cell
    that matches the best subgroup
    """

    median_cell = median_unit_cell(experiments)
    groups = metric_subgroups(
        experiments[0]
        .crystal.get_crystal_symmetry()
        .customized_copy(unit_cell=median_cell),
        max_delta,
        enforce_max_delta_for_generated_two_folds=True,
    )
    match = None
    for g in groups.result_groups:
        if (
            g["best_subsym"]
            .unit_cell()
            .is_similar_to(
                best_subgroup["best_subsym"].unit_cell(),
                relative_length_tolerance=relative_length_tolerance,
                absolute_angle_tolerance=absolute_angle_tolerance,
            )
        ) and (
            sgtbx.lattice_symmetry_group(g["best_subsym"].unit_cell(), max_delta=0)
            == sgtbx.lattice_symmetry_group(
                best_subgroup["best_subsym"].unit_cell(), max_delta=0
            )
        ):
            match = g
            break
    if not match:
        raise RuntimeError(
            "Unable to determine reindexing operator from minumum cells to best cell.\n"
            + "This may be fixed by increasing relative_length_tolerance or absolute_angle_tolerance."
        )
    cb_op = match["cb_op_inp_best"]
    return cb_op
