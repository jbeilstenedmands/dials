from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
        logger.info(f"K: {k} BIC1: {bic1} BIC2: {bic2}")
        if bic2 < best_bic:
            best_bic = bic2
            best_k = k

    # Return the best split and the evidence strength
    delta_bic = bic1 - best_bic
    logger.info(f"Best k: {best_k}")
    logger.info(f"delta BIC:  {delta_bic}")
    return best_k, float(delta_bic)


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
    dx = x[-1] - x[:-1]
    dy = y[-1] - y[:-1]
    # avoid /0; if dx==0, set slope to +inf
    slopes = np.where(np.abs(dx) > 0, dy / dx, np.sign(dy) * np.inf)
    p_m = int(np.argmin(slopes))  # steepest descent

    # line through P1 -> P2
    P1 = np.array([x[p_m], y[p_m]], dtype=float)
    P2 = np.array([x[-1], y[-1]], dtype=float)
    v = P2 - P1
    if np.allclose(v, 0):
        return int(x[p_m])

    # unit normal to v
    nrm = np.array([v[1], -v[0]], dtype=float)
    nrm /= np.linalg.norm(nrm)

    # distances from points p_m..end to the line (signed projection onto normal)
    Xi = np.column_stack([x[p_m:], y[p_m:]])
    R = P1 - Xi
    dists = np.abs(R @ nrm)
    j = int(np.argmax(dists))
    elbow = int(x[p_m + j])

    return elbow


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
        test_start_dimension: int = 4,
    ):
        self.delta_bic_min = delta_bic_min
        self.consensus_snapshots = consensus_snapshots
        self.test_start_dimension = test_start_dimension

        self._functional_values = []
        self._dim_count: int = 0  # how many dimensions processed so far

        # History of calculated values for consensus
        self._history_bic: List[Optional[int]] = []
        self._history_elbow: List[Optional[int]] = []

    def _detect_on_snapshot(self, variance_ratios: np.ndarray):
        """
        Run BIC detection + tail stability on the current variance ratio list.
        Returns 1-based index of first point after the drop, or 0.
        """
        if len(variance_ratios) < self.test_start_dimension:
            return 0, 0

        return piecewise_constant_bic(variance_ratios.astype(float))

    def update(
        self, functional_current: float, variance_ratios_current: np.ndarray
    ) -> Optional[int]:
        """
        Returns the 1-based dimension index when consensus is achieved, else None.
        """
        self._functional_values.append(functional_current)
        self._dim_count += 1

        # Run snapshot detection on current variance ratio list
        k_snapshot, dBic = self._detect_on_snapshot(
            np.asarray(variance_ratios_current, dtype=float)
        )
        k_snapshot += 1  # convert from index in list to number of dimensions.
        # i.e. k=1 means a step change between index 0 and 1, so we want to run
        # eith 2 dimensions.
        self._history_bic.append(k_snapshot)

        if self._dim_count >= self.test_start_dimension:
            elbow = elbow_point(
                list(range(1, self._dim_count + 1)), self._functional_values
            )
            logger.debug(f"Current elbow point : {elbow}")
            self._history_elbow.append(int(elbow))

        # Consensus check
        tail = self._history_bic[-self.consensus_snapshots :]
        tail_elbow = self._history_elbow[-self.consensus_snapshots :]

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
                if abs(tail_elbow[-1] - tail[-1]) <= 1:
                    # Having one too few dimensions is much worse than one too
                    # many, so go with the higher.
                    return max(tail[-1], tail_elbow[-1])

        return None
