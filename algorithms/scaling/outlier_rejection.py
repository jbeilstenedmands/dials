"""
Definitions of outlier rejection algorithms.

These algorithms use the Ih_table datastructures to perform calculations
in groups of symmetry equivalent reflections. Two functions are provided,
reject_outliers, to reject outlier and set flags given a reflection table
and experiment object, and determine_outlier_index_arrays, which takes an
Ih_table and returns flex.size_t index arrays of the outlier positions.
"""
from __future__ import absolute_import, division, print_function

import copy
import logging
from collections import defaultdict

from cctbx.array_family import flex

from dials.algorithms.scaling.Ih_table import IhTable
from dials.util import tabulate
from dials_scaling_ext import determine_outlier_indices, limit_outlier_weights

logger = logging.getLogger("dials")


def reject_outliers(reflection_table, experiment, method="standard", zmax=6.0):
    """
    Run an outlier algorithm on symmetry-equivalent intensities.

    This method runs an intensity-based outlier rejection algorithm, comparing
    the deviations from the weighted mean in groups of symmetry equivalent
    reflections. The outliers are determined and the outlier_in_scaling flag
    is set in the reflection table.

    The values intensity and variance must be set in the reflection table;
    these should be corrected but unscaled values, as an inverse_scale_factor
    will be applied during outlier rejection if this is present in the reflection
    table. The reflection table should also be prefiltered (e.g. not-integrated
    reflections should not be present) as no further filtering is done on the
    input table.

    Args:
        reflection_table: A reflection table.
        experiment: A single experiment object.
        method (str): Name (alias) of outlier rejection algorithm to use.
        zmax (float): Normalised deviation threshold for classifying an outlier.

    Returns:
        reflection_table: The input table with the outlier_in_scaling flag set.
    """
    assert "intensity" in reflection_table, "reflection table has no 'intensity' column"
    assert "variance" in reflection_table, "reflection table has no 'variance' column"

    if "inverse_scale_factor" not in reflection_table:
        reflection_table["inverse_scale_factor"] = flex.double(
            reflection_table.size(), 1.0
        )

    Ih_table = IhTable(
        [reflection_table], experiment.crystal.get_space_group(), nblocks=1
    )
    outlier_indices, _ = determine_outlier_index_arrays(
        Ih_table, method=method, zmax=zmax
    )[0]

    # Unset any existing outlier flags before setting the new ones
    reflection_table.unset_flags(
        reflection_table.get_flags(reflection_table.flags.outlier_in_scaling),
        reflection_table.flags.outlier_in_scaling,
    )
    reflection_table.set_flags(
        outlier_indices, reflection_table.flags.outlier_in_scaling
    )

    return reflection_table


def determine_outlier_index_arrays(Ih_table, method="standard", zmax=6.0, target=None):
    """
    Run an outlier algorithm and return the outlier indices.

    Args:
        Ih_table: A dials.algorithms.scaling.Ih_table.IhTable.
        method (str): Name (alias) of outlier rejection algorithm to use. If
            method=target, then the optional argument target must also
            be specified. Implemented methods; standard, simple, target.
        zmax (float): Normalised deviation threshold for classifying an outlier.
        target (Optional[IhTable]): An IhTable to use to obtain target Ih for
            outlier rejectiob, if method=target.

    Returns:
        outlier_index_arrays (list): A list of flex.size_t arrays, with one
            array per dataset that was used to create the Ih_table. Importantly,
            the indices are the indices of the reflections in the initial
            reflection table used to create the Ih_table, not the indices of the
            data in the Ih_table.

    Raises:
        ValueError: if an invalid choice is made for the method.
    """
    outlier_rej = None
    if method == "standard":
        outlier_rej = NormDevOutlierRejection(Ih_table, zmax)
    elif method == "simple":
        outlier_rej = SimpleNormDevOutlierRejection(Ih_table, zmax)
    elif method == "target":
        assert target is not None
        outlier_rej = TargetedOutlierRejection(Ih_table, zmax, target)
    elif method is not None:
        raise ValueError("Invalid choice of outlier rejection method: %s" % method)
    if not outlier_rej:
        return [flex.size_t([]) for _ in range(Ih_table.n_datasets)]
    outlier_rej.run()
    outlier_index_arrays = outlier_rej.final_outlier_arrays
    if Ih_table.n_datasets > 1:
        msg = (
            "Combined outlier rejection has been performed across multiple datasets, \n"
        )
    else:
        msg = "A round of outlier rejection has been performed, \n"
    n_outliers = sum(len(i) for i in outlier_index_arrays)
    msg += "{} outliers have been identified. \n".format(n_outliers)
    logger.info(msg)
    return outlier_index_arrays, outlier_rej.outlier_groups


class OutlierGroups(object):
    def __init__(self, n_datasets):
        self.n_datasets = n_datasets
        self.outlier_groups = defaultdict(int)
        self.outlier_group_multiplicities = {}
        self.suspect_groups = flex.miller_index()
        self.suspect_outlier_arrays = []

    def add_outliers(self, asu_indices, multiplicities):
        for idx, n in zip(asu_indices, multiplicities):
            self.outlier_groups[idx] += 1
            if idx not in self.outlier_group_multiplicities:
                self.outlier_group_multiplicities[idx] = n

    def check_for_suspect_groups(self):
        if not self.suspect_groups:
            suspect_groups = flex.miller_index()
            # best way - look at distribution of groups with more than 2 outliers and
            # check likelihood of separation.
            for k, v in self.outlier_groups.items():
                # suspect group:
                # if 2/4 outliers >= 50%
                # 3/5, 3/6, 3/7, > 40%
                # 4/8, 4/9, 4/10, 4/11 > 33%
                # i.e. >33% for groups >4
                n = self.outlier_group_multiplicities[k]
                if n < 5 and v > 1:
                    suspect_groups.append(k)
                elif n < 8 and v > 3:
                    suspect_groups.append(k)
                elif (v / n) > (1 / 2.0):
                    suspect_groups.append(k)
            self.suspect_groups = suspect_groups
        # also determine indices of suspect groups?

    def determine_suspect_indices(self, Ih_table):
        Ih_table = Ih_table.Ih_table_blocks[0]
        self.check_for_suspect_groups()
        logger.info(self)
        selection = flex.bool(Ih_table.size, False)

        for k in self.suspect_groups:
            sel = k == Ih_table.asu_miller_index
            selection.set_selected(sel.iselection(), True)

        outlier_indices = Ih_table.Ih_table["loc_indices"].select(selection)
        datasets = Ih_table.Ih_table["dataset_id"].select(selection)

        if self.n_datasets == 1:
            self.suspect_outlier_arrays = [outlier_indices]
        final_outlier_arrays = []
        for i in range(self.n_datasets):
            final_outlier_arrays.append(outlier_indices.select(datasets == i))
        self.suspect_outlier_arrays = final_outlier_arrays

    def __str__(self):
        self.check_for_suspect_groups()
        rows = []
        nbad = len(self.suspect_groups)
        for k, v in self.outlier_groups.items():
            # suspect group: if 2/4 outliers 3/5, 3/6, 3/7, 4/8, 4/9, 4/10, 4/11 >33% for groups >4
            n = self.outlier_group_multiplicities[k]
            bad = "o"
            if k in self.suspect_groups:
                bad = "x"
            rows.append([k, v, n, bad])
        s = tabulate(
            rows, ["asu index", "n_outliers", "group multiplicity", "good group"]
        )
        if nbad:
            s += f"\n{nbad} suspect groups with a high fraction of outliers\n"
        return s


class OutlierRejectionBase(object):
    """
    Base class for outlier rejection algorithms using an IhTable datastructure.

    Subclasses must implement the _do_outlier_rejection method, which must
    add the indices of outliers to the _outlier_indices attribute. The algorithms
    are run upon initialisation and result in the population of the
    :obj:`final_outlier_arrays`.

    Attributes:
        final_outlier_arrays (:obj:`list`): A list of flex.size_t arrays of outlier
            indices w.r.t. the order of the initial reflection tables used to
            create the Ih_table.
    """

    def __init__(self, Ih_table, zmax):
        """Set up and run the outlier rejection algorithm."""
        assert (
            Ih_table.n_work_blocks == 1
        ), """
Outlier rejection algorithms require an Ih_table with nblocks = 1"""
        # Note: could be possible to code for nblocks > 1
        self._Ih_table_block = Ih_table.blocked_data_list[0]
        self._n_datasets = Ih_table.n_datasets
        self._block_selections = Ih_table.blocked_selection_list[0]
        self._datasets = flex.int([])
        self._zmax = zmax
        self._outlier_indices = flex.size_t([])
        self.final_outlier_arrays = None
        self.outlier_groups = OutlierGroups(self._n_datasets)

    def run(self):
        """Run the outlier rejection algorithm, implemented by a subclass."""
        self._do_outlier_rejection()
        self.final_outlier_arrays = self._determine_outlier_indices(
            self._outlier_indices, self._datasets
        )

    def _determine_outlier_indices(self, outlier_indices, datasets):
        """
        Determine outlier indices with respect to the input reflection tables.

        Transform the outlier indices w.r.t the Ih_table, determined during the
        algorithm, to outlier indices w.r.t the initial reflection tables used
        to create the Ih_table, separated by reflection table.

        Returns:
            final_outlier_arrays (:obj:`list`): A list of flex.size_t arrays of
                outlier indices w.r.t. the order of the data in the initial
                reflection tables used to create the Ih_table.
        """
        if self._n_datasets == 1:
            return [outlier_indices]
        final_outlier_arrays = []
        for i in range(self._n_datasets):
            final_outlier_arrays.append(outlier_indices.select(datasets == i))
        return final_outlier_arrays

    def _do_outlier_rejection(self):
        """Add indices (w.r.t. the Ih_table data) to self._outlier_indices."""
        raise NotImplementedError()


class TargetedOutlierRejection(OutlierRejectionBase):
    """Implementation of an outlier rejection algorithm against a target.

    This algorithm requires a target Ih_table in addition to an Ih_table
    for the dataset under investigation. Normalised deviations are
    calculated from the intensity values in the target table.
    """

    def __init__(self, Ih_table, zmax, target):
        """Set a target Ih_table and run the outlier rejection."""
        assert (
            target.n_work_blocks == 1
        ), """
Targeted outlier rejection requires a target Ih_table with nblocks = 1"""
        self._target_Ih_table_block = target.blocked_data_list[0]
        self._target_Ih_table_block.calc_Ih()
        super(TargetedOutlierRejection, self).__init__(Ih_table, zmax)

    def _do_outlier_rejection(self):
        """Add indices (w.r.t. the Ih_table data) to self._outlier_indices."""
        Ih_table = self._Ih_table_block
        target = self._target_Ih_table_block
        target_asu_Ih_dict = dict(
            zip(target.asu_miller_index, zip(target.Ih_values, target.variances))
        )
        Ih_table.Ih_table["target_Ih_value"] = flex.double(Ih_table.size, 0.0)
        Ih_table.Ih_table["target_Ih_sigmasq"] = flex.double(Ih_table.size, 0.0)
        for j, miller_idx in enumerate(Ih_table.asu_miller_index):
            if miller_idx in target_asu_Ih_dict:
                Ih_table.Ih_table["target_Ih_value"][j] = target_asu_Ih_dict[
                    miller_idx
                ][0]
                Ih_table.Ih_table["target_Ih_sigmasq"][j] = target_asu_Ih_dict[
                    miller_idx
                ][1]

        nz_sel = Ih_table.Ih_table["target_Ih_value"] != 0.0
        Ih_table = Ih_table.select(nz_sel)
        norm_dev = (
            Ih_table.intensities
            - (Ih_table.inverse_scale_factors * Ih_table.Ih_table["target_Ih_value"])
        ) / (
            flex.sqrt(
                Ih_table.variances
                + (
                    flex.pow2(Ih_table.inverse_scale_factors)
                    * Ih_table.Ih_table["target_Ih_sigmasq"]
                )
            )
        )
        outliers_sel = flex.abs(norm_dev) > self._zmax
        outliers_isel = nz_sel.iselection().select(outliers_sel)

        outliers = flex.bool(self._Ih_table_block.size, False)
        outliers.set_selected(outliers_isel, True)

        self._outlier_indices.extend(
            self._Ih_table_block.Ih_table["loc_indices"].select(outliers)
        )
        self._datasets.extend(
            self._Ih_table_block.Ih_table["dataset_id"].select(outliers)
        )


class SimpleNormDevOutlierRejection(OutlierRejectionBase):
    """Algorithm using normalised deviations from the weighted intensity means.

    In this case, the weighted mean is calculated from all reflections in
    the symmetry group excluding the test reflection.
    """

    def __init__(self, Ih_table, zmax):
        super(SimpleNormDevOutlierRejection, self).__init__(Ih_table, zmax)
        self.weights = limit_outlier_weights(
            copy.deepcopy(self._Ih_table_block.weights),
            self._Ih_table_block.h_index_matrix,
        )

    def _do_outlier_rejection(self):
        """Add indices (w.r.t. the Ih_table data) to self._outlier_indices."""
        Ih_table = self._Ih_table_block
        intensity = Ih_table.intensities
        g = Ih_table.inverse_scale_factors
        w = self.weights
        wgIsum = (
            (w * g * intensity) * Ih_table.h_index_matrix
        ) * Ih_table.h_expand_matrix
        wg2sum = ((w * g * g) * Ih_table.h_index_matrix) * Ih_table.h_expand_matrix

        # guard against zero divison errors - can happen due to rounding errors
        # or bad data giving g values are very small
        zero_sel = wg2sum == 0.0
        # set as one for now, then mark as outlier below. This will only affect if
        # g is near zero, if w is zero then throw an assertionerror.
        wg2sum.set_selected(zero_sel, 1.0)

        assert w.all_gt(0)  # guard against division by zero
        norm_dev = (intensity - (g * wgIsum / wg2sum)) / (
            flex.sqrt((1.0 / w) + flex.pow2(g / wg2sum))
        )
        norm_dev.set_selected(zero_sel, 1000)  # to trigger rejection
        outliers = flex.abs(norm_dev) > self._zmax

        asu_indices = Ih_table.asu_miller_index.select(outliers)
        n_group = Ih_table.calc_nh().select(outliers)
        self.outlier_groups.add_outliers(asu_indices, n_group)

        self._outlier_indices.extend(Ih_table.Ih_table["loc_indices"].select(outliers))
        self._datasets.extend(
            self._Ih_table_block.Ih_table["dataset_id"].select(outliers)
        )


class NormDevOutlierRejection(OutlierRejectionBase):
    """Algorithm using normalised deviations from the weighted intensity means.

    In this case, the weighted mean is calculated from all reflections in
    the symmetry group excluding the test reflection.
    """

    def __init__(self, Ih_table, zmax):
        super(NormDevOutlierRejection, self).__init__(Ih_table, zmax)
        self.weights = limit_outlier_weights(
            copy.deepcopy(self._Ih_table_block.weights),
            self._Ih_table_block.h_index_matrix,
        )

    def _do_outlier_rejection(self):
        """Add indices (w.r.t. the Ih_table data) to self._outlier_indices."""
        self._round_of_outlier_rejection()
        n_outliers = len(self._outlier_indices)
        n_new_outliers = n_outliers
        while n_new_outliers:
            self._round_of_outlier_rejection()
            n_new_outliers = len(self._outlier_indices) - n_outliers
            n_outliers = len(self._outlier_indices)

    def _round_of_outlier_rejection(self):
        """
        Calculate normal deviations from the data in the Ih_table.
        """
        Ih_table = self._Ih_table_block
        intensity = Ih_table.intensities
        g = Ih_table.inverse_scale_factors
        w = self.weights
        wgIsum = (
            (w * g * intensity) * Ih_table.h_index_matrix
        ) * Ih_table.h_expand_matrix
        wg2sum = ((w * g * g) * Ih_table.h_index_matrix) * Ih_table.h_expand_matrix
        wgIsum_others = wgIsum - (w * g * intensity)
        wg2sum_others = wg2sum - (w * g * g)
        # Now do the rejection analyis if n_in_group > 2
        nh = Ih_table.calc_nh()
        sel = nh > 2
        wg2sum_others_sel = wg2sum_others.select(sel)
        wgIsum_others_sel = wgIsum_others.select(sel)

        # guard against zero divison errors - can happen due to rounding errors
        # or bad data giving g values are very small
        zero_sel = wg2sum_others_sel == 0.0
        # set as one for now, then mark as outlier below. This will only affect if
        # g is near zero, if w is zero then throw an assertionerror.
        wg2sum_others_sel.set_selected(zero_sel, 1.0)
        g_sel = g.select(sel)
        I_sel = intensity.select(sel)
        w_sel = w.select(sel)

        assert w_sel.all_gt(0)  # guard against division by zero
        norm_dev = (I_sel - (g_sel * wgIsum_others_sel / wg2sum_others_sel)) / (
            flex.sqrt((1.0 / w_sel) + (flex.pow2(g_sel) / wg2sum_others_sel))
        )
        norm_dev.set_selected(zero_sel, 1000)  # to trigger rejection
        z_score = flex.abs(norm_dev)

        # Want an array same size as Ih table.
        all_z_scores = flex.double(Ih_table.size, 0.0)
        all_z_scores.set_selected(sel.iselection(), z_score)
        outlier_indices, other_potential_outliers = determine_outlier_indices(
            Ih_table.h_index_matrix, all_z_scores, self._zmax
        )

        asu_indices = Ih_table.asu_miller_index.select(outlier_indices)
        n_group = nh.select(outlier_indices)
        self.outlier_groups.add_outliers(asu_indices, n_group)
        self._outlier_indices.extend(
            self._Ih_table_block.Ih_table["loc_indices"].select(outlier_indices)
        )
        self._datasets.extend(
            self._Ih_table_block.Ih_table["dataset_id"].select(outlier_indices)
        )
        sel = flex.bool(Ih_table.size, False)
        sel.set_selected(other_potential_outliers, True)
        self._Ih_table_block = self._Ih_table_block.select(sel)
        self.weights = self.weights.select(sel)
