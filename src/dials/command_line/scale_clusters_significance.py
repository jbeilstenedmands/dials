from __future__ import annotations

import logging
import sys

from dxtbx.model import ExperimentList
from libtbx import phil

from dials.algorithms.scaling.scaling_library import (
    scale_against_target,
    scaled_data_as_miller_array,
)
from dials.array_family import flex
from dials.command_line.scale import phil_scope
from dials.util import log, show_mail_handle_errors
from dials.util.options import ArgumentParser
from dials.util.version import dials_version

try:
    from typing import List
except ImportError:
    pass

from scipy.stats.distributions import chi2


def calculate_significance(arr1, arr2):
    # unsure if need, but not always there so doing just in case for now
    arr1.is_xray_intensity_array()
    arr2.is_xray_intensity_array()

    # Do need this
    arr1 = arr1.customized_copy(crystal_symmetry=arr2.crystal_symmetry())

    int1, int2 = arr1.common_sets(arr2)

    difference = int1.data() - int2.data()
    difference_sigmas = (int1.sigmas() ** (2) + int2.sigmas() ** (2)) ** 1 / 2
    q = 0
    dof = len(difference)
    for i, j in zip(difference, difference_sigmas):
        z = i / j
        z2 = z**2
        q += z2
    p_value = chi2.sf(q, dof)
    significance = 0.05
    if p_value < significance:
        significant_cluster = True
    else:
        significant_cluster = False
    return significant_cluster, p_value, q, dof


logger = logging.getLogger("dials")


@show_mail_handle_errors()
def run(args: List[str] = None, phil: phil.scope = phil_scope) -> None:
    """Run the scaling from the command-line."""
    usage = """Usage: dials.scale integrated.refl integrated.expt
[integrated.refl(2) integrated.expt(2) ....] [options]"""

    parser = ArgumentParser(
        usage=usage,
        read_experiments=True,
        read_reflections=True,
        phil=phil,
        check_format=False,
        epilog=__doc__,
    )
    params, options = parser.parse_args(args=args, show_diff_phil=False)
    params.weighting.error_model.error_model = None

    if not params.input.experiments or not params.input.reflections:
        parser.print_help()
        sys.exit()
    log.config(verbosity=options.verbose, logfile=params.output.log)
    logger.info(dials_version())
    assert len(params.input.reflections) == 2
    assert len(params.input.experiments) == 2

    r1 = params.input.reflections[0].data
    r2 = params.input.reflections[1].data

    e1 = params.input.experiments[0].data
    e2 = params.input.experiments[1].data

    # First calculate the significance between the two clusters

    a1 = scaled_data_as_miller_array([r1], e1)
    a2 = scaled_data_as_miller_array([r2], e2)
    a1 = a1.merge_equivalents().array()
    a2 = a2.merge_equivalents().array()
    logger.info("Significance of difference of input datasets")
    res = calculate_significance(a1, a2)
    logger.info(
        f"significant_cluster: {res[0]}\np_value: {res[1]}\n q:{res[2]}\n dof:{res[3]}"
    )

    # Do it this way so that the intensities have the proper error model adjustment
    r1["intensity.sum.value"] = r1["intensity.scale.value"]
    r1["intensity.sum.variance"] = r1["intensity.scale.variance"]

    r1["intensity.sum.value"] /= r1["inverse_scale_factor"]
    r1["intensity.sum.variance"] /= r1["inverse_scale_factor"] ** 2

    r2["intensity.sum.value"] = r2["intensity.scale.value"]
    r2["intensity.sum.variance"] = r2["intensity.scale.variance"]

    r2["intensity.sum.value"] /= r2["inverse_scale_factor"]
    r2["intensity.sum.variance"] /= r2["inverse_scale_factor"] ** 2

    # delete things so that the 'sum' intensity won't be corrected any further
    for k in [
        "lp",
        "qe",
        "dqe",
        "partiality",
        "intensity.prf.value",
        "intensity.prf.variance",
        "intensity.scale.value",
        "intensity.scale.variance",
        "inverse_scale_factor",
    ]:
        if k in r1:
            del r1[k]
        if k in r2:
            del r2[k]

    # reset some ids
    r1["id"] = flex.int(r1.size(), 0)
    r2["id"] = flex.int(r2.size(), 1)
    for k in list(r1.experiment_identifiers().keys()):
        del r1.experiment_identifiers()[k]
    for k in list(r2.experiment_identifiers().keys()):
        del r2.experiment_identifiers()[k]
    r1.experiment_identifiers()[0] = "0"
    r2.experiment_identifiers()[1] = "1"

    # remove the existing scaling models

    for e in e1:
        e.scaling_model = None
    for e in e2:
        e.scaling_model = None

    e1[0].identifier = "0"
    e2[0].identifier = "1"

    elist1 = ExperimentList([e1[0]])
    elist2 = ExperimentList([e2[0]])
    params.model = "KB"
    result = scale_against_target(r1, elist1, r2, elist2, params)

    logger.info("\nFinal scaling model")
    logger.info(
        f"Scale factor: {elist1.scaling_models()[0].to_dict()['scale']['parameters'][0]}"
    )
    logger.info(
        f"B factor: {elist1.scaling_models()[0].to_dict()['decay']['parameters'][0]}"
    )
    # now calculate the significance
    a3 = scaled_data_as_miller_array([result], elist1)
    a3 = a3.merge_equivalents().array()

    logger.info("\nSignificance of difference of input datasets after coarse scaling")
    res = calculate_significance(a3, a2)
    logger.info(
        f"significant_cluster: {res[0]}\np_value: {res[1]}\n q:{res[2]}\n dof:{res[3]}"
    )


if __name__ == "__main__":
    run()
