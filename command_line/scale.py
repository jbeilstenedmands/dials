# coding: utf-8
"""
This program performs scaling on integrated datasets, which attempts to improve
the internal consistency of the reflection intensities by correcting for
various experimental effects. By default, a physically motivated scaling model
is used, with a scale, decay (B-factor) and absorption correction.
If the input files contain multiple datasets, all data will be scaled against
a common target of unique reflection intensities.

The program outputs one scaled.refl file, which contains updated reflection
intensities, variances and per-refelction scale factors, and one scaled.expt
containing the scaling models. These values can then be used to merge the data
with dials.merge for downstream structural solution. Alternatively, the
scaled.expt and scaled.refl files can be passed back to dials.scale, and
further scaling will be performed, starting from where the previous job finished.

A scaling.html file is also generated, containing interactive plots of merging
statistics and scaling model plots.

Example use cases

Regular single-sequence scaling, with no absorption correction::

  dials.scale integrated.refl integrated.expt physical.absorption_correction=False

Scaling multiple datasets, specifying a resolution limit::

  dials.scale 1_integrated.refl 1_integrated.expt 2_integrated.refl 2_integrated.expt d_min=1.4

Incremental scaling (with different options per dataset)::

  dials.scale integrated.refl integrated.expt physical.scale_interval=10.0

  dials.scale integrated_2.refl integrated_2.expt scaled.refl scaled.expt physical.scale_interval=15.0
"""
from __future__ import absolute_import, division, print_function
import logging
import sys
from libtbx import phil
from six.moves import cStringIO as StringIO
from dials.util import log, show_mail_on_error, Sorry
from dials.util.options import OptionParser, reflections_and_experiments_from_files
from dials.util.version import dials_version
from dials.algorithms.scaling.algorithm import ScalingAlgorithm, ScaleAndFilterAlgorithm
from dials.algorithms.scaling.observers import (
    register_default_scaling_observers,
    register_merging_stats_observers,
    register_scale_and_filter_observers,
)

try:
    from typing import List
except ImportError:
    pass

logger = logging.getLogger("dials")
phil_scope = phil.parse(
    """
  include scope dials.algorithms.scaling.model.model.model_phil_scope
  output {
    log = dials.scale.log
      .type = str
      .help = "The log filename"
    experiments = "scaled.expt"
      .type = str
      .help = "Option to set filepath for output json."
    reflections = "scaled.refl"
      .type = str
      .help = "Option to set filepath for output pickle file of scaled
               intensities."
    html = "scaling.html"
      .type = str
      .help = "Filename for html report."
    json = None
      .type = str
      .help = "Filename to save html report data in json format."
    unmerged_mtz = None
      .type = str
      .help = "Filename to export an unmerged_mtz file using dials.export."
    merged_mtz = None
      .type = str
      .help = "Filename to export a merged_mtz file."
    crystal_name = XTAL
      .type = str
      .help = "The crystal name to be exported in the mtz file metadata"
      .expert_level = 1
    use_internal_variance = False
      .type = bool
      .help = "Option to use internal spread of the intensities when merging
              reflection groups and calculating sigI, rather than using the
              sigmas of the individual reflections."
      .expert_level = 1
    merging.nbins = 20
      .type = int
      .help = "Number of bins to use for calculating and plotting merging stats."
      .expert_level = 1
    delete_integration_shoeboxes = True
      .type = bool
      .help = "Discard integration shoebox data from scaling output, to help"
              "with memory management."
      .expert_level = 2
  }
  include scope dials.algorithms.scaling.scaling_options.phil_scope
  include scope dials.algorithms.scaling.cross_validation.cross_validate.phil_scope
  include scope dials.algorithms.scaling.scaling_refiner.scaling_refinery_phil_scope
  include scope dials.algorithms.scaling.scale_and_filter.phil_scope
  include scope dials.util.exclude_images.phil_scope
  include scope dials.util.multi_dataset_handling.phil_scope
""",
    process_includes=True,
)


def _export_merged_mtz(params, experiments, joint_table):
    """Export merged data."""
    # call dials.merge
    from dials.command_line.merge import merge_data_to_mtz
    from dials.command_line.merge import phil_scope as merge_phil_scope

    merge_params = merge_phil_scope.extract()
    merge_params.reporting.wilson_stats = False
    merge_params.reporting.merging_stats = False
    merge_params.assess_space_group = False
    merge_params.partiality_threshold = params.cut_data.partiality_cutoff
    mtz_file = merge_data_to_mtz(merge_params, experiments, [joint_table])
    logger.info("\nWriting merged data to %s", (params.output.merged_mtz))
    out = StringIO()
    mtz_file.show_summary(out=out)
    logger.info(out.getvalue())
    mtz_file.write(params.output.merged_mtz)


def _export_unmerged_mtz(params, experiments, reflection_table):
    """Export data to unmerged_mtz format."""
    from dials.command_line.export import export_mtz
    from dials.command_line.export import phil_scope as export_phil_scope

    export_params = export_phil_scope.extract()
    export_params.intensity = ["scale"]
    export_params.mtz.partiality_threshold = params.cut_data.partiality_cutoff
    export_params.mtz.crystal_name = params.output.crystal_name
    if params.cut_data.d_min:
        export_params.mtz.d_min = params.cut_data.d_min
    logger.info(
        "\nSaving output to an unmerged mtz file to %s.", params.output.unmerged_mtz
    )
    export_params.mtz.hklout = params.output.unmerged_mtz
    export_mtz(export_params, experiments, [reflection_table])


def run_scaling(params, experiments, reflections):
    """Run scaling algorithms; cross validation, scaling + filtering or standard.

    Returns:
        experiments: an experiment list with scaled data (if created)
        joint_table: a single reflection table containing scaled data (if created).
    """

    if params.output.delete_integration_shoeboxes:
        for r in reflections:
            del r["shoebox"]

    if params.cross_validation.cross_validation_mode:
        from dials.algorithms.scaling.cross_validation.cross_validate import (
            cross_validate,
        )
        from dials.algorithms.scaling.cross_validation.crossvalidator import (
            DialsScaleCrossValidator,
        )

        cross_validator = DialsScaleCrossValidator(experiments, reflections)
        cross_validate(params, cross_validator)

        logger.info(
            "Cross validation analysis does not produce scaling output files, rather\n"
            "it gives insight into the dataset. Choose an appropriate parameterisation\n"
            "and rerun scaling without cross_validation_mode.\n"
        )
        return (None, None)

    else:
        # Register the observers at the highest level
        if params.filtering.method:
            algorithm = ScaleAndFilterAlgorithm(params, experiments, reflections)
            register_scale_and_filter_observers(algorithm)
        else:
            algorithm = ScalingAlgorithm(params, experiments, reflections)

        if params.output.html:
            register_default_scaling_observers(algorithm)
        else:
            register_merging_stats_observers(algorithm)

        algorithm.run()

        experiments, joint_table = algorithm.finish()

        return experiments, joint_table


def run(args=None, phil=phil_scope):  # type: (List[str], phil.scope) -> None
    """Run the scaling from the command-line."""
    usage = """Usage: dials.scale integrated.refl integrated.expt
[integrated.refl(2) integrated.expt(2) ....] [options]"""

    parser = OptionParser(
        usage=usage,
        read_experiments=True,
        read_reflections=True,
        phil=phil,
        check_format=False,
        epilog=__doc__,
    )
    params, options = parser.parse_args(args=args, show_diff_phil=False)

    if not params.input.experiments or not params.input.reflections:
        parser.print_help()
        sys.exit()

    reflections, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )

    log.config(verbosity=options.verbose, logfile=params.output.log)
    logger.info(dials_version())

    diff_phil = parser.diff_phil.as_str()
    if diff_phil:
        logger.info("The following parameters have been modified:\n%s", diff_phil)

    scaled_experiments, joint_table = (None, None)

    try:
        scaled_experiments, joint_table = run_scaling(params, experiments, reflections)
    except ValueError as e:
        raise Sorry(e)

    # Note, cross validation mode does not produce scaled datafiles
    if scaled_experiments and joint_table:
        logger.info("Saving the experiments to %s", params.output.experiments)
        scaled_experiments.as_file(params.output.experiments)
        logger.info("Saving the scaled reflections to %s", params.output.reflections)
        joint_table.as_file(params.output.reflections)

        if params.output.unmerged_mtz:
            _export_unmerged_mtz(params, scaled_experiments, joint_table)

        if params.output.merged_mtz:
            _export_merged_mtz(params, scaled_experiments, joint_table)

    logger.info(
        "See dials.github.io/dials_scale_user_guide.html for more info on scaling options"
    )


if __name__ == "__main__":
    with show_mail_on_error():
        run()
