# LIBTBX_SET_DISPATCHER_NAME dev.dials.simple_sum_integrate
from __future__ import annotations

import logging

import cctbx.array_family.flex

import dials.util.log
from dials.algorithms.integration.report import IntegrationReport
from dials.algorithms.profile_model.gaussian_rs import Model as GaussianRSProfileModel
from dials.algorithms.profile_model.gaussian_rs.calculator import (
    ComputeEsdBeamDivergence,
    ComputeEsdReflectingRange,
)
from dials.array_family import flex
from dials.command_line.integrate import filter_reference_pixels, process_reference
from dials.model.data import make_image
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files
from dials.util.phil import parse
from dials.util.version import dials_version
from dials_algorithms_integration_integrator_ext import (
    ShoeboxProcessor,
    ShoeboxProcessorV2,
)

logger = logging.getLogger("dials.command_line.simple_integrate")

phil_scope = parse(
    """
sigma_b = None
  .type = float(allow_none=True)
sigma_m = None
  .type = float(allow_none=True)
output {
reflections = 'simple_integrated.refl'
    .type = str
    .help = "The integrated output filename"

phil = 'dials.simple_integrate.phil'
    .type = str
    .help = "The output phil file"

log = 'dials.simple_integrate.log'
    .type = str
    .help = "The log filename"
}
"""
)

"""
Kabsch 2010 refers to
Kabsch W., Integration, scaling, space-group assignment and
post-refinment, Acta Crystallographica Section D, 2010, D66, 133-144

Usage:
$ dev.dials.simple_integrate.py refined.expt refined.refl
"""


def run():
    """
    Input setup
    """

    phil = phil_scope.fetch()

    usage = "usage: dev.dials.simple_integrate.py models.expt reflections.expt"
    parser = ArgumentParser(
        usage=usage,
        phil=phil,
        epilog=__doc__,
        read_experiments=True,
        read_reflections=True,
        check_format=True,
    )

    params, options = parser.parse_args(args=None, show_diff_phil=False)

    dials.util.log.config(verbosity=options.verbose, logfile=params.output.log)
    logger.info(dials_version())

    """
    Load experiment and reflections
    """

    reflections, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )
    reflections = reflections[0]

    reflections["id"] = cctbx.array_family.flex.int(len(reflections), 0)
    reflections["imageset_id"] = cctbx.array_family.flex.int(len(reflections), 0)

    integrated_reflections = run_simple_integrate(params, experiments, reflections)
    integrated_reflections.as_msgpack_file(params.output.reflections)


def run_simple_integrate(params, experiments, reflections):
    experiment = experiments[0]

    # Remove bad reflections (e.g. those not indexed)
    reflections, _ = process_reference(reflections)
    # Mask neighbouring pixels to shoeboxes
    reflections = filter_reference_pixels(reflections, experiments)

    """
    Predict reflections using experiment crystal
    """

    predicted_reflections = flex.reflection_table.from_predictions(
        experiment, padding=1.0
    )
    predicted_reflections["id"] = cctbx.array_family.flex.int(
        len(predicted_reflections), 0
    )
    predicted_reflections["imageset_id"] = cctbx.array_family.flex.int(
        len(predicted_reflections), 0
    )
    # Updates flags to set which reflections to use in generating reference profiles
    matched, reflections, unmatched = predicted_reflections.match_with_reference(
        reflections
    )

    """
    Create profile model and add it to experiment.
    This is used to predict reflection properties.
    """

    # Filter reflections to use to create the model
    min_zeta = 0.05
    # reflections.map_centroids_to_reciprocal_space(experiments)
    '''from dxtbx import flumpy
    from scitbx import matrix

    sbox = reflections["shoebox"]
    ## we want s1 based on observed centroid positions.
    reflections.map_centroids_to_reciprocal_space(experiments)
    s1vecs = reflections["s1"]
    s0 = matrix.col(experiment.beam.get_s0())
    m2 = matrix.col(experiment.goniometer.get_rotation_axis_datum())
    sample_rotation = matrix.sqr(experiment.goniometer.get_fixed_rotation()).inverse()
    setting_rotation = matrix.sqr(
        experiment.goniometer.get_setting_rotation()
    ).inverse()
    detector = experiment.detector
    scan = experiment.scan
    import numpy as np

    # centroids = model
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    zbox_sizes = []
    for s1, box in zip(s1vecs, sbox):
        s1 = matrix.col(s1)
        e1 = s1.cross(s0).normalize()
        e2 = s1.cross(e1).normalize()
        e3 = (s1 + s0).normalize()
        """R = np.array([list(e1), list(e2), list(e3)])"""
        zeta = m2.dot(e1)
        mask = box.mask != 0
        values = flumpy.to_numpy(box.values(mask))

        coords = flumpy.to_numpy(box.coords(mask))
        s1primes = box.beam_vectors(detector, mask)
        kabsch_coords = []
        # mags1 = (s1.dot(s1)) ** 0.5

        # first calculate the centroid in pixel space
        x = 0
        y = 0
        z = 0
        n = 0
        for v, c in zip(values, coords):
            n += v
            x += v * c[0]
            y += v * c[1]
            z += v * c[2]
        x = x / n
        y = y / n
        z = z / n
        # print(coords)
        # print(f"{x}, {y}")
        x1c, y1c, phic = detector[0].get_pixel_lab_coord((x, y))
        # print(s1primes[0])
        # phic = scan.get_angle_from_array_index(z, deg=False)
        s1obs = matrix.col((x1c, y1c, phic))
        mags1 = (s1obs.dot(s1obs)) ** 0.5
        # print(s1, s1obs)
        # assert 0
        # then transform to kabsch space
        bbox_z_width = box.zsize()
        # phi_0p5 = scan.get_angle_from_array_index(z, deg=False)
        # phi1 = scan.get_angle_from_array_index(z + (1.0 / 6.0), deg=False)
        # eps3_lim = zeta * (phi1 - phi_0p5)
        for s1p, c in zip(s1primes, coords):
            # get sprime = s1 to that pixel
            # get phi'
            eps1 = e1.dot(matrix.col(s1p) - s1obs) / mags1
            eps2 = e2.dot(matrix.col(s1p) - s1obs) / mags1
            phip = scan.get_angle_from_array_index(int(c[2]), deg=False)
            phi = scan.get_angle_from_array_index(z, deg=False)
            # if bbox_z_width == 1:
            #    eps3 = eps3_lim
            ## investigate setting this, but probably just want enough for decent statistic e.g. a few hundred.
            ## probably need at least 3.
            if bbox_z_width >= 8:
                eps3 = zeta * (phip - phi)
            else:
                eps3 = 0.0
            kabsch_coords.append([eps1, eps2, eps3])
        zbox_sizes.append(bbox_z_width)

        # transform the centre into kabsch space
        """x1c, y1c = detector[0].get_pixel_lab_coord((x,y))[:2]
        phic = scan.get_angle_from_array_index(z, deg=False)
        s1p = matrix.col((x1c,y1c,phic))
        eps1 = e1.dot(matrix.col(s1p)-s1)/ mags1
        eps2 = e2.dot(matrix.col(s1p)-s1)/ mags1"""
        # print(coords)
        # print(coords.shape)

        # first transform to reciprocal space

        # now loop through and get weighted differences.
        cov = np.zeros(shape=(3, 3))
        for v, c in zip(values, kabsch_coords):
            dx = c[0]  # - x
            dy = c[1]  # - y
            dz = c[2]  # - z
            cov[0, 0] += (v * dx * dx) * 0.5
            cov[1, 1] += (v * dy * dy) * 0.5
            cov[2, 2] += (v * dz * dz) * 0.5
            cov[0, 1] += v * dx * dy
            cov[0, 2] += v * dx * dz
            cov[1, 2] += v * dy * dz
        cov += cov.T
        cov = cov / n
        # now rotate:

        # cov = R * cov
        # if cov[0,0] != 0.0:
        c0.append(cov[0, 0])
        # if cov[1,1] != 0.0:
        c1.append(cov[1, 1])
        c2.append(cov[0, 1])
        if cov[2, 2] != 0.0:
            c3.append(cov[2, 2])
        """else:
            phi = scan.get_angle_from_array_index(z, deg=False)
            phi1 = scan.get_angle_from_array_index(z+0.25, deg=False)
            eps3 = zeta*(phi1-phi)
            c3.append(eps3**2)"""
    # sigma was about 0.5 pixels - convert to angle to check against sigma_d
    # could in theory use different sigma_d_x and sigma_d_y?
    mean_sigd1 = np.mean(np.array(c0)) ** 0.5 * 180 / 3.14
    mean_sigd2 = np.mean(np.array(c1)) ** 0.5 * 180 / 3.14
    mean_sigm = np.mean(np.array(c3)) ** 0.5 * 180 / 3.14
    print(mean_sigd1)
    print(mean_sigd2)
    print(mean_sigm)
    med_sigd1 = np.median(np.array(c0)) ** 0.5 * 180 / 3.14
    med_sigd2 = np.median(np.array(c1)) ** 0.5 * 180 / 3.14
    med_sigm = np.median(np.array(c3)) ** 0.5 * 180 / 3.14
    print(med_sigd1)
    print(med_sigd2)
    print(med_sigm)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.hist(zbox_sizes, bins=20)
    plt.show()
    plt.hist(c0, bins=100)
    plt.show()
    plt.hist(c1, bins=100)
    plt.show()
    plt.hist(c2, bins=100)
    plt.show()
    plt.hist(c3, bins=100)
    plt.show()
    assert 0'''

    used_in_ref = reflections.get_flags(reflections.flags.used_in_refinement)
    model_reflections = reflections.select(used_in_ref)
    zeta = model_reflections.compute_zeta(experiment)
    model_reflections = model_reflections.select(flex.abs(zeta) >= min_zeta)
    # sbox = model_reflections["shoebox"]

    # centroids = model
    # sigma_D in 3.1 of Kabsch 2010
    """sigma_b = ComputeEsdBeamDivergence(
        experiment.detector, model_reflections, centroid_definition="com"
    ).sigma()
    logger.info(f"Sigma_b com: {sigma_b}")"""
    if params.sigma_b:
        sigma_b = params.sigma_b
    else:
        sigma_b = ComputeEsdBeamDivergence(
            experiment.detector, model_reflections, centroid_definition="s1"
        ).sigma()
    logger.info(f"Sigma_b s1: {sigma_b}")

    # sigma_m in 3.1 of Kabsch 2010
    """sigma_m = ComputeEsdReflectingRange(
        experiment.crystal,
        experiment.beam,
        experiment.detector,
        experiment.goniometer,
        experiment.scan,
        model_reflections,
        algorithm="extended",
    ).sigma()"""
    # print(f"Sigma_m xyzcal: {sigma_m}")
    # model_reflections["xyzcal.mm"] = model_reflections["xyzobs.mm.value"]
    if params.sigma_m:
        sigma_m = params.sigma_m
    else:
        sigma_m = ComputeEsdReflectingRange(
            experiment.crystal,
            experiment.beam,
            experiment.detector,
            experiment.goniometer,
            experiment.scan,
            model_reflections,
            algorithm="extended",
        ).sigma()
    print(f"Sigma_m xyobs: {sigma_m}")
    ## try new method
    """from dxtbx import flumpy

    sbox = model_reflections["shoebox"]
    # centroids = model
    for box in sbox:
        mask = box.mask != 0
        values = flumpy.to_numpy(box.values(mask))
        coords = flumpy.to_numpy(box.coords(mask))
        x = 0
        y = 0
        z = 0
        n = 0
        for v, c in zip(values, coords):
            n += v
            x += v * c[0]
            y += v * c[1]
            z += v * c[2]
        x = x / n
        y = y / n
        z = z / n
        print(x, y, z)
        assert 0

    assert 0"""
    # background_algorithm = SimpleBackgroundExt(params=None, experiments=experiments)
    # success = background_algorithm.compute_background(model_reflections)
    # model_reflections.set_flags(
    #    ~success, model_reflections.flags.failed_during_background_modelling
    # )
    """sigma_m = ComputeEsdReflectingRange(
        experiment.crystal,
        experiment.beam,
        experiment.detector,
        experiment.goniometer,
        experiment.scan,
        model_reflections,
        algorithm="extended",
    ).sigma()
    print(sigma_b, sigma_m)"""
    # assert 0
    # The Gaussian model given in 2.3 of Kabsch 2010
    experiment.profile = GaussianRSProfileModel(
        params=params, n_sigma=3, sigma_b=sigma_b, sigma_m=sigma_m
    )

    """
    Compute properties for predicted reflections using profile model,
    accessed via experiment.profile_model. These reflection_table
    methods are largely just wrappers for profile_model.compute_bbox etc.

    Note: I do not think all these properties are needed for integration,
    but are all present in the current dials.integrate output.
    """

    predicted_reflections.compute_bbox(experiments)
    predicted_reflections.compute_d(experiments)
    min_zeta = 0.05
    zeta = predicted_reflections.compute_zeta(experiment)
    predicted_reflections = predicted_reflections.select(flex.abs(zeta) >= min_zeta)
    predicted_reflections.compute_partiality(experiments)

    # Shoeboxes
    predicted_reflections["shoebox"] = flex.shoebox(
        predicted_reflections["panel"],
        predicted_reflections["bbox"],
        allocate=False,
        flatten=False,
    )
    n_sigma = 3
    """from dials.algorithms.profile_model.gaussian_rs import MaskCalculator3D

    mask_foreground = MaskCalculator3D(
        experiment.beam,
        experiment.detector,
        experiment.goniometer,
        experiment.scan,
        n_sigma * sigma_b,
        n_sigma * sigma_m,
    )

    # Mask the foreground
    mask_foreground(
        predicted_reflections["shoebox"], predicted_reflections["s1"],
        predicted_reflections["xyzcal.px"].parts()[2], predicted_reflections["panel"]
    )"""

    # Get actual shoebox values and the reflections for each image
    imageset = experiment.imageset
    frame0, frame1 = imageset.get_array_range()
    use_subrange = True
    use_new_method = True
    if use_subrange:
        frame1 = 1000
        predicted_reflections = predicted_reflections.select(
            predicted_reflections["d"] > 4.0
        )
        predicted_reflections = predicted_reflections[100:110]
    if use_new_method:
        # nb overlap adjacency list for each image separately?
        # ideal efficient algorithm
        # make a transform spec for profile mapping.
        # for each reflection make a coordinatesystem object (kabsch coord system).
        # for each image, for each refl:
        # calculate dxyz array for slice of pixels based on pixel indices.
        # initialise background accumulator if first slice.
        # check for overlapping regions, calculate dxyz to relevant neighbouring reflections.
        # iterate though pixels:
        #   first check if valid pixel, then
        #   if overlapping pixel, check if foreground to overlap, else skip to determine if foreground or background.
        #      if overlapping fg, skip, else determine if background or foreground for our refl.
        #          if background, add to background accumulator: (pixel, i,j,k) - for constant3d, ignores pixels and if int just stores in efficient histogram.
        #          else add to foreground accumulator - sum just sums, profile modeller maps based on coordinate system and transform.
        #
        shoebox_processor = ShoeboxProcessorV2(
            predicted_reflections,
            len(experiment.detector),
            frame0,
            frame1,
            False,
            experiment.scan,
            experiment.beam,
            experiment.goniometer,
            experiment.detector,
            sigma_b * n_sigma,
            sigma_m * n_sigma,
        )

        for i in range(frame1 - frame0):  # len(experiment.imageset)):
            image = experiment.imageset.get_corrected_data(i)
            mask = experiment.imageset.get_mask(i)
            shoebox_processor.next(make_image(image, mask))
            print(i)
        intensity = shoebox_processor.finalise(predicted_reflections)
        predicted_reflections["intensity_sum_value"] = intensity
    else:
        shoebox_processor = ShoeboxProcessor(
            predicted_reflections,
            len(experiment.detector),
            frame0,
            frame1,
            False,
        )

        for i in range(frame1 - frame0):  # len(experiment.imageset)):
            image = experiment.imageset.get_corrected_data(i)
            mask = experiment.imageset.get_mask(i)
            shoebox_processor.next_data_only(make_image(image, mask))
            print(i)
        predicted_reflections.is_overloaded(experiments)
        predicted_reflections.compute_mask(experiments)
        predicted_reflections.contains_invalid_pixels()
        from dials.extensions.simple_background_ext import SimpleBackgroundExt
        from dials.extensions.simple_centroid_ext import SimpleCentroidExt

        # Background calculated explicitly to expose underlying algorithm
        background_algorithm = SimpleBackgroundExt(params=None, experiments=experiments)
        success = background_algorithm.compute_background(predicted_reflections)
        predicted_reflections.set_flags(
            ~success, predicted_reflections.flags.failed_during_background_modelling
        )

        # Centroids calculated explicitly to expose underlying algorithm
        centroid_algorithm = SimpleCentroidExt(params=None, experiments=experiments)
        centroid_algorithm.compute_centroid(predicted_reflections)

        predicted_reflections.compute_summed_intensity()
    print("done")
    predicted_reflections.as_file("test.refl")

    '''
    """
    # Filter reflections with a high fraction of masked foreground
    valid_foreground_threshold = 0.75  # DIALS default
    sboxs = predicted_reflections["shoebox"]
    nvalfg = sboxs.count_mask_values(MaskCode.Valid | MaskCode.Foreground)
    nforeg = sboxs.count_mask_values(MaskCode.Foreground)
    fraction_valid = nvalfg.as_double() / nforeg.as_double()
    selection = fraction_valid < valid_foreground_threshold
    predicted_reflections.set_flags(
        selection, predicted_reflections.flags.dont_integrate
    )

    predicted_reflections["num_pixels.valid"] = sboxs.count_mask_values(MaskCode.Valid)
    predicted_reflections["num_pixels.background"] = sboxs.count_mask_values(
        MaskCode.Valid | MaskCode.Background
    )
    predicted_reflections["num_pixels.background_used"] = sboxs.count_mask_values(
        MaskCode.Valid | MaskCode.Background | MaskCode.BackgroundUsed
    )
    predicted_reflections["num_pixels.foreground"] = nvalfg"""

    """
    Load modeller that will calculate reference profiles and
    do the actual profile fitting integration.
    """

    # Default params when running dials.integrate with C2sum_1_*.cbf.gz
    fit_method = 1  # reciprocal space fitter (called explicitly below)
    grid_method = 2  # regular grid
    grid_size = 5  # Downsampling grid size described in 3.3 of Kabsch 2010
    # Get the number of scan points
    scan_step = 5
    scan_range = experiment.scan.get_oscillation_range(deg=True)
    scan_range = abs(scan_range[1] - scan_range[0])
    num_scan_points = int(ceil(scan_range / scan_step))
    n_sigma = 4.5  # multiplier to expand bounding boxes
    fitting_threshold = 0.02
    reference_profile_modeller = GaussianRSProfileModeller(
        experiment.beam,
        experiment.detector,
        experiment.goniometer,
        experiment.scan,
        sigma_b,
        sigma_m,
        n_sigma,
        grid_size,
        num_scan_points,
        fitting_threshold,
        grid_method,
        fit_method,
    )

    """
    Calculate grid of reference profiles from predicted reflections
    that matched observed.
    ("Learning phase" of 3.3 in Kabsch 2010)
    """

    sel = predicted_reflections.get_flags(predicted_reflections.flags.reference_spot)
    reference_reflections = predicted_reflections.select(sel)
    sel = reference_reflections.get_flags(reference_reflections.flags.dont_integrate)
    sel = ~sel
    reference_reflections = reference_reflections.select(sel)
    reference_profile_modeller.model(reference_reflections)
    reference_profile_modeller.normalize_profiles()

    profile_model_report = ProfileModelReport(
        experiments, [reference_profile_modeller], model_reflections
    )
    logger.info("")
    logger.info(profile_model_report.as_str(prefix=" "))

    """
    Carry out the integration by fitting to reference profiles in 1D.
    (Calculates intensity using 3.4 of Kabsch 2010)
    """

    sel = predicted_reflections.get_flags(predicted_reflections.flags.dont_integrate)
    sel = ~sel
    predicted_reflections = predicted_reflections.select(sel)
    reference_profile_modeller.fit_reciprocal_space(predicted_reflections)'''
    predicted_reflections.compute_corrections(experiments)

    integration_report = IntegrationReport(experiments, predicted_reflections)
    logger.info("")
    logger.info(integration_report.as_str(prefix=" "))

    """
    Filter for integrated reflections and remove shoeboxes
    """

    del predicted_reflections["shoebox"]
    sel = predicted_reflections.get_flags(
        predicted_reflections.flags.integrated, all=False
    )
    predicted_reflections = predicted_reflections.select(sel)
    return predicted_reflections


if __name__ == "__main__":
    run()
