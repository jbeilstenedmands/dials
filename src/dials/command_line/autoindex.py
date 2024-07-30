# input is a strong spot list

from __future__ import annotations

# need a detector model - what exactly?
# detector distance and orientation (0,0,1) wl = 1.23985
# goniometer matrix
# imagine all coming from a nexus file
import math
import time

from cctbx import crystal, uctbx, xray

from dials.algorithms.indexing.basis_vector_search.utils import (
    group_vectors,
    is_approximate_integer_multiple,
)
from dials_algorithms_indexing_ext import (  # , map_centroids_to_reciprocal_space_grid_cpp
    do_fft3d,
)
from dials_algorithms_indexing_ext import xyz_to_rlp as xyz_to_rlp_cpp


def _find_peaks(
    grid_real, d_min, rmsd_cutoff=15, n_points=256, peak_volume_cutoff=0.15
):
    grid_real_binary = grid_real.deep_copy()
    rmsd = math.sqrt(
        flex.mean(
            flex.pow2(grid_real_binary.as_1d() - flex.mean(grid_real_binary.as_1d()))
        )
    )
    grid_real_binary.set_selected(grid_real_binary < (rmsd_cutoff) * rmsd, 0)
    grid_real_binary.as_1d().set_selected(grid_real_binary.as_1d() > 0, 1)
    grid_real_binary = grid_real_binary.iround()
    from cctbx import masks

    # real space FFT grid dimensions
    cell_lengths = [n_points * d_min / 2 for i in range(3)]
    fft_cell = uctbx.unit_cell(cell_lengths + [90] * 3)
    flood_fill = masks.flood_fill(grid_real_binary, fft_cell)

    if flood_fill.n_voids() < 4:
        # Require at least peak at origin and one peak for each basis vector
        raise RuntimeError(
            "Indexing failed: fft3d peak search failed to find sufficient number of peaks."
        )

    # the peak at the origin might have a significantly larger volume than the
    # rest so exclude any anomalously large peaks from determining minimum volume
    from scitbx.math import five_number_summary

    outliers = flex.bool(flood_fill.n_voids(), False)
    grid_points_per_void = flood_fill.grid_points_per_void()
    min_x, q1_x, med_x, q3_x, max_x = five_number_summary(grid_points_per_void)
    iqr_multiplier = 5
    iqr_x = q3_x - q1_x
    cut_x = iqr_multiplier * iqr_x
    outliers.set_selected(grid_points_per_void.as_double() > (q3_x + cut_x), True)
    # print q3_x + cut_x, outliers.count(True)
    isel = (
        grid_points_per_void
        > int(peak_volume_cutoff * flex.max(grid_points_per_void.select(~outliers)))
    ).iselection()

    sites = flood_fill.centres_of_mass_frac().select(isel)
    volumes = flood_fill.grid_points_per_void().select(isel)

    return sites, volumes, fft_cell


from scitbx import fftpack, matrix

import dials_algorithms_indexing_ext


def do_cpp_fft3d(rlp, d_min):
    b_iso = -4 * d_min**2 * math.log(0.05)
    used_in_indexing = flex.bool(rlp.size(), True)
    return do_fft3d(rlp, d_min, b_iso)


def fft3d(rlp, d_min):
    n_points = 256
    gridding = fftpack.adjust_gridding_triple(
        (n_points, n_points, n_points), max_prime=5
    )
    from scitbx.array_family import flex

    grid = flex.double(flex.grid(gridding), 0)
    b_iso = -4 * d_min**2 * math.log(0.05)

    used_in_indexing = flex.bool(rlp.size(), True)

    dials_algorithms_indexing_ext.map_centroids_to_reciprocal_space_grid(
        grid,
        rlp,
        used_in_indexing,  # do we really need this?
        d_min,
        b_iso=b_iso,
    )
    reciprocal_space_grid = grid
    print(f"Number of centroids used: {(reciprocal_space_grid > 0).count(True)}")

    fft = fftpack.complex_to_complex_3d(gridding)
    grid_complex = flex.complex_double(
        reals=reciprocal_space_grid,
        imags=flex.double(reciprocal_space_grid.size(), 0),
    )
    grid_transformed = fft.forward(grid_complex)
    grid_real = flex.pow2(flex.real(grid_transformed))
    del grid_transformed
    return grid_real, used_in_indexing


def candidate_basis_vecs_from_grid(grid_real, d_min):
    sites, volumes, fft_cell = _find_peaks(grid_real, d_min)
    candidate_basis_vectors = sites_to_vecs(sites, volumes, fft_cell)
    return candidate_basis_vectors


def sites_to_vecs(sites, volumes, fft_cell, min_cell=3, max_cell=92.3):
    crystal_symmetry = crystal.symmetry(unit_cell=fft_cell, space_group_symbol="P1")
    xs = xray.structure(crystal_symmetry=crystal_symmetry)
    for i, site in enumerate(sites):
        xs.add_scatterer(xray.scatterer("C%i" % i, site=site))

    xs = xs.sites_mod_short()
    sites_cart = xs.sites_cart()
    lengths = flex.double([matrix.col(sc).length() for sc in sites_cart])
    perm = flex.sort_permutation(lengths)
    xs = xs.select(perm)
    volumes = volumes.select(perm)

    vectors = xs.sites_cart()
    norms = vectors.norms()
    sel = (norms > min_cell) & (norms < (2 * max_cell))
    vectors = vectors.select(sel)
    vectors = [matrix.col(v) for v in vectors]
    volumes = volumes.select(sel)
    norms = norms.select(sel)
    vector_groups = group_vectors(vectors, volumes)
    vectors = [g.mean for g in vector_groups]
    volumes = flex.double(max(g.weights) for g in vector_groups)

    # sort by peak size
    perm = flex.sort_permutation(volumes, reverse=True)
    volumes = volumes.select(perm)
    vectors = [vectors[i] for i in perm]

    # sort by length
    lengths = flex.double(v.length() for v in vectors)
    perm = flex.sort_permutation(lengths)

    # exclude vectors that are (approximately) integer multiples of a shorter
    # vector
    unique_vectors = []
    unique_volumes = flex.double()
    for p in perm:
        v = vectors[p]
        is_unique = True
        for i, v_u in enumerate(unique_vectors):
            if (unique_volumes[i] > volumes[p]) and is_approximate_integer_multiple(
                v_u, v
            ):
                print("rejecting %s: integer multiple of %s", v.length(), v_u.length())
                is_unique = False
                break
        if is_unique:
            unique_vectors.append(v)
            unique_volumes.append(volumes[p])

    # re-sort by peak volume
    perm = flex.sort_permutation(unique_volumes, reverse=True)
    candidate_basis_vectors = [unique_vectors[i] for i in perm]
    return candidate_basis_vectors


# xyzobs.px.valuemap from pixel xy to
from dials.array_family import flex

# r = flex.reflection_table.from_file("strong_1_60.refl")
r = flex.reflection_table.from_file("../strong.refl")
xyzobs_px = r["xyzobs.px.value"]

from dxtbx.serialize import load

# expt = load.experiment_list("imported_1_60.expt", check_format=False)[0]
expt = load.experiment_list("../imported.expt", check_format=False)[0]


def xyz_to_rlp(xyzobs_px, expt):
    # from flex_ext, map_centroids_to_reciprocal_space
    from cctbx.array_family import flex

    pixel_size = 0.172
    image_range_start = 1
    osc_start = 0
    osc_width = 0.5
    i_panel = 0
    wavelength = 1.23985
    s0 = (0, 0, -1.0 / wavelength)

    setting_rotation = matrix.sqr(expt.goniometer.get_setting_rotation())
    rotation_axis = expt.goniometer.get_rotation_axis_datum()
    sample_rotation = matrix.sqr(expt.goniometer.get_fixed_rotation())

    # centroid_px_to_mm_panel
    x, y, z = xyzobs_px.parts()
    # fixme this is simple px to mm, but really should be ParallaxCorrectedPxMmStrategy.to_millimeter
    x_mm = x * pixel_size
    y_mm = y * pixel_size
    DEG2RAD = math.pi / 180
    # get_angle_from_array_index
    rot_angle = (
        z + 1 - image_range_start
    ) * osc_width + osc_start  # i.e z_mm (but its actually)
    rot_angle *= DEG2RAD

    # detector things for get_lab_coord
    # s1 = expt.detector[i_panel].get_lab_coord(flex.vec2_double(x_mm, y_mm)) # requires fast, slow, origin etc.
    f = expt.detector[i_panel].get_fast_axis()
    s = expt.detector[i_panel].get_slow_axis()
    n = expt.detector[i_panel].get_normal()
    origin = expt.detector[i_panel].get_origin()
    d_ = matrix.sqr(
        (
            f[0],
            s[0],
            n[0] + origin[0],
            f[1],
            s[1],
            n[1] + origin[1],
            f[2],
            s[2],
            n[2] + origin[2],
        )
    )

    s1 = flex.mat3_double(x_mm.size(), d_) * flex.vec3_double(
        x_mm, y_mm, flex.double(x_mm.size(), 1.0)
    )
    s1 = s1 / s1.norms() * (1 / wavelength)
    S = s1 - s0
    # transform S to rlp in correct frame.
    rlp = tuple(setting_rotation.inverse()) * S
    rlp = rlp.rotate_around_origin(rotation_axis, -rot_angle)
    rlp = tuple(sample_rotation.inverse()) * rlp

    return rlp


def run_main_version():
    st = time.time()
    rlp = xyz_to_rlp(xyzobs_px, expt)
    end = time.time()
    print(f"Time for xyz to rlp: {(end - st):.6f}")

    st = time.time()
    res, used = fft3d(rlp, d_min=1.8)
    end = time.time()
    print(f"Time for fft3d: {(end - st):.6f}")
    return res, used


def run_cpp_version():

    st = time.time()
    i_panel = 0
    f = expt.detector[i_panel].get_fast_axis()
    s = expt.detector[i_panel].get_slow_axis()
    n = expt.detector[i_panel].get_normal()
    origin = expt.detector[i_panel].get_origin()
    d_ = matrix.sqr(
        (
            f[0],
            s[0],
            n[0] + origin[0],
            f[1],
            s[1],
            n[1] + origin[1],
            f[2],
            s[2],
            n[2] + origin[2],
        )
    )
    osc_start, osc_width = expt.scan.get_oscillation()
    image_range_start = expt.scan.get_image_range()[0]
    rlp = xyz_to_rlp_cpp(
        xyzobs_px,
        matrix.sqr(expt.goniometer.get_fixed_rotation()),
        d_,
        expt.beam.get_wavelength(),
        expt.detector[i_panel].get_pixel_size()[0],
        image_range_start,
        osc_start,
        osc_width,
        expt.goniometer.get_rotation_axis(),
        expt.goniometer.get_setting_rotation(),
    )
    end = time.time()
    print(f"Time for xyz to rlp: {(end - st):.6f}")

    st = time.time()
    res = do_cpp_fft3d(rlp, d_min=1.8)
    end = time.time()
    print(f"Time for fft3d: {(end - st):.6f}")
    return res


print("running main version")
res_main, used = run_main_version()

print("running cpp version")
res_cpp = run_cpp_version()


def check_fft_equivalence(res_cpp, res_main):
    print("Checking equivalence of fft results")
    assert res_cpp.size() == res_main.size()
    for i, (r1, r2) in enumerate(zip(res_cpp, res_main)):
        if abs(r2 - r1) > 1e-6:
            print(r1, r2, i)
            assert 0


# check_fft_equivalence(res_cpp, res_main)

use_cpp = False
if use_cpp:
    grid_real = flex.double(flex.grid((256, 256, 256)), 0)
    for i, v in enumerate(res_cpp):
        grid_real[i] = v
else:
    grid_real = res_main

st = time.time()
candidate_basis_vecs_from_grid(grid_real, d_min=1.8)
end = time.time()
print(f"Time for finding candidate basis vecs: {(end - st):.6f}")

# end
def check_rlp_equivalence(rlp, rlp2):
    for r1, r2 in zip(rlp, rlp2):
        for i in range(0, 3):
            assert abs(r1[i] - r2[i]) < 1e-6, f"{r1[i]}, {r2[i]}"


def test_gridding_equivalence():
    """d_min=1.8
    b_iso = -4 * d_min**2 * math.log(0.05)
    g1 = map_centroids_to_reciprocal_space_grid_cpp(
        rlp2, 1.8, b_iso)
    n_points = 256
    gridding = fftpack.adjust_gridding_triple(
        (n_points, n_points, n_points), max_prime=5
    )
    from scitbx.array_family import flex

    grid = flex.double(flex.grid(gridding), 0)
    b_iso = -4 * d_min**2 * math.log(0.05)

    used_in_indexing = flex.bool(rlp.size(), True)

    dials_algorithms_indexing_ext.map_centroids_to_reciprocal_space_grid(
        grid,
        rlp,
        used_in_indexing,  # do we really need this?
        d_min,
        b_iso=b_iso,
    )
    g2 = grid
    for i, (g,gi) in enumerate(zip(g1,g2)):
        if abs(g-gi) > 1e-6:
            print(i, g, gi)
            assert 0
    # end check gridding"""
    pass
