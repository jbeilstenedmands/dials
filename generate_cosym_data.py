from __future__ import annotations

import scitbx.matrix
from cctbx import crystal, sgtbx, uctbx
from dxtbx.imageset import ImageSequence
from dxtbx.model import (
    Beam,
    Crystal,
    DetectorFactory,
    Experiment,
    ExperimentList,
    GoniometerFactory,
    Scan,
)

from dials.array_family import flex


## FIXME, are we expanding out symmetry equivalents yet?
def generate_intensities(crystal_symmetry, anomalous_flag=False, d_min=2):
    from cctbx import miller

    indices = miller.index_generator(
        crystal_symmetry.unit_cell(),
        crystal_symmetry.space_group().type(),
        anomalous_flag,
        d_min,
    ).to_array()
    miller_set = crystal_symmetry.miller_set(indices, anomalous_flag)
    intensities = flex.random_double(indices.size()) * 1000
    miller_array = miller.array(
        miller_set, data=intensities, sigmas=flex.sqrt(intensities)
    ).set_observation_type_xray_intensity()
    return miller_array


unit_cell = uctbx.unit_cell((100, 100, 100, 90, 90, 90))
space_group = sgtbx.space_group_info("P222").group()

cs = crystal.symmetry(unit_cell=unit_cell, space_group_info=space_group.info())

intensities = generate_intensities(cs, d_min=1)

expts = ExperimentList()
B = scitbx.matrix.sqr(unit_cell.fractionalization_matrix()).transpose()
detector = DetectorFactory.simple(
    sensor="PAD",
    distance=100,
    beam_centre=(112.5, 112.5),
    fast_direction="+x",
    slow_direction="-y",
    pixel_size=(0.075, 0.075),
    image_size=(3000, 3000),
)
scan = Scan(image_range=(0, 20), oscillation=(0.0, 1.0))
beam = Beam(wavelength=1.0, direction=(0, 0, 1))
goniometer = GoniometerFactory.single_axis()
iset = ImageSequence(
    "",  ##FIXME how to create a fake imagesequence without actual images
    beam=beam,
    detector=detector,
    goniometer=goniometer,
    scan=scan,
    indices=list(range(0, 10)),
)
expts.append(
    Experiment(
        crystal=Crystal(B, space_group=space_group, reciprocal=True),
        scan=scan,
        beam=beam,
        detector=detector,
        imageset=iset,
    )
)


refls = flex.reflection_table([])
refls["intensity.sum.value"] = intensities.data()
refls["intensity.sum.variance"] = flex.sqrt(intensities.data())
refls["miller_index"] = intensities.indices()
refls["id"] = flex.int(refls.size(), 0)
refls["d"] = intensities.d_spacings().data()

predicted = flex.reflection_table.from_predictions(expts[0], dmin=2.0)
predicted.calculate_entering_flags(expts)

# match by hkle
"""hkl = self["miller_index"].as_vec3_double().parts()
hkl = (part.as_numpy_array().astype(int) for part in hkl)
e = self["entering"].as_numpy_array().astype(int)
n = np.arange(e.size)
p0 = pd.DataFrame(dict(zip("hklen", (*hkl, e, n))), copy=False)

hkl = other["miller_index"].as_vec3_double().parts()
hkl = (part.as_numpy_array().astype(int) for part in hkl)
e = other["entering"].as_numpy_array().astype(int)
n = np.arange(e.size)
p1 = pd.DataFrame(dict(zip("hklen", (*hkl, e, n))), copy=False)

merged = pd.merge(p0, p1, on=["h", "k", "l", "e"], suffixes=[0, 1])

n0 = cctbx.array_family.flex.size_t(merged.n0.values)
n1 = cctbx.array_family.flex.size_t(merged.n1.values)"""


# n0,n1=predicted.match_by_hkle(refls)

predicted.as_file("example.refl")
print(predicted.size())
