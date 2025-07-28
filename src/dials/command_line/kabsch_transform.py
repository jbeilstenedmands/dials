from __future__ import annotations

from dxtbx.serialize import load

from dials.algorithms.profile_model.gaussian_rs import Model as GaussianRSProfileModel
from dials.algorithms.profile_model.gaussian_rs.model import phil_scope
from dials.array_family import flex
from dials_algorithms_integration_integrator_ext import kabsch_transform

expts = load.experiment_list("indexed.expt", check_format=False)
r1 = flex.reflection_table.from_file("indexed.refl")
r1 = r1.select(r1["id"] > -1)
r1 = r1[10000:10002]
del r1["shoebox"]

params = phil_scope.extract()
sigma_b = 0.0002
sigma_m = 0.002
expts[0].profile = GaussianRSProfileModel(
    params=params, n_sigma=3, sigma_b=sigma_b, sigma_m=sigma_m
)
r1.compute_zeta(expts[0])
r1 = r1.select(flex.abs(r1["zeta"]) > 0.05)
r1.compute_bbox(expts)
r1["shoebox"] = flex.shoebox(r1["panel"], r1["bbox"], allocate=False, flatten=False)
r1.as_file("filtered.refl")
coords_0 = kabsch_transform(
    r1[0:1], expts[0].beam, expts[0].goniometer, expts[0].scan, expts[0].detector
)
coords_1 = kabsch_transform(
    r1[1:2], expts[0].beam, expts[0].goniometer, expts[0].scan, expts[0].detector
)
t1 = flex.reflection_table()
id_1 = flex.int(coords_0.size(), 0)
id_2 = flex.int(coords_1.size(), 1)
id_1.extend(id_2)
coords_0.extend(coords_1)

t1.experiment_identifiers()[0] = "refl0"
t1.experiment_identifiers()[1] = "refl1"
t1["id"] = id_1
t1["kabsch_coordinates"] = coords_0
t1.as_file("coordinates.refl")
