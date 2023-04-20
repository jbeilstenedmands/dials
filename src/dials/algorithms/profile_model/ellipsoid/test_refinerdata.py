from __future__ import annotations

import time

from dxtbx.serialize import load

from dials.algorithms.profile_model.ellipsoid.refiner import (
    RefinerData as RefinerDataPy,
)
from dials.array_family import flex
from dials_algorithms_profile_model_ellipsoid_refiner_ext import (
    RefinerData as RefinerDataCPP,
)

expt = load.experiment_list("../indexed.expt")[0]
refls = flex.reflection_table.from_file("../indexed.refl").split_by_experiment_id()[0]


st = time.time()
for _ in range(10):
    rdp = RefinerDataPy.from_reflections(expt, refls)
    print(rdp)
end = time.time()
print(end - st)
st = time.time()
for _ in range(10):
    rdc = RefinerDataCPP(expt, refls)
    print(rdc)
end = time.time()
print(end - st)

