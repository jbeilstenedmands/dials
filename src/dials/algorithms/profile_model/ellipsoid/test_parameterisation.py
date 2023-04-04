from __future__ import annotations

import time

from cctbx import sgtbx
from dxtbx.model import Crystal
from dxtbx.serialize import load
from scitbx.array_family import flex

from dials.algorithms.refinement.parameterisation.crystal_parameters import (
    CrystalUnitCellParameterisation,
)
from dials_algorithms_profile_model_ellipsoid_parameterisation_ext import (
    SimpleCellParameterisation,
)

expt = load.experiment_list("../indexed.expt")[0]
expt.crystal = Crystal(
    A=(-0.0076, -0.0054, 0.0043, -0.00197, -0.0041, -0.0090, 0.0066, -0.0076, 0.0022),
    space_group_symbol="P1",
)

simplec = SimpleCellParameterisation(expt.crystal)
p = list(simplec.get_params())
print(p)
print(list(simplec.get_dS_dp()))
print(type(simplec.get_params()))
st = time.time()
for i in range(10):
    simplec.set_params([p_ + 0.001 * i for p_ in p])
    print(list(simplec.get_params()))
    print(list(simplec.get_dS_dp()))
print(f"total time 1 {time.time() - st}")

print("Now with python version")
c = CrystalUnitCellParameterisation(expt.crystal)
print(list(c.get_param_vals()))
print(list(c.get_ds_dp()))
st = time.time()
for i in range(10):
    c.set_param_vals([p_ + 0.001 * i for p_ in p])
    print(list(c.get_param_vals()))
    print(list(c.get_ds_dp()))
print(f"total time 2 {time.time() - st}")
