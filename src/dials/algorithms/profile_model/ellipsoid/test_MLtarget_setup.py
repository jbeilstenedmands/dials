from __future__ import annotations

from dxtbx.serialize import load

from dials.array_family import flex

# from dials_algorithms_profile_model_ellipsoid_refiner_ext import RefinerData
from dials_algorithms_profile_model_ellipsoid_parameterisation_ext import (
    ModelState,
    Simple6MosaicityParameterisation,
    WavelengthSpreadParameterisation,
)

expt = load.experiment_list("../indexed.expt")[0]
refls = flex.reflection_table.from_file("../indexed.refl").split_by_experiment_id()[0]


# rd = RefinerData(expt, refls)

wave = WavelengthSpreadParameterisation(0.01)
print(wave.sigma())
print(wave.get_param())
M = Simple6MosaicityParameterisation(
    flex.double([0.001, 0.001, 0.002, 0.0001, 0.0001, 0.0001])
)
print(hex(id(M)))
print(M.sigma())
print(list(M.get_params()))
model = ModelState(expt.crystal, M, wave, False, False, False, False)
print(list(model.active_parameters()))
print(model.mosaicity_covariance_matrix())
print(model.n_active_parameters())
new_p = flex.double(
    [0.011, 0.012, 0.013, 10.5, 0.0015, 0.0016, 0.0017, 0.0002, 0.0003, 0.0004, 0.009]
)
print(new_p.size())
model.set_active_parameters(new_p)
print(list(model.active_parameters()))
print("done")
