from __future__ import annotations

import copy
import logging

from dxtbx.model import ExperimentList
from dxtbx.serialize import load
from libtbx import phil

import dials.util.log
from dials.algorithms.scaling.model.model import KBScalingModel
from dials.algorithms.scaling.scaling_library import (
    create_datastructures_for_reference_file,
    scale_against_target,
)
from dials.array_family import flex
from dials.util.options import ArgumentParser

logger = logging.getLogger(__name__)
dials.util.log.config(1, "out.log")

filename = "~/dials/build/dials_data/cunir_serial/2BW4.pdb"
refls = flex.reflection_table.from_file("scaled_batch1.refl")
expts = load.experiment_list("scaled_batch1.expt")
refls_copy = copy.deepcopy(refls)

## upscale all intensities
scale = refls["inverse_scale_factor"]
refls["intensity.sum.value"] = refls["intensity.scale.value"] / scale
refls["intensity.sum.variance"] = refls["intensity.scale.variance"] / (scale**2)
del refls["intensity.scale.value"]
del refls["intensity.scale.variance"]
del refls["intensity.prf.value"]
del refls["intensity.prf.variance"]
del refls["inverse_scale_factor"]
del refls["inverse_scale_factor_variance"]
del refls["partiality"]
del refls["lp"]
del refls["qe"]

refls["id"] = flex.int(refls.size(), 0)
for k in dict(refls.experiment_identifiers()).keys():
    del refls.experiment_identifiers()[k]
refls.experiment_identifiers()[0] = expts[0].identifier


phil_scope = phil.parse(
    """
    include scope dials.algorithms.scaling.scaling_options.phil_scope
    include scope dials.algorithms.scaling.model.model.model_phil_scope
    include scope dials.algorithms.scaling.scaling_refiner.scaling_refinery_phil_scope
""",
    process_includes=True,
)
parser = ArgumentParser(phil=phil_scope, check_format=False)
params, _ = parser.parse_args(args=[], quick_parse=True)
params.model = "KB"
use_decay_correction = True
if not use_decay_correction:
    params.KB.decay_correction = False
expts = expts[0:1]
expts[0].scaling_model = KBScalingModel.from_data(params, [], [])

ref_expt, reference_refl = create_datastructures_for_reference_file(expts, filename)
reference_refl["intensity.sum.value"] = reference_refl["intensity"]
reference_refl["intensity.sum.variance"] = flex.double(reference_refl.size(), 1)
ref_expts = ExperimentList([ref_expt])
result = scale_against_target(refls, expts, reference_refl, ref_expts)

print(expts[0].scaling_model.to_dict())

expts[0].scaling_model.components["scale"].data = {"id": flex.int(refls_copy.size(), 0)}
expts[0].scaling_model.components["scale"].update_reflection_data()
K = expts[0].scaling_model.components["scale"].calculate_scales()
if use_decay_correction:
    expts[0].scaling_model.components["decay"].data = {"d": refls_copy["d"]}
    expts[0].scaling_model.components["decay"].update_reflection_data()
    B = expts[0].scaling_model.components["decay"].calculate_scales()

if use_decay_correction:
    inverse_scale_factors = K * B
else:
    inverse_scale_factors = K
refls_copy["inverse_scale_factor"] *= inverse_scale_factors
refls_copy.as_file("rescaled.refl")
