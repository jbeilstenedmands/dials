from __future__ import annotations

import logging

from dxtbx.serialize import load

import dials.util.log
from dials.algorithms.integration.integrator import create_integrator
from dials.algorithms.profile_model.factory import ProfileModelFactory
from dials.array_family import flex
from dials.command_line.integrate import working_phil

## Just some logging for this script.
logger = logging.getLogger("dials.example")
dials.util.log.config(logfile="example.log")

## The inputs - refl output from dials.predict and expt output from dials.index
## plus estimates of the sigma parameters off the strong reflections.
refls = flex.reflection_table.from_file("predicted.refl")
expts = load.experiment_list("indexed.expt")
sigma_m = 0.01
sigma_b = 0.01

## Set the relevant parameters
params = working_phil.extract()
params.integration.integrator = "inflight"
params.profile.gaussian_rs.parameters.sigma_b = sigma_b
params.profile.gaussian_rs.parameters.sigma_m = sigma_m

## Set some stuff up - we are using the InFlight integrator class,
## which uses ShoeboxProcessorV2 from processor.h to do the actual
## integration.
expts = ProfileModelFactory.create(params, expts)
integrator = create_integrator(params, expts, refls)

# Integrate the reflections
reflections = integrator.integrate()
reflections.as_file("example_integrated.refl")
