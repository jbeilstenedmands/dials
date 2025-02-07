from __future__ import annotations

import math

import numpy as np

from dxtbx import flumpy
from dxtbx.serialize import load

from dials.array_family import flex

f = "strong.refl"
e = "imported.expt"
refl = flex.reflection_table.from_file(f)
expt = load.experiment_list(e)


## Calculate the variances needed for sigma_b calculation.
# centroid definiton - don't have s1 that we can calc from crystal, so use 'com' (centre of mass) method

s1_centroid = []
variance = np.array([], dtype=np.float64)
shoebox = refl["shoebox"]
detector = expt.detectors()[0]
for i, (p, xyz) in enumerate(zip(refl["panel"], refl["xyzobs.px.value"])):
    s1_centroid.append(detector[p].get_pixel_lab_coord(xyz[0:2]))

for r in range(len(refl)):
    # as noted in dials, background subtraction appears to be missing here.
    mask = shoebox[r].mask != 0
    values = flumpy.to_numpy(shoebox[r].values(mask))
    s1 = shoebox[r].beam_vectors(
        detector, mask
    )  # a call to get_pixel_lab_coord for each pixel
    angles = flumpy.to_numpy(s1.angle(s1_centroid[r], deg=False))
    if np.sum(values) > 1:
        var = np.sum(values * np.square(angles)) / (np.sum(values) - 1)
        variance = np.append(variance, var)
    else:
        variance = np.append(
            variance, 0.0
        )  # we need to store something, then filter later when doing calculation.

refl["sigma_b_variance"] = flumpy.from_numpy(variance)


## Calculate the variances needed for sigma_m calculation.
# ComputeEsdReflectingRange.ExtendedEstimator
from scitbx import simplex

# Get the oscillation width
dphi2 = scan.get_oscillation(deg=False)[1] / 2.0

# Calculate a list of angles and zeta's
# tau is the angular difference between the rotation angle phi
# at its bragg maximum and the centre of the oscillation angles covered by the image
# (zeta is m2 dot e1 in the kabsch 2010 paper).


def calculate_tau_and_zeta(scan, reflections):
    from dials.algorithms.shoebox import MaskCode

    mask_code = MaskCode.Valid | MaskCode.Foreground

    # Calculate the list of frames and z coords
    sbox = reflections["shoebox"]
    phi = reflections["xyzcal.mm"].parts()[2]

    # Calculate the zeta list
    zeta = reflections["zeta"]

    # Calculate the list of tau values
    tau = []
    zeta2 = []
    num = []
    indices = [0]
    for s, p, z in zip(sbox, phi, zeta):
        b = s.bbox
        for z0, f in enumerate(range(b[4], b[5])):
            phi0 = scan.get_angle_from_array_index(int(f), deg=False)
            phi1 = scan.get_angle_from_array_index(int(f) + 1, deg=False)
            d = s.data[z0 : z0 + 1, :, :]
            m = s.mask[z0 : z0 + 1, :, :]
            d = flex.sum(d.as_1d().select(m.as_1d() == mask_code))
            if d > 0:
                tau.append((phi1 + phi0) / 2.0 - p)
                zeta2.append(z)
                num.append(d)
        if len(zeta2) > indices[-1]:
            indices.append(len(zeta2))

    # Return the list of tau and zeta
    return (
        flex.double(tau),
        flex.double(zeta2),
        flex.double(num),
        flex.size_t(indices),
    )


tau, zeta, n, indices = calculate_tau_and_zeta(scan, reflections)

# Calculate zeta * (tau +- dphi / 2) / math.sqrt(2)
e1 = (tau + dphi2) * flex.abs(zeta) / math.sqrt(2.0)  # used in the target
e2 = (tau - dphi2) * flex.abs(zeta) / math.sqrt(2.0)  # used in the target

if len(e1) == 0:
    raise RuntimeError(
        "Something went wrong. Zero pixels selected for estimation of profile parameters."
    )

# Compute intensity
K = flex.double()
nj = []
for i0, i1 in zip(indices[:-1], indices[1:]):
    nj = n[i0:i1]
    K.append(flex.sum(nj))
    nj.append(nj)

# Set the starting values to try 1, 3 degrees seems sensible for
# crystal mosaic spread
start = math.log(0.1 * math.pi / 180)
stop = math.log(1 * math.pi / 180)
starting_simplex = [flex.double([start]), flex.double([stop])]

# Initialise the optimizer
optimizer = simplex.simplex_opt(
    1, matrix=starting_simplex, evaluator=self, tolerance=1e-3
)

# Get the solution
sigma = math.exp(optimizer.get_solution()[0])

# Save the result
self.sigma = sigma

refl.as_file("strong_mod.refl")

variance = variance[variance > 0]
sigma = math.sqrt(np.sum(variance) / variance.size)
print(sigma)
