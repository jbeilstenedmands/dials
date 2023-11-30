from __future__ import annotations

import numpy as np

from dxtbx import flumpy
from scitbx import matrix

from dials.algorithms.profile_model.ellipsoid.refiner import ConditionalDistribution
from dials.array_family import flex


def test_equivalence():

    # copy of some test data.
    norm_s0 = 0.726686
    mu = (1.4654e-05, 4.0118e-05, 7.2654e-01)
    dmu = flex.vec3_double(
        [
            (-6.04538430e-06, -4.79266282e-05, 2.58915729e-04),
            (4.82067272e-05, -5.92986582e-06, 3.20202693e-05),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ]
    )
    S = matrix.sqr(
        (
            1.57194461e-07,
            -3.03137672e-10,
            1.64014291e-09,
            -3.03137672e-10,
            1.60196971e-07,
            2.83399363e-08,
            1.64014291e-09,
            2.83399363e-08,
            1.23415987e-08,
        )
    )
    Snp = np.array(
        [
            1.57194461e-07,
            -3.03137672e-10,
            1.64014291e-09,
            -3.03137672e-10,
            1.60196971e-07,
            2.83399363e-08,
            1.64014291e-09,
            2.83399363e-08,
            1.23415987e-08,
        ]
    ).reshape(3, 3)

    dS = flex.mat3_double(
        [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (
                7.92955130e-04,
                -8.07998127e-07,
                4.12441448e-06,
                -8.07998127e-07,
                8.25616327e-11,
                -2.00825882e-10,
                4.12441448e-06,
                -2.00825882e-10,
                -1.66681314e-10,
            ),
            (
                -8.15627488e-09,
                7.21646318e-05,
                -3.89854794e-04,
                7.21646318e-05,
                -2.86566012e-07,
                1.52621558e-06,
                -3.89854794e-04,
                1.52621558e-06,
                -8.12676428e-06,
            ),
        ]
    )

    c = ConditionalDistribution(
        norm_s0,
        np.array(mu).reshape(3, 1),
        flumpy.to_numpy(dmu).reshape(4, 3).transpose(),
        Snp,
        flumpy.to_numpy(dS)
        .reshape(4, 3, 3)
        .transpose(1, 2, 0),  # expects 3x3x4, need to do it this way
    )

    derivs_list = np.array([])
    for d in c.first_derivatives_of_mean():
        derivs_list = np.append(derivs_list, d.flatten())
    sigmas_list = np.array([])
    for d in c.first_derivatives_of_sigma():
        sigmas_list = np.append(sigmas_list, d.flatten())

    from dials_algorithms_profile_model_ellipsoid_refiner_ext import test_conditional

    print(c.mean())
    print(c.sigma())
    print("dmu")
    print(derivs_list)
    print("dsigma")
    print(sigmas_list)

    test_conditional(norm_s0, mu, dmu, S, dS)


if __name__ == "__main__":
    test_equivalence()
