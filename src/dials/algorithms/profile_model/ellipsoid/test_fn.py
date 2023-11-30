from __future__ import annotations

from scitbx import matrix
from scitbx.array_family import flex

from dials.algorithms.profile_model.ellipsoid import rse

x = matrix.sqr([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
y = (1.0, 1.0)
z = (2.0, 2.0)

res = rse(x, y, z)
print(res)
