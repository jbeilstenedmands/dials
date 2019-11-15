"""
Tests for scale components module.
"""
from __future__ import absolute_import, division, print_function
from math import exp, pi
import pytest
from scitbx import sparse
from dials.array_family import flex
from dials.algorithms.scaling.model.components.scale_components import (
    SHScaleComponent,
    SingleBScaleFactor,
    SingleScaleFactor,
    ScaleComponentBase,
    SphericalAbsorptionComponent,
)
from dials.algorithms.scaling.model.components.smooth_scale_components import (
    SmoothScaleComponent1D,
    SmoothBScaleComponent1D,
    SmoothScaleComponent2D,
    SmoothScaleComponent3D,
    SmoothMixin,
)

from dials.algorithms.scaling.parameter_handler import scaling_active_parameter_manager
from dials.algorithms.scaling.target_function import ScalingTarget
from dials.algorithms.scaling.Ih_table import IhTable
from dials.algorithms.scaling.basis_functions import RefinerCalculator
from dials.algorithms.refinement.engine import SimpleLBFGS
from cctbx.sgtbx import space_group


def test_SphericalAbsorptionComponent():
    """Test the implementation of the Spherical Absorption correction."""

    theta = flex.double([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    sinsqtheta = flex.sin(theta * pi / 180.0) ** 2
    # muR = 0.3 for a sphere, values for Astar.  g = 1/A*
    Astar_0p3 = flex.double(
        [1.5574, 1.5571, 1.5561, 1.5546, 1.5525, 1.5497, 1.5463, 1.5426, 1.5383]
    )

    Astar_1p0 = flex.double(
        [4.1237, 4.1131, 4.0815, 4.0304, 3.9625, 3.8816, 3.7917, 3.6966, 3.6001]
    )

    class SingleComponentRefiner(SimpleLBFGS):
        def __init__(
            self, target, prediction_parameterisation, Ih_table, *args, **kwargs
        ):
            SimpleLBFGS.__init__(
                self,
                target=target,
                prediction_parameterisation=prediction_parameterisation,
                *args,
                **kwargs
            )
            self._target = target
            self._parameters = prediction_parameterisation
            self._Ih_table = Ih_table

        def prepare_for_step(self):
            self._parameters.set_param_vals(self.x)
            return

        def update_journal(self):
            pass

        def compute_functional_gradients_and_curvatures(self):
            """overwrite method to avoid calls to 'blocks' methods of target"""
            self.prepare_for_step()

            # update for minimisation
            s, d = RefinerCalculator.calculate_scales_and_derivatives(
                self._parameters, 0
            )
            self._Ih_table.set_derivatives(d, 0)
            self._Ih_table.set_inverse_scale_factors(s, 0)
            self._Ih_table.update_weights(0)
            self._Ih_table.calc_Ih(0)

            f, g = self._target.compute_functional_gradients(
                self._Ih_table.blocked_data_list[0]
            )
            return f, g, None

    # Icor = I / g = Imeas * Astar
    # so if Icor = 1, Imeas = 1 / Astar
    data = {"sinsqtheta": sinsqtheta}
    reflections = flex.reflection_table()

    reflections["variance"] = flex.double(9, 1.0)
    reflections["miller_index"] = flex.miller_index([(0, 0, 1)] * 9)
    reflections["inverse_scale_factor"] = flex.double(9, 1.0)

    for A in [Astar_0p3, Astar_1p0]:
        component = SphericalAbsorptionComponent(
            initial_values=flex.double([0.00, 0.0, 0.0, 0.0])
        )
        intensities = 1.0 / A
        component.data = data
        component.update_reflection_data()

        components = {"abs": component}
        apm = scaling_active_parameter_manager(components, ["abs"])
        target = ScalingTarget()
        reflections["intensity"] = intensities
        sg = space_group("P 1")
        Ih_table = IhTable([reflections], sg)

        refiner = SingleComponentRefiner(target, apm, Ih_table)
        refiner.run()

        scales = component.calculate_scales()
        determined_correction = (1.0 / scales) * A[0]  # scale by 'global scale'
        print("deltas:")
        print(list(determined_correction - A))
        assert list(determined_correction) == pytest.approx(A, rel=0.001)


def test_ScaleComponentBase():
    """Test for the ScaleComponentBase class."""

    # Test initialisation with no parameter esds.
    base_SF = ScaleComponentBase(flex.double([1.0] * 3))
    assert base_SF.n_params == 3
    assert base_SF.parameter_esds is None
    assert list(base_SF.parameters) == [1.0, 1.0, 1.0]

    # Test updating of parameters
    base_SF.parameters = flex.double([2.0, 2.0, 2.0])
    assert list(base_SF.parameters) == [2.0, 2.0, 2.0]
    with pytest.raises(AssertionError):
        # Try to change the number of parameters - should fail
        base_SF.parameters = flex.double([2.0, 2.0, 2.0, 2.0])

    # Test setting of var_cov matrix.
    assert base_SF.var_cov_matrix is None
    base_SF.var_cov_matrix = [1.0, 0.0, 0.0]
    assert base_SF.var_cov_matrix == [1.0, 0.0, 0.0]

    assert base_SF.calculate_restraints() is None
    assert base_SF.calculate_jacobian_restraints() is None

    base_SF = ScaleComponentBase(flex.double(3, 1.0), flex.double(3, 0.1))
    assert list(base_SF.parameter_esds) == [0.1] * 3


def test_SingleScaleFactor():
    """Test for SingleScaleFactor class."""
    KSF = SingleScaleFactor(flex.double([2.0]))
    assert KSF.n_params == 1
    assert list(KSF.parameters) == [2.0]
    rt = flex.reflection_table()
    rt["d"] = flex.double([1.0, 1.0])
    rt["id"] = flex.int([0, 0])
    KSF.data = {"id": rt["id"]}
    KSF.update_reflection_data()
    assert KSF.n_refl == [2]
    s, d = KSF.calculate_scales_and_derivatives()
    assert list(s) == [2.0, 2.0]
    assert d[0, 0] == 1
    assert d[1, 0] == 1
    s, d = KSF.calculate_scales_and_derivatives()
    KSF.update_reflection_data(flex.bool([True, False]))  # Test selection.
    assert KSF.n_refl[0] == 1


def test_SingleBScaleFactor():
    """Test forSingleBScaleFactor class."""
    BSF = SingleBScaleFactor(flex.double([0.0]))
    assert BSF.n_params == 1
    assert list(BSF.parameters) == [0.0]
    rt = flex.reflection_table()
    rt["d"] = flex.double([1.0, 1.0])
    rt["id"] = flex.int([0, 0])
    BSF.data = {"d": rt["d"], "id": rt["id"]}
    BSF.update_reflection_data()
    assert BSF.n_refl == [2]
    assert list(BSF.d_values[0]) == [1.0, 1.0]
    s, d = BSF.calculate_scales_and_derivatives()
    assert list(s) == [1.0, 1.0]
    assert d[0, 0] == 0.5
    assert d[1, 0] == 0.5
    s, d = BSF.calculate_scales_and_derivatives()
    BSF.update_reflection_data(flex.bool([True, False]))  # Test selection.
    assert BSF.n_refl[0] == 1


def test_SHScalefactor():
    """Test the spherical harmonic absorption component."""
    initial_param = 0.1
    initial_val = 0.2

    SF = SHScaleComponent(flex.double([initial_param] * 3))
    assert SF.n_params == 3
    assert list(SF.parameters) == [initial_param] * 3

    # Test functionality just by setting sph_harm_table directly and calling
    # update_reflection_data to initialise the harmonic values.
    harmonic_values = sparse.matrix(3, 1)
    harmonic_values[0, 0] = initial_val
    harmonic_values[1, 0] = initial_val
    harmonic_values[2, 0] = initial_val
    SF.data = {"sph_harm_table": harmonic_values}
    SF.update_reflection_data()
    print(SF.harmonic_values)
    assert SF.harmonic_values[0][0, 0] == initial_val
    assert SF.harmonic_values[0][0, 1] == initial_val
    assert SF.harmonic_values[0][0, 2] == initial_val
    s, d = SF.calculate_scales_and_derivatives()
    assert list(s) == [1.0 + (3.0 * initial_val * initial_param)]
    assert d[0, 0] == initial_val
    assert d[0, 1] == initial_val
    assert d[0, 2] == initial_val
    s, d = SF.calculate_scales_and_derivatives()

    # Test functionality of passing in a selection
    harmonic_values = sparse.matrix(3, 2)
    harmonic_values[0, 0] = initial_val
    harmonic_values[0, 1] = initial_val
    harmonic_values[2, 0] = initial_val
    SF.data = {"sph_harm_table": harmonic_values}
    SF.update_reflection_data(flex.bool([False, True]))
    assert SF.harmonic_values[0].n_rows == 1
    assert SF.harmonic_values[0].n_cols == 3
    assert SF.n_refl[0] == 1

    # Test setting of restraints and that restraints are calculated.
    # Not testing actual calculation as may want to change the form.
    SF.parameter_restraints = flex.double([0.1, 0.2, 0.3])
    assert SF.parameter_restraints == flex.double([0.1, 0.2, 0.3])
    restraints = SF.calculate_restraints()
    assert restraints[0] is not None
    assert restraints[1] is not None
    jacobian_restraints = SF.calculate_jacobian_restraints()
    assert jacobian_restraints[0] is not None
    assert jacobian_restraints[1] is not None


def test_SmoothMixin():
    """Simple test for the Smooth Mixin class."""
    Smooth_mixin_class = SmoothMixin()
    assert hasattr(Smooth_mixin_class, "smoother")
    assert Smooth_mixin_class.nparam_to_val(2) == 1
    assert Smooth_mixin_class.nparam_to_val(3) == 2
    assert Smooth_mixin_class.nparam_to_val(5) == 3
    assert Smooth_mixin_class.nparam_to_val(6) == 4


def test_SmoothScaleFactor1D():
    """Test for the gaussian smoothed 1D scalefactor class."""
    SF = SmoothScaleComponent1D(flex.double(5, 1.1))
    assert SF.n_params == 5
    assert list(SF.parameters) == [1.1, 1.1, 1.1, 1.1, 1.1]
    norm_rot = flex.double([0.5, 1.0, 2.5, 0.0])
    SF.data = {"x": norm_rot}
    SF.update_reflection_data()
    assert list(SF.normalised_values[0]) == [0.5, 1.0, 2.5, 0.0]
    SF.smoother.set_smoothing(4, 1.0)
    assert list(SF.smoother.positions()) == [-0.5, 0.5, 1.5, 2.5, 3.5]
    s, d = SF.calculate_scales_and_derivatives()
    s2 = SF.calculate_scales()
    assert list(s) == list(s2)
    assert list(s) == pytest.approx([1.1, 1.1, 1.1, 1.1])
    assert d[0, 0] / d[0, 1] == pytest.approx(exp(-1.0) / exp(0.0))
    sumexp = exp(-1.0 / 1.0) + exp(-0.0 / 1.0) + exp(-1.0 / 1.0)  # only averages 3 when
    # normalised position is exactly on a smoother position.
    assert d[0, 1] == pytest.approx((exp(0.0) / sumexp))
    T = d.transpose()
    assert sum(list(T[:, 0].as_dense_vector())) == 1.0  # should always be 1.0
    assert sum(list(T[:, 1].as_dense_vector())) == 1.0
    assert sum(list(T[:, 2].as_dense_vector())) == 1.0
    assert d[1, 1] == d[1, 2]
    assert d[1, 0] == d[1, 3]
    s, d = SF.calculate_scales_and_derivatives()

    # Test that if one or none in block, then doesn't fail but returns sensible value
    SF._normalised_values = [flex.double([0.5])]
    SF._n_refl = [1]
    assert list(SF.normalised_values[0]) == [0.5]
    s, d = SF.calculate_scales_and_derivatives()
    s2 = SF.calculate_scales()
    assert list(s) == list(s2)
    SF._normalised_values = [flex.double()]
    SF._n_refl = [0]
    assert list(SF.normalised_values[0]) == []
    s, d = SF.calculate_scales_and_derivatives()
    s2 = SF.calculate_scales()
    assert list(s) == list(s2)


def test_SmoothBScaleFactor1D():
    "test for a gaussian smoothed 1D scalefactor object"
    SF = SmoothBScaleComponent1D(flex.double(5, 0.0))
    assert SF.n_params == 5
    assert list(SF.parameters) == [0.0] * 5
    norm_rot = flex.double([0.5, 1.0, 2.5, 0.0])
    d = flex.double([1.0, 1.0, 1.0, 1.0])
    SF.data = {"x": norm_rot, "d": d}
    SF.update_reflection_data()
    assert list(SF.normalised_values[0]) == [0.5, 1.0, 2.5, 0.0]
    assert list(SF.d_values[0]) == [1.0, 1.0, 1.0, 1.0]
    SF.smoother.set_smoothing(4, 1.0)
    s, d = SF.calculate_scales_and_derivatives()
    s2 = SF.calculate_scales()
    assert list(s) == list(s2)
    assert list(s) == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert d[0, 0] / d[0, 1] == pytest.approx(
        exp(-1.0) / exp(0.0)
    )  # derivative ratio of two adjacent params (at +-0.5)
    sumexp = exp(-1.0 / 1.0) + exp(-0.0 / 1.0) + exp(-1.0 / 1.0)
    assert d[0, 1] == pytest.approx((exp(0.0) / sumexp) * s[1] / 2.0)
    T = d.transpose()
    assert sum(list(T[:, 0].as_dense_vector())) == 0.5  # value depends on d
    assert sum(list(T[:, 1].as_dense_vector())) == 0.5
    assert sum(list(T[:, 2].as_dense_vector())) == 0.5
    assert d[1, 1] == d[1, 2]
    assert d[1, 0] == d[1, 3]
    s, d = SF.calculate_scales_and_derivatives()

    SF._normalised_values = [flex.double([0.5])]
    SF._n_refl = [1]
    SF._d_values = [flex.double([1.0])]
    assert list(SF.normalised_values[0]) == [0.5]
    s, d = SF.calculate_scales_and_derivatives()
    s2 = SF.calculate_scales()
    assert list(s) == list(s2)


def test_SmoothScaleFactor2D():
    """Test the 2D smooth scale factor class."""
    with pytest.raises(AssertionError):  # Test incorrect shape initialisation
        SF = SmoothScaleComponent2D(flex.double(30, 1.1), shape=(5, 5))
    SF = SmoothScaleComponent2D(flex.double(30, 1.1), shape=(6, 5))
    assert SF.n_x_params == 6
    assert SF.n_y_params == 5
    assert SF.n_params == 30

    assert list(SF.parameters) == [1.1] * 30
    norm_rot = flex.double(30, 0.5)
    norm_time = flex.double(30, 0.5)
    norm_rot[0] = 0.0
    norm_time[0] = 0.0
    norm_rot[29] = 3.99
    norm_time[29] = 2.99
    SF.data = {"x": norm_rot, "y": norm_time}
    SF.update_reflection_data()
    # assert list(SF.normalised_x_values) == list(flex.double(
    #  [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]))
    # assert list(SF.normalised_y_values) == list(flex.double(
    #  [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]))
    SF.smoother.set_smoothing(4, 1.0)  # will average 3 in x,y dims.
    assert list(SF.smoother.x_positions()) == [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    assert list(SF.smoother.y_positions()) == [-0.5, 0.5, 1.5, 2.5, 3.5]
    s, d = SF.calculate_scales_and_derivatives()
    assert list(s) == pytest.approx([1.1] * 30)
    sumexp = exp(0.0) + (4.0 * exp(-1.0 / 1.0)) + (4.0 * exp(-2.0 / 1.0))
    assert d[1, 7] == pytest.approx(exp(-0.0) / sumexp)

    # Test again with a small number of params to check different behaviour.
    SF = SmoothScaleComponent2D(flex.double(6, 1.1), shape=(3, 2))
    _ = flex.reflection_table()
    norm_rot = flex.double(6, 0.5)
    norm_time = flex.double(6, 0.5)
    norm_rot[0] = 0.0
    norm_time[0] = 0.0
    norm_rot[5] = 1.99
    norm_time[5] = 0.99
    SF.data = {"x": norm_rot, "y": norm_time}
    SF.update_reflection_data()
    SF.smoother.set_smoothing(4, 1.0)  # will average 3,2 in x,y dims.
    assert list(SF.smoother.x_positions()) == [0.0, 1.0, 2.0]
    assert list(SF.smoother.y_positions()) == [0.0, 1.0]
    s, d = SF.calculate_scales_and_derivatives()
    s2 = SF.calculate_scales()
    assert list(s) == list(s2)
    sumexp = (4.0 * exp(-0.5 / 1.0)) + (2.0 * exp(-2.5 / 1.0))
    assert d[1, 1] == pytest.approx(exp(-0.5) / sumexp)

    # Test that if one or none in block, then doesn't fail but returns sensible value
    SF._normalised_x_values = [flex.double([0.5])]
    SF._normalised_y_values = [flex.double([0.5])]
    SF._n_refl = [1]
    assert list(SF.normalised_x_values[0]) == [0.5]
    assert list(SF.normalised_y_values[0]) == [0.5]
    s, d = SF.calculate_scales_and_derivatives()
    s2 = SF.calculate_scales()
    assert list(s) == list(s2)
    assert list(s) == pytest.approx([1.1])
    SF._normalised_x_values = [flex.double()]
    SF._normalised_y_values = [flex.double()]
    SF._n_refl = [0]
    assert list(SF.normalised_x_values[0]) == []
    assert list(SF.normalised_y_values[0]) == []
    s, d = SF.calculate_scales_and_derivatives()
    s2 = SF.calculate_scales()
    assert list(s) == list(s2)
    assert list(s) == []
    assert d == sparse.matrix(0, 0)


def test_SmoothScaleFactor3D():
    """Test the 2D smooth scale factor class."""
    with pytest.raises(AssertionError):  # Test incorrect shape initialisation
        SF = SmoothScaleComponent3D(flex.double(150, 1.1), shape=(5, 5, 5))
    SF = SmoothScaleComponent3D(flex.double(150, 1.1), shape=(6, 5, 5))
    assert SF.n_x_params == 6
    assert SF.n_y_params == 5
    assert SF.n_z_params == 5
    assert SF.n_params == 150

    assert list(SF.parameters) == [1.1] * 150
    norm_rot = flex.double(150, 0.5)
    norm_time = flex.double(150, 0.5)
    norm_z = flex.double(150, 0.5)
    norm_rot[0] = 0.0
    norm_time[0] = 0.0
    norm_z[0] = 0.0
    norm_rot[149] = 3.99
    norm_time[149] = 2.99
    norm_z[149] = 2.99
    SF.data = {"x": norm_rot, "y": norm_time, "z": norm_z}
    SF.update_reflection_data()
    SF.smoother.set_smoothing(3, 1.0)  # will average 3 in x,y,z dims for test.
    assert list(SF.smoother.x_positions()) == [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    assert list(SF.smoother.y_positions()) == [-0.5, 0.5, 1.5, 2.5, 3.5]
    assert list(SF.smoother.z_positions()) == [-0.5, 0.5, 1.5, 2.5, 3.5]
    s, d = SF.calculate_scales_and_derivatives()
    s2 = SF.calculate_scales()
    assert list(s) == list(s2)
    assert list(s) == pytest.approx([1.1] * 150)
    sumexp = (
        exp(-0.0)
        + (6.0 * exp(-1.0 / 1.0))
        + (8.0 * exp(-3.0 / 1.0))
        + (12.0 * exp(-2.0 / 1.0))
    )
    assert d[1, 7] == pytest.approx(exp(-1.0) / sumexp)  # Just check one

    # Test that if one or none in block, then doesn't fail but returns sensible value
    SF._normalised_x_values = [flex.double([0.5])]
    SF._normalised_y_values = [flex.double([0.5])]
    SF._normalised_z_values = [flex.double([0.5])]
    SF._n_refl = [1]
    assert list(SF.normalised_x_values[0]) == [0.5]
    assert list(SF.normalised_y_values[0]) == [0.5]
    assert list(SF.normalised_z_values[0]) == [0.5]
    s, d = SF.calculate_scales_and_derivatives()
    s2 = SF.calculate_scales()
    assert list(s) == list(s2)
    assert list(s) == pytest.approx([1.1])
    SF._normalised_x_values = [flex.double()]
    SF._normalised_y_values = [flex.double()]
    SF._normalised_z_values = [flex.double()]
    SF._n_refl = [0]
    assert list(SF.normalised_x_values[0]) == []
    assert list(SF.normalised_y_values[0]) == []
    assert list(SF.normalised_z_values[0]) == []
    s, d = SF.calculate_scales_and_derivatives()
    assert list(s) == []
    s2 = SF.calculate_scales()
    assert list(s) == list(s2)
    assert d == sparse.matrix(0, 0)
