"""
Tests for the functions in dials.util.options
"""
from __future__ import absolute_import, division, print_function

from mock import Mock
from dials.util.options import (
    flatten_experiments,
    OptionParser,
    reflections_and_experiments_from_files,
)
from dials.test.util import mock_reflection_file_object, mock_two_reflection_file_object
from dxtbx.model import Experiment, ExperimentList


def test_can_read_headerless_h5_and_no_detector_is_present(dials_data):
    data_h5 = dials_data("vmxi_thaumatin").join("image_15799_data_000001.h5").strpath
    parser = OptionParser(read_experiments=True, read_experiments_from_images=True)
    params, _ = parser.parse_args([data_h5])
    experiments = flatten_experiments(params.input.experiments)
    assert len(experiments) == 1
    assert not experiments[0].detector


def test_reflections_and_experiments_from_files():
    """Test correct extracting of reflections and experiments."""
    # Test when input reflections order matches the experiments order
    refl_file_list = [
        mock_two_reflection_file_object(ids=[0, 1]),
        mock_reflection_file_object(id_=2),
    ]

    def mock_exp_obj(id_=0):
        """Make a mock experiments file object."""
        exp = Mock()
        exp.data = ExperimentList()
        exp.data.append(Experiment(identifier=str(id_)))
        return exp

    exp_file_list = [mock_exp_obj(id_=i) for i in [0, 1, 2]]

    refls, expts = reflections_and_experiments_from_files(refl_file_list, exp_file_list)
    assert refls[0] is refl_file_list[0].data
    assert refls[1] is refl_file_list[1].data
    assert expts[0].identifier == "0"
    assert expts[1].identifier == "1"
    assert expts[2].identifier == "2"

    # Test when input reflections order does not match experiments order.
    refl_file_list = [
        mock_reflection_file_object(id_=2),
        mock_two_reflection_file_object(ids=[0, 1]),
    ]
    refls, expts = reflections_and_experiments_from_files(refl_file_list, exp_file_list)
    assert refls[0] is refl_file_list[1].data
    assert refls[1] is refl_file_list[0].data
    assert expts[0].identifier == "0"
    assert expts[1].identifier == "1"
    assert expts[2].identifier == "2"
