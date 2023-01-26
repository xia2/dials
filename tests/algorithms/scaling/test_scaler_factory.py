"""
Tests for the scaler factory classes and helper functions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from dxtbx.model import Crystal
from libtbx import phil

from dials.algorithms.scaling.error_model.error_model import BasicErrorModel
from dials.algorithms.scaling.scaler import (
    MultiScaler,
    NullScaler,
    SingleScaler,
    TargetScaler,
)
from dials.algorithms.scaling.scaler_factory import (
    MultiScalerFactory,
    SingleScalerFactory,
    TargetScalerFactory,
    create_scaler,
)
from dials.array_family import flex
from dials.util.options import ArgumentParser


def generated_refl(not_integrated=False, idval=0):
    """Generate a test reflection table."""
    reflections = flex.reflection_table()
    reflections["intensity.prf.value"] = flex.double([1.0, 10.0, 100.0, 1.0])
    reflections["intensity.prf.variance"] = flex.double([1.0, 10.0, 100.0, 1.0])
    reflections["intensity.sum.value"] = flex.double([12.0, 120.0, 1200.0, 21.0])
    reflections["intensity.sum.variance"] = flex.double([12.0, 120.0, 2100.0, 1.0])
    # reflections['inverse_scale_factor'] = flex.double(4, 1.0)
    reflections["miller_index"] = flex.miller_index(
        [(1, 0, 0), (0, 0, 1), (2, 0, 0), (2, 2, 2)]
    )  # don't change
    reflections["d"] = flex.double([0.8, 2.0, 2.0, 0.0])  # don't change
    reflections["partiality"] = flex.double([1.0, 1.0, 1.0, 1.0])
    reflections["xyzobs.px.value"] = flex.vec3_double(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 10.0), (0.0, 0.0, 10.0)]
    )
    reflections["s1"] = flex.vec3_double(
        [(0.0, 0.1, 1.0), (0.0, 0.1, 1.0), (0.0, 0.1, 1.0), (0.0, 0.1, 1.0)]
    )
    if not_integrated:
        reflections.set_flags(
            flex.bool([False, False, False, False]), reflections.flags.integrated
        )
    else:
        reflections.set_flags(
            flex.bool([True, True, False, False]), reflections.flags.integrated
        )
    reflections.set_flags(
        flex.bool([False, False, True, True]), reflections.flags.bad_for_scaling
    )
    reflections["id"] = flex.int(4, idval)
    reflections.experiment_identifiers()[idval] = str(idval)
    return reflections


@pytest.fixture
def refl_to_filter():
    """Generate a separate reflection table for filtering"""
    reflections = flex.reflection_table()
    reflections["partiality"] = flex.double([0.1, 1.0, 1.0, 1.0, 1.0, 1.0])
    reflections["id"] = flex.int(reflections.size(), 0)
    reflections["intensity.sum.value"] = flex.double([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    reflections["intensity.sum.variance"] = flex.double([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    reflections.set_flags(
        flex.bool([True, False, True, True, True, True]),
        reflections.flags.integrated_sum,
    )
    return reflections


@pytest.fixture
def prf_sum_refl_to_filter():
    """Generate a separate reflection table for filtering"""
    reflections = flex.reflection_table()
    reflections["partiality"] = flex.double(5, 1.0)
    reflections["id"] = flex.int(reflections.size(), 0)
    reflections.experiment_identifiers()[0] = "0"
    reflections["intensity.sum.value"] = flex.double([1.0, 2.0, 3.0, 4.0, 5.0])
    reflections["intensity.sum.variance"] = flex.double(5, 1.0)
    reflections["intensity.prf.value"] = flex.double([11.0, 12.0, 13.0, 14.0, 15.0])
    reflections["intensity.prf.variance"] = flex.double(5, 1.0)
    reflections["miller_index"] = flex.miller_index([(0, 0, 1)] * 5)
    reflections.set_flags(
        flex.bool([False, False, True, True, True]),
        reflections.flags.integrated_sum,
    )
    reflections.set_flags(
        flex.bool([True, False, False, True, True]),
        reflections.flags.integrated_prf,
    )
    return reflections


def generate_refls_expts(mock_scaling_component, n=1):
    rlist = []
    explist = []
    for i in range(n):
        rlist.append(generated_refl(idval=i))
        explist.append(mock_exp(mock_scaling_component, idval=i))
    return rlist, explist


@pytest.fixture
def refl_list():
    """Make a list of three reflection tables."""
    refl_list = [generated_refl()]
    refl_list.append(generated_refl())
    refl_list.append(generated_refl())
    return refl_list


@pytest.fixture
def generated_param():
    """Generate a param phil scope."""
    phil_scope = phil.parse(
        """
      include scope dials.algorithms.scaling.scaling_options.phil_scope
  """,
        process_includes=True,
    )
    parser = ArgumentParser(phil=phil_scope, check_format=False)
    parameters, _ = parser.parse_args(args=[], quick_parse=True, show_diff_phil=False)
    parameters.__inject__("model", "KB")
    parameters.scaling_options.free_set_percentage = 50.0
    parameters.scaling_options.emax = 0
    parameters.reflection_selection.method = "use_all"
    return parameters


@pytest.fixture
def mock_scaling_component():
    """Mock scaling component to allow creation of a scaling model."""
    component = MagicMock()
    component.n_params = 2
    component.inverse_scales = flex.double([0.9, 1.1])
    return component


def mock_exp(mock_scaling_component, idval=0):
    """Mock experiments object for initialising a scaler."""

    def side_effect_config_table(*args):
        """Side effect to mock configure reflection table
        call during initialisation."""
        return args[0]

    exp = MagicMock()
    exp.identifier = str(idval)
    exp.scaling_model.components = {"scale": mock_scaling_component}
    exp.scaling_model.consecutive_refinement_order = ["scale"]
    exp.scaling_model.is_scaled = False
    exp.scaling_model.error_model = BasicErrorModel()
    exp.scaling_model.configure_reflection_table.side_effect = side_effect_config_table
    exp_dict = {
        "__id__": "crystal",
        "real_space_a": [1.0, 0.0, 0.0],
        "real_space_b": [0.0, 1.0, 0.0],
        "real_space_c": [0.0, 0.0, 2.0],
        "space_group_hall_symbol": " C 2y",
    }
    exp.crystal = Crystal.from_dict(exp_dict)
    exp.scan.get_oscillation.return_value = (0, 1.0)
    exp.beam.get_sample_to_source_direction.return_value = (0.0, 0.0, -1.0)
    exp.goniometer.get_rotation_axis.return_value = (0.0, 0.0, 1.0)
    return exp


def mock_explist_3exp(mock_scaling_component):
    """A mock experimentlist, containing one mock exp instance three times."""
    exp = [mock_exp(mock_scaling_component)]
    exp.append(mock_exp(mock_scaling_component))
    exp.append(mock_exp(mock_scaling_component))
    return exp


@pytest.fixture
def mock_scaled_exp():
    """A mock experiments object with scaling_model.is_scaled = True"""
    exp = Mock()
    exp.scaling_model.is_scaled = True
    return exp


@pytest.fixture
def mock_unscaled_exp():
    """A mock experiments object with scaling_model.is_scaled = False"""
    exp = Mock()
    exp.scaling_model.is_scaled = False
    return exp


@pytest.fixture
def mock_experimentlist(mock_scaled_exp, mock_unscaled_exp):
    """A mock experimentlist of mock scaled/unscaled mock exp."""
    explist = [
        mock_scaled_exp,
        mock_scaled_exp,
        mock_unscaled_exp,
        mock_scaled_exp,
        mock_unscaled_exp,
    ]
    return explist


def test_SingleScalerFactory(generated_param, refl_to_filter, mock_scaling_component):
    """Test the single scaler factory."""
    test_refls, expts = generate_refls_expts(mock_scaling_component)
    # Test that all required attributes get added with standard params.
    assert all(
        (i not in test_refls[0])
        for i in ["inverse_scale_factor", "intensity", "variance"]
    )
    # Test default, (no split into free set)
    ss = SingleScalerFactory.create(generated_param, expts[0], test_refls[0])
    assert isinstance(ss, SingleScaler)
    assert all(
        i in ss.reflection_table
        for i in ["inverse_scale_factor", "intensity", "variance"]
    )

    # Test reflection filtering
    rt = SingleScalerFactory.filter_bad_reflections(refl_to_filter)
    assert list(rt.get_flags(rt.flags.excluded_for_scaling)) == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_selection_of_profile_or_summation_intensities(
    generated_param, prf_sum_refl_to_filter, mock_scaling_component
):
    _, expts = generate_refls_expts(mock_scaling_component)
    # Test that all required attributes get added with standard params.
    assert all(
        (i not in prf_sum_refl_to_filter)
        for i in ["inverse_scale_factor", "intensity", "variance"]
    )
    # Test default, (no split into free set)
    ss = SingleScalerFactory.create(generated_param, expts[0], prf_sum_refl_to_filter)
    assert isinstance(ss, SingleScaler)
    rt = ss.reflection_table
    assert all(i in rt for i in ["inverse_scale_factor", "intensity", "variance"])
    assert list(rt.get_flags(rt.flags.excluded_for_scaling)) == [
        False,
        True,
        False,
        False,
        False,
    ]
    # test correct initial intensities have been chosen - should be prf then sum
    assert list(rt["intensity"]) == [11.0, 2.0, 3.0, 14.0, 15.0]


def test_TargetScalerFactory(generated_param, mock_scaling_component):
    """Test the target scaler factory."""

    refl_list, explist = generate_refls_expts(mock_scaling_component, 3)

    # Test standard initialisation.
    assert generated_param.scaling_options.use_free_set is False  # just to check
    explist[0].scaling_model.is_scaled = True
    explist[1].scaling_model.is_scaled = True
    target = TargetScalerFactory.create(generated_param, explist, refl_list)
    assert isinstance(target, TargetScaler)
    assert len(target.single_scalers) == 2
    assert len(target.unscaled_scalers) == 1
    assert set(target.single_scalers[0].reflection_table["id"]) == {0}
    assert set(target.single_scalers[1].reflection_table["id"]) == {1}
    assert set(target.unscaled_scalers[0].reflection_table["id"]) == {2}

    # Now test converting targetscaler to multiscaler
    multiscaler = MultiScalerFactory.create_from_targetscaler(target)
    assert isinstance(multiscaler, MultiScaler)
    assert len(multiscaler.single_scalers) == 3

    # Test for correct initialisation when scaling against a target model.
    generated_param.scaling_options.reference = True
    target = TargetScalerFactory.create_for_target_against_reference(
        generated_param, explist, refl_list
    )
    assert isinstance(target.single_scalers[0], NullScaler)

    # This time make one dataset bad, and check it gets removed
    refl_list, explist = generate_refls_expts(mock_scaling_component, 3)
    generated_param.scaling_options.reference = False
    refl_list[1].unset_flags(flex.bool(4, True), refl_list[1].flags.integrated_prf)
    refl_list[1].unset_flags(flex.bool(4, True), refl_list[1].flags.integrated_sum)
    explist[0].scaling_model.is_scaled = True
    target = TargetScalerFactory.create(generated_param, explist, refl_list)
    assert isinstance(target, TargetScaler)
    assert len(target.single_scalers) == 1
    assert len(target.unscaled_scalers) == 1
    assert set(target.single_scalers[0].reflection_table["id"]) == {0}
    assert set(target.unscaled_scalers[0].reflection_table["id"]) == {2}

    refl_list, explist = generate_refls_expts(mock_scaling_component, 3)
    refl_list[0].unset_flags(flex.bool(4, True), refl_list[0].flags.integrated_prf)
    refl_list[0].unset_flags(flex.bool(4, True), refl_list[0].flags.integrated_sum)
    explist[0].scaling_model.is_scaled = True
    explist[1].scaling_model.is_scaled = True
    target = TargetScalerFactory.create(generated_param, explist, refl_list)
    assert isinstance(target, TargetScaler)
    assert len(target.single_scalers) == 1
    assert len(target.unscaled_scalers) == 1
    assert set(target.single_scalers[0].reflection_table["id"]) == {1}
    assert set(target.unscaled_scalers[0].reflection_table["id"]) == {2}


def test_MultiScalerFactory(generated_param, mock_scaling_component, refl_list):
    """Test the MultiScalerFactory."""

    refl_list, explist = generate_refls_expts(mock_scaling_component, 3)

    multiscaler = MultiScalerFactory.create(generated_param, explist, refl_list)
    assert isinstance(multiscaler, MultiScaler)
    assert len(multiscaler.single_scalers) == 3
    for i in range(3):
        assert set(multiscaler.single_scalers[i].reflection_table["id"]) == {i}

    # This time make one dataset bad, and check it gets removed
    r1 = generated_refl(not_integrated=True)
    r2 = generated_refl()
    r3 = generated_refl()
    new_list = [r1, r2, r3]
    multiscaler = MultiScalerFactory.create(
        generated_param, mock_explist_3exp(mock_scaling_component), new_list
    )
    assert isinstance(multiscaler, MultiScaler)
    assert len(multiscaler.single_scalers) == 2
    r1 = multiscaler.single_scalers[0].reflection_table
    assert list(r1.get_flags(r1.flags.integrated)) == [True, True, False, False]
    r2 = multiscaler.single_scalers[1].reflection_table
    assert list(r2.get_flags(r2.flags.integrated)) == [True, True, False, False]


def test_scaler_factory_helper_functions(
    mock_experimentlist, generated_param, refl_list, mock_scaling_component
):
    """Test the helper functions."""

    test_refls, expts = generate_refls_expts(mock_scaling_component)

    # Test create_scaler
    # Test case for single refl and exp
    scaler = create_scaler(generated_param, expts, test_refls)
    assert isinstance(scaler, SingleScaler)

    # If none or allscaled
    explist = mock_explist_3exp(mock_scaling_component)
    scaler = create_scaler(generated_param, explist, refl_list)
    assert isinstance(scaler, MultiScaler)

    explist[0].scaling_model.is_scaled = False
    # ^ changes all in list as same instance of exp.
    scaler = create_scaler(
        generated_param, mock_explist_3exp(mock_scaling_component), refl_list
    )
    assert isinstance(scaler, MultiScaler)

    # If only some scaled
    explist = []
    explist.append(mock_exp(mock_scaling_component))
    explist.append(mock_exp(mock_scaling_component))
    explist[1].scaling_model.is_scaled = True
    r1 = generated_refl()
    r2 = generated_refl()
    refl_list = [r1, r2]
    scaler = create_scaler(generated_param, explist, refl_list)
    assert isinstance(scaler, TargetScaler)

    # If no reflections passed in.
    with pytest.raises(ValueError):
        scaler = create_scaler(
            generated_param, mock_explist_3exp(mock_scaling_component), []
        )
