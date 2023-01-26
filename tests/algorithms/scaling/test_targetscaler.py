"""
Test the output of targeted scaling - by calling the scale_against_target
scaling_library function and by directly invoking the perform method scaling
of a TargetScaler.
"""

from __future__ import annotations

from math import log

import pytest

from dxtbx.model import Crystal, Experiment, ExperimentList
from libtbx import phil

from dials.algorithms.scaling.model.model import KBScalingModel
from dials.algorithms.scaling.scaler_factory import TargetScalerFactory
from dials.algorithms.scaling.scaling_library import scale_against_target
from dials.array_family import flex
from dials.util.options import ArgumentParser


def generated_target_refl():
    """Test target reflection table."""
    reflections = flex.reflection_table()
    reflections["intensity.prf.value"] = flex.double([1.0, 10.0, 1.0])
    reflections["intensity.prf.variance"] = flex.double([1.0, 10.0, 1.0])
    reflections["intensity.sum.value"] = flex.double([1.0, 10.0, 1.0])
    reflections["intensity.sum.variance"] = flex.double([1.0, 10.0, 1.0])
    reflections["miller_index"] = flex.miller_index(
        [(1, 0, 0), (0, 0, 1), (1, 0, 0)]
    )  # don't change
    reflections["d"] = flex.double([1.0, 2.0, 1.0])
    reflections["partiality"] = flex.double([1.0, 1.0, 1.0])
    reflections.set_flags(flex.bool([True, True, True]), reflections.flags.integrated)
    reflections["id"] = flex.int(3, 0)
    reflections.experiment_identifiers()[0] = str(0)
    return reflections


def generated_refl_to_scale():
    """Generate a reflection table for targeted scaling."""
    reflections = flex.reflection_table()
    reflections["intensity.prf.value"] = flex.double([2.0, 5.0, 2.0, 1.0])
    reflections["intensity.prf.variance"] = flex.double([2.0, 5.0, 2.0, 1.0])
    reflections["intensity.sum.value"] = flex.double([2.0, 5.0, 2.0, 1.0])
    reflections["intensity.sum.variance"] = flex.double([2.0, 5.0, 2.0, 1.0])
    reflections["miller_index"] = flex.miller_index(
        [(1, 0, 0), (0, 0, 1), (1, 0, 0), (10, 0, 0)]
    )  # don't change
    reflections["d"] = flex.double([1.0, 2.0, 1.0, (4.0 / 3.0) ** 0.5])
    reflections["partiality"] = flex.double([1.0, 1.0, 1.0, 1.0])
    reflections.set_flags(
        flex.bool([True, True, True, True]), reflections.flags.integrated
    )
    reflections["id"] = flex.int(4, 1)
    reflections.experiment_identifiers()[1] = str(1)
    return reflections


def generate_test_expt(idval=0):
    """Test experiments object."""
    experiments = ExperimentList()
    exp_dict = {
        "__id__": "crystal",
        "real_space_a": [1.0, 0.0, 0.0],
        "real_space_b": [0.0, 1.0, 0.0],
        "real_space_c": [0.0, 0.0, 2.0],
        "space_group_hall_symbol": " C 2y",
    }
    crystal = Crystal.from_dict(exp_dict)
    experiments.append(Experiment(crystal=crystal))
    experiments[0].identifier = str(idval)
    return experiments


@pytest.fixture
def KB_test_param():
    """Generate a params phil scope."""
    phil_scope = phil.parse(
        """
      include scope dials.algorithms.scaling.scaling_options.phil_scope
      include scope dials.algorithms.scaling.model.model.model_phil_scope
      include scope dials.algorithms.scaling.scaling_refiner.scaling_refinery_phil_str
  """,
        process_includes=True,
    )

    parser = ArgumentParser(phil=phil_scope, check_format=False)
    parameters, _ = parser.parse_args(args=[], quick_parse=True, show_diff_phil=False)
    parameters.model = "KB"
    parameters.reflection_selection.method = "use_all"
    return parameters


def test_scale_against_target(KB_test_param):
    """Integration testing of the scale_against_target library function/targeted
    scaling."""
    # Based on input - have Target Ih of 1.0, 10.0 and d of 1.0, 2.0. Then refl to
    # target have I's of 2 and 5, (with the same ds). Therefore for a KB model the
    # problem can be minimised exactly by solving the equations:
    # 2 = K * exp(B/2)
    # 1/2 = K * exp(B/8)
    # Solving these gives the form tested for at the end of this test.
    target_reflections = generated_target_refl()
    reflections = generated_refl_to_scale()
    target_experiments = generate_test_expt()
    experiments = generate_test_expt(idval=1)
    scaled_reflections = scale_against_target(
        reflections, experiments, target_reflections, target_experiments, KB_test_param
    )
    assert list(scaled_reflections["inverse_scale_factor"]) == pytest.approx(
        [2.0, 0.5, 2.0, 2.0 * (4.0 ** (-1.0 / 3.0))]
    )

    experiments = generate_test_expt()
    experiments.append(generate_test_expt(idval=1)[0])
    experiments[0].scaling_model = KBScalingModel.from_data(KB_test_param, [], [])
    experiments[0].scaling_model.set_scaling_model_as_scaled()
    experiments[1].scaling_model = KBScalingModel.from_data(KB_test_param, [], [])
    target_reflections = generated_target_refl()
    reflections = generated_refl_to_scale()
    # Repeat the test but calling the TargetScaler directly, to allow inspection
    # of the model components.
    targetscaler = TargetScalerFactory.create(
        KB_test_param, experiments, [target_reflections, reflections]
    )
    targetscaler.perform_scaling()
    assert list(
        targetscaler.unscaled_scalers[0].components["scale"].parameters
    ) == pytest.approx([(4.0 ** (-1.0 / 3.0)) / 2.0])
    assert list(
        targetscaler.unscaled_scalers[0].components["decay"].parameters
    ) == pytest.approx([(log(4.0) * 8.0 / 3.0)])
