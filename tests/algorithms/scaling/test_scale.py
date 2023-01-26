"""
Test the command line script dials.scale, for successful completion.
"""

from __future__ import annotations

import json
import os

import procrunner
import pytest

import iotbx.merging_statistics
import iotbx.mtz
from cctbx import uctbx
from dxtbx.model import Beam, Crystal, Detector, Experiment, Goniometer, Scan
from dxtbx.model.experiment_list import ExperimentList
from dxtbx.serialize import load
from libtbx import phil

from dials.algorithms.scaling.algorithm import ScalingAlgorithm, prepare_input
from dials.array_family import flex
from dials.command_line import merge, report, scale
from dials.util.options import ArgumentParser


def run_one_scaling(working_directory, argument_list):
    """Run the dials.scale algorithm."""
    command = ["dials.scale"] + argument_list
    print(command)
    result = procrunner.run(command, working_directory=working_directory)
    print(result.stderr)
    assert not result.returncode and not result.stderr
    assert (working_directory / "scaled.expt").is_file()
    assert (working_directory / "scaled.refl").is_file()
    assert (working_directory / "dials.scale.html").is_file()

    table = flex.reflection_table.from_file(working_directory / "scaled.refl")

    assert "inverse_scale_factor" in table
    assert "inverse_scale_factor_variance" in table


def get_merging_stats(
    scaled_unmerged_mtz,
    anomalous=False,
    n_bins=20,
    use_internal_variance=False,
    eliminate_sys_absent=False,
    data_labels=None,
):
    """Return a merging statistics result from an mtz file."""

    i_obs = iotbx.merging_statistics.select_data(
        str(scaled_unmerged_mtz), data_labels=data_labels
    )
    i_obs = i_obs.customized_copy(anomalous_flag=False, info=i_obs.info())
    result = iotbx.merging_statistics.dataset_statistics(
        i_obs=i_obs,
        n_bins=n_bins,
        anomalous=anomalous,
        use_internal_variance=use_internal_variance,
        eliminate_sys_absent=eliminate_sys_absent,
    )
    return result


def generated_exp(n=1):
    """Generate an experiment list with two experiments."""
    experiments = ExperimentList()
    exp_dict = {
        "__id__": "crystal",
        "real_space_a": [1.0, 0.0, 0.0],
        "real_space_b": [0.0, 1.0, 0.0],
        "real_space_c": [0.0, 0.0, 2.0],
        "space_group_hall_symbol": " C 2y",
    }
    crystal = Crystal.from_dict(exp_dict)
    scan = Scan(image_range=[0, 90], oscillation=[0.0, 1.0])
    beam = Beam(s0=(0.0, 0.0, 1.01))
    goniometer = Goniometer((1.0, 0.0, 0.0))
    detector = Detector()
    experiments.append(
        Experiment(
            beam=beam,
            scan=scan,
            goniometer=goniometer,
            detector=detector,
            crystal=crystal,
        )
    )
    if n > 1:
        for _ in range(n - 1):
            experiments.append(
                Experiment(
                    beam=beam,
                    scan=scan,
                    goniometer=goniometer,
                    detector=detector,
                    crystal=crystal,
                )
            )
    return experiments


def generated_param():
    """Generate the default scaling parameters object."""
    phil_scope = phil.parse(
        """
      include scope dials.command_line.scale.phil_scope
  """,
        process_includes=True,
    )

    parser = ArgumentParser(phil=phil_scope, check_format=False)
    parameters, _ = parser.parse_args(args=[], quick_parse=True, show_diff_phil=False)
    return parameters


def generate_test_reflections():
    """Generate a small reflection table."""
    reflections = flex.reflection_table()
    reflections["intensity.sum.value"] = flex.double([1.0, 2.0, 3.0, 4.0])
    reflections["intensity.sum.variance"] = flex.double([1.0, 2.0, 3.0, 4.0])
    reflections["miller_index"] = flex.miller_index(
        [(0, 0, 1), (0, 0, 1), (0, 0, 2), (0, 0, 2)]
    )
    reflections["xyzobs.px.value"] = flex.vec3_double(
        [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]
    )
    reflections["id"] = flex.int([0, 0, 0, 0])
    return reflections


def generate_test_input(n=1):
    """Generate params, exps and refls."""
    reflections = []
    for _ in range(n):
        reflections.append(generate_test_reflections())
    return generated_param(), generated_exp(n), reflections


def test_scale_script_prepare_input():
    """Test prepare_input method of scaling script."""

    # test the components of the scaling script directly with a test reflection
    # table, experiments list and params.

    params, exp, reflections = generate_test_input()
    # try to pass in unequal number of reflections and experiments
    reflections.append(generate_test_reflections())
    with pytest.raises(ValueError):
        _ = ScalingAlgorithm(params, exp, reflections)

    params, exp, reflections = generate_test_input()
    # Try to use use_datasets when not identifiers set
    params.dataset_selection.use_datasets = [0]
    with pytest.raises(ValueError):
        _ = ScalingAlgorithm(params, exp, reflections)
    # Try to use use_datasets when not identifiers set
    params.dataset_selection.use_datasets = None
    params.dataset_selection.exclude_datasets = [0]
    with pytest.raises(ValueError):
        _ = ScalingAlgorithm(params, exp, reflections)

    # Now make two experiments with identifiers and select on them
    params, exp, reflections = generate_test_input(n=2)
    exp[0].identifier = "0"
    reflections[0].experiment_identifiers()[0] = "0"
    exp[1].identifier = "1"
    reflections[1].experiment_identifiers()[0] = "1"
    list1 = ExperimentList().append(exp[0])
    list2 = ExperimentList().append(exp[1])
    reflections[0].assert_experiment_identifiers_are_consistent(list1)
    reflections[1].assert_experiment_identifiers_are_consistent(list2)
    params.dataset_selection.use_datasets = [0]
    params, exp, script_reflections = prepare_input(params, exp, reflections)

    assert len(script_reflections) == 1

    # Try again, this time excluding
    params, exp, reflections = generate_test_input(n=2)
    exp[0].identifier = "0"
    reflections[0].experiment_identifiers()[0] = "0"
    exp[1].identifier = "1"
    reflections[1].experiment_identifiers()[1] = "1"
    params.dataset_selection.exclude_datasets = [0]
    params, exp, script_reflections = prepare_input(params, exp, reflections)

    assert len(script_reflections) == 1
    assert script_reflections[0] is reflections[1]

    # Try having two unequal space groups
    params, exp, reflections = generate_test_input(n=2)
    exp_dict = {
        "__id__": "crystal",
        "real_space_a": [1.0, 0.0, 0.0],
        "real_space_b": [0.0, 1.0, 0.0],
        "real_space_c": [0.0, 0.0, 2.0],
        "space_group_hall_symbol": " P 1",
    }
    crystal = Crystal.from_dict(exp_dict)
    exp[0].crystal = crystal
    with pytest.raises(ValueError):
        _ = prepare_input(params, exp, reflections)

    # Test cutting data
    params, exp, reflections = generate_test_input(n=1)
    params.cut_data.d_min = 1.5
    params, _, script_reflections = prepare_input(params, exp, reflections)
    r = script_reflections[0]
    assert list(r.get_flags(r.flags.user_excluded_in_scaling)) == [
        False,
        False,
        True,
        True,
    ]

    # Ensure that the user_excluded_in_scaling flags are reset before applying any new
    # cutoffs by re-passing script_reflections to prepare_input
    params.cut_data.d_min = None
    params, _, script_reflections = prepare_input(params, exp, script_reflections)
    r = script_reflections[0]
    assert list(r.get_flags(r.flags.user_excluded_in_scaling)) == [
        False,
        False,
        False,
        False,
    ]

    params.cut_data.d_max = 1.25
    params, _, script_reflections = prepare_input(params, exp, reflections)
    r = script_reflections[0]
    assert list(r.get_flags(r.flags.user_excluded_in_scaling)) == [
        True,
        True,
        False,
        False,
    ]

    params, exp, reflections = generate_test_input(n=1)
    reflections[0]["partiality"] = flex.double([0.5, 0.8, 1.0, 1.0])
    params.cut_data.partiality_cutoff = 0.75
    _, __, script_reflections = prepare_input(params, exp, reflections)
    r = script_reflections[0]
    assert list(r.get_flags(r.flags.user_excluded_in_scaling)) == [
        True,
        False,
        False,
        False,
    ]


def test_targeted_scaling_against_mtz(dials_data, tmp_path):
    """Test targeted scaling against an mtz generated with dials.scale."""
    location = dials_data("l_cysteine_4_sweeps_scaled", pathlib=True)
    refl = location / "scaled_35.refl"
    expt = location / "scaled_35.expt"
    command = ["dials.scale", refl, expt, "unmerged_mtz=unmerged.mtz"]

    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.expt").is_file()
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "unmerged.mtz").is_file()

    refl = location / "scaled_30.refl"
    expt = location / "scaled_30.expt"
    reference = tmp_path / "unmerged.mtz"
    command = [
        "dials.scale",
        refl,
        expt,
        f"reference={reference}",
        "unmerged_mtz=unmerged_2.mtz",
    ]

    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.expt").is_file()
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "unmerged_2.mtz").is_file()
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert len(expts) == 1
    result = get_merging_stats(tmp_path / "unmerged_2.mtz")
    assert result.overall.r_pim < 0.024  # at 30/03/22, value was 0.022
    assert result.overall.cc_one_half > 0.995  # at 30/03/22, value was 0.998
    assert result.overall.n_obs > 3150  # at 30/03/22, value was 3188


@pytest.mark.parametrize(
    "option",
    [
        None,
        "reflection_selection.method=random",
        "reflection_selection.method=intensity_ranges",
        "reflection_selection.method=use_all",
        "intensity_choice=sum",
        "intensity=profile",
    ],
)
def test_scale_single_dataset_with_options(dials_data, tmp_path, option):
    """Test different non-default command-line options with a single dataset."""
    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    args = [refl_1, expt_1]
    if option:
        args.append(option)
    run_one_scaling(tmp_path, args)


@pytest.fixture(scope="session")
def vmxi_protk_reindexed(dials_data, tmp_path_factory):
    """Reindex the protk data to be in the correct space group."""
    location = dials_data("vmxi_proteinase_k_sweeps", pathlib=True)

    command = [
        "dials.reindex",
        location / "experiments_0.json",
        location / "reflections_0.pickle",
        "space_group=P422",
    ]
    tmp_path = tmp_path_factory.mktemp("vmxi_protk_reindexed")
    procrunner.run(command, working_directory=tmp_path)
    return tmp_path / "reindexed.expt", tmp_path / "reindexed.refl"


@pytest.mark.parametrize(
    ("options", "expected", "tolerances"),
    [
        (["error_model=None"], None, None),
        (
            ["error_model=basic", "basic.minimisation=individual"],
            (0.61, 0.049),
            (0.05, 0.005),
        ),
        (["error_model.basic.a=0.61"], (0.61, 0.049), (1e-6, 0.005)),
        (["error_model.basic.b=0.049"], (0.61, 0.049), (0.05, 1e-6)),
        (
            ["error_model.basic.b=0.02", "error_model.basic.a=1.5"],
            (1.50, 0.02),
            (1e-6, 1e-6),
        ),
        (
            ["error_model=basic", "basic.minimisation=regression"],
            (0.995, 0.051),
            (0.05, 0.005),
        ),
        (
            ["error_model.basic.a=0.99", "basic.minimisation=regression"],
            (0.99, 0.051),
            (1e-6, 0.005),
        ),
        (
            ["error_model.basic.b=0.051", "basic.minimisation=regression"],
            (1.0, 0.051),
            (0.05, 1e-6),
        ),
    ],
)
def test_error_model_options(
    vmxi_protk_reindexed, tmp_path, options, expected, tolerances
):
    """Test different non-default command-line options with a single dataset.

    Current values taken at 14.11.19"""
    expt_1, refl_1 = vmxi_protk_reindexed
    args = [refl_1, expt_1] + list(options)
    run_one_scaling(tmp_path, args)
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    config = expts[0].scaling_model.configdict
    if not expected:
        assert "error_model_parameters" not in config
    else:
        params = expts[0].scaling_model.configdict["error_model_parameters"]
        print(list(params))
        assert params[0] == pytest.approx(expected[0], abs=tolerances[0])
        assert params[1] == pytest.approx(expected[1], abs=tolerances[1])


@pytest.mark.parametrize(
    "option",
    [
        None,
        "combine.joint_analysis=False",
        "reflection_selection.method=quasi_random",
        "reflection_selection.method=random",
        "reflection_selection.method=intensity_ranges",
        "reflection_selection.method=use_all",
        "anomalous=True",
    ],
)
def test_scale_multiple_datasets_with_options(dials_data, tmp_path, option):
    """Test different non-defaul command-line options with multiple datasets."""
    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    refl_2 = data_dir / "25_integrated.pickle"
    expt_2 = data_dir / "25_integrated_experiments.json"
    args = [refl_1, expt_1, refl_2, expt_2]
    if option:
        args.append(option)
    run_one_scaling(tmp_path, args)


def test_scale_physical(dials_data, tmp_path):
    """Test standard scaling of one dataset."""

    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    extra_args = [
        "model=physical",
        "merged_mtz=merged.mtz",
        "error_model=None",
        "intensity_choice=profile",
        "unmerged_mtz=unmerged.mtz",
        "crystal_name=foo",
        "project_name=bar",
        "use_free_set=1",
        "outlier_rejection=simple",
        "json=scaling.json",
        "additional_stats=True",
    ]
    run_one_scaling(tmp_path, [refl_1, expt_1] + extra_args)
    unmerged_mtz = tmp_path / "unmerged.mtz"
    merged_mtz = tmp_path / "merged.mtz"
    assert unmerged_mtz.is_file()
    assert merged_mtz.is_file()
    assert (tmp_path / "scaling.json").is_file()
    for f in (unmerged_mtz, merged_mtz):
        mtz_obj = iotbx.mtz.object(str(f))
        assert mtz_obj.crystals()[1].name() == "foo"
        assert mtz_obj.crystals()[1].project_name() == "bar"

    # Now inspect output, check it hasn't changed drastically, or if so verify
    # that the new behaviour is more correct and update test accordingly.
    result = get_merging_stats(unmerged_mtz)
    print(result.overall.r_pim, result.overall.cc_one_half, result.overall.n_obs)
    assert result.overall.r_pim < 0.0255  # at 30/01/19, value was 0.02410
    assert result.overall.cc_one_half > 0.9955  # at 30/01/19, value was 0.9960
    assert result.overall.n_obs > 2300  # at 30/01/19, was 2320

    refls = flex.reflection_table.from_file(tmp_path / "scaled.refl")
    n_scaled = refls.get_flags(refls.flags.scaled).count(True)
    assert n_scaled == result.overall.n_obs
    assert n_scaled == refls.get_flags(refls.flags.bad_for_scaling, all=False).count(
        False
    )

    # run with a precalculated analytical correction (set to 1 for test to check result the same)
    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    r2 = flex.reflection_table.from_file(data_dir / "20_integrated.pickle")
    r2["analytical_correction"] = flex.double(r2.size(), 1.0)
    r2.as_file(tmp_path / "modified.refl")
    extra_args = [
        "model=physical",
        "physical.analytical_correction=True",
        "error_model=None",
        "intensity_choice=profile",
        "unmerged_mtz=unmerged.mtz",
        "use_free_set=1",
        "outlier_rejection=simple",
    ]
    import os

    run_one_scaling(
        tmp_path, [os.fspath(tmp_path / "modified.refl"), expt_1] + extra_args
    )
    unmerged_mtz = tmp_path / "unmerged.mtz"
    assert unmerged_mtz.is_file()
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert "analytical" in expts[0].scaling_model.components

    # Now inspect output, check it's identical
    result2 = get_merging_stats(unmerged_mtz)
    assert result2.overall.r_pim == result.overall.r_pim
    assert result2.overall.cc_one_half == result.overall.cc_one_half
    assert result2.overall.n_obs == result.overall.n_obs

    # run again with the concurrent scaling option turned off and the 'standard'
    # outlier rejection
    extra_args = [
        "model=physical",
        "merged_mtz=merged.mtz",
        "unmerged_mtz=unmerged.mtz",
        "use_free_set=1",
        "outlier_rejection=standard",
        "refinement_order=consecutive",
        "intensity_choice=combine",
    ]
    run_one_scaling(tmp_path, [refl_1, expt_1] + extra_args)

    # Now inspect output, check it hasn't changed drastically, or if so verify
    # that the new behaviour is more correct and update test accordingly.
    result = get_merging_stats(tmp_path / "unmerged.mtz")
    assert (
        result.overall.r_pim < 0.024
    )  # at 07/01/19, value was 0.02372, at 30/01/19 was 0.021498
    assert (
        result.overall.cc_one_half > 0.995
    )  # at 07/01/19, value was 0.99568, at 30/01/19 was 0.9961
    assert result.overall.n_obs > 2300  # at 07/01/19, was 2336, at 22/05/19 was 2311


def test_scale_set_absorption_level(dials_data, tmp_path):
    """Test that the absorption parameters are correctly set for the absorption option."""
    location = dials_data("l_cysteine_dials_output", pathlib=True)
    refl = location / "20_integrated.pickle"
    expt = location / "20_integrated_experiments.json"

    # exclude a central region of data to force the failure of the full matrix
    # minimiser due to indeterminate solution of the normal equations. In this
    # case, the error should be caught and scaling can proceed.
    command = [
        "dials.scale",
        refl,
        expt,
        "absorption_level=medium",
        "unmerged_mtz=unmerged.mtz",
    ]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert expts[0].scaling_model.configdict["lmax"] == 6
    assert expts[0].scaling_model.configdict["abs_surface_weight"] == 5e4
    abs_params = expts[0].scaling_model.components["absorption"].parameters
    result = get_merging_stats(tmp_path / "unmerged.mtz")
    assert result.overall.r_pim < 0.024
    assert result.overall.cc_one_half > 0.995
    assert result.overall.n_obs > 2300

    ## now scale again with different options, but fix the absorption surface to
    # test the correction.fix option.
    command = [
        "dials.scale",
        tmp_path / "scaled.refl",
        tmp_path / "scaled.expt",
        "error_model=None",
        "physical.correction.fix=absorption",
    ]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    new_abs_params = expts[0].scaling_model.components["absorption"].parameters
    assert abs_params == new_abs_params


def test_scale_normal_equations_failure(dials_data, tmp_path):
    location = dials_data("l_cysteine_dials_output", pathlib=True)
    refl = location / "20_integrated.pickle"
    expt = location / "20_integrated_experiments.json"

    # exclude a central region of data to force the failure of the full matrix
    # minimiser due to indeterminate solution of the normal equations. In this
    # case, the error should be caught and scaling can proceed.
    command = ["dials.scale", refl, expt, "exclude_images=800:1400"]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()


def test_scale_and_filter_image_group_mode(dials_data, tmp_path):
    """Test the scale and filter command line program."""
    location = dials_data("multi_crystal_proteinase_k", pathlib=True)

    command = [
        "dials.scale",
        "filtering.method=deltacchalf",
        "stdcutoff=4.0",
        "mode=image_group",
        "max_percent_removed=6.0",
        "max_cycles=6",
        "d_min=2.0",
        "group_size=5",
        "unmerged_mtz=unmerged.mtz",
        "scale_and_filter_results=analysis_results.json",
        "error_model=None",
    ]
    for i in [1, 2, 3, 4, 5, 7, 10]:
        command.append(location / f"experiments_{i}.json")
        command.append(location / f"reflections_{i}.pickle")

    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()
    assert (tmp_path / "analysis_results.json").is_file()
    result = get_merging_stats(tmp_path / "unmerged.mtz")
    assert result.overall.r_pim < 0.15  # on 04/05/22 on mac, was 0.143,
    assert result.overall.cc_one_half > 0.89  # on 04/05/22 on mac was 0.893
    assert result.overall.n_obs > 20000  # on 04/05/22 on mac was 20188

    with open(tmp_path / "analysis_results.json") as fh:
        analysis_results = json.load(fh)
    assert analysis_results["cycle_results"]["1"]["image_ranges_removed"] == [
        [[16, 24], 4]
    ]
    assert analysis_results["cycle_results"]["2"]["image_ranges_removed"] == []
    assert analysis_results["termination_reason"] == "no_more_removed"


def test_scale_and_filter_termination(dials_data, tmp_path):
    """Test the scale and filter command line program,
    when it terminates with a cycle when going over the max percent removed."""
    location = dials_data("multi_crystal_proteinase_k", pathlib=True)

    command = [
        "dials.scale",
        "filtering.method=deltacchalf",
        "stdcutoff=2.0",
        "max_percent_removed=5.0",
        "max_cycles=8",
        "d_min=2.0",
        "unmerged_mtz=unmerged.mtz",
        "scale_and_filter_results=analysis_results.json",
        "error_model=None",
        "full_matrix=False",
    ]
    for i in [1, 2, 3, 4, 5, 7, 10]:
        command.append(location / f"experiments_{i}.json")
        command.append(location / f"reflections_{i}.pickle")

    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()
    assert (tmp_path / "analysis_results.json").is_file()

    with open(tmp_path / "analysis_results.json") as fh:
        analysis_results = json.load(fh)
    assert analysis_results["termination_reason"] == "max_percent_removed"
    assert len(analysis_results["cycle_results"]["1"]["removed_datasets"]) == 1
    refls = flex.reflection_table.from_file(tmp_path / "scaled.refl")
    assert len(set(refls["id"])) == 6
    assert len(set(refls["imageset_id"])) == 6


def test_scale_and_filter_image_group_single_dataset(dials_data, tmp_path):
    """Test the scale and filter deltacchalf.mode=image_group on a
    single data set."""
    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    command = [
        "dials.scale",
        data_dir / "20_integrated.pickle",
        data_dir / "20_integrated_experiments.json",
        "filtering.method=deltacchalf",
        "stdcutoff=3.0",
        "mode=image_group",
        "max_cycles=1",
        "scale_and_filter_results=analysis_results.json",
        "error_model=None",
    ]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()
    assert (tmp_path / "analysis_results.json").is_file()

    with open(tmp_path / "analysis_results.json") as fh:
        analysis_results = json.load(fh)
    assert analysis_results["cycle_results"]["1"]["image_ranges_removed"] == []
    assert len(analysis_results["cycle_results"].keys()) == 1
    assert analysis_results["termination_reason"] == "no_more_removed"


def test_scale_when_a_dataset_is_filtered_out(dials_data, tmp_path):
    location = dials_data("multi_crystal_proteinase_k", pathlib=True)
    # Modify one of the input tables so that it gets filtered out
    # during scaler initialisation. Triggers the bug referenced in
    # https://github.com/dials/dials/issues/2045
    refls = flex.reflection_table.from_file(location / "reflections_3.pickle")
    refls["partiality"] = flex.double(refls.size(), 0.3)
    refls.as_file(tmp_path / "modified_3.refl")
    command = ["dials.scale", "d_min=2.0", tmp_path / "modified_3.refl"]
    for i in [1, 2, 3, 4]:
        command.append(location / f"experiments_{i}.json")
    for i in [1, 2, 4]:
        command.append(location / f"reflections_{i}.pickle")
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.refl").is_file()
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert len(expts) == 3


def test_scale_dose_decay_model(dials_data, tmp_path):
    """Test the scale and filter command line program."""
    location = dials_data("multi_crystal_proteinase_k", pathlib=True)
    command = ["dials.scale", "d_min=2.0", "model=dose_decay"]
    for i in [1, 2, 3, 4, 5, 7, 10]:
        command.append(location / f"experiments_{i}.json")
        command.append(location / f"reflections_{i}.pickle")

    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()
    assert (tmp_path / "dials.scale.html").is_file()
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert expts[0].scaling_model.id_ == "dose_decay"


def test_scale_best_unit_cell_d_min(dials_data, tmp_path):
    location = dials_data("multi_crystal_proteinase_k", pathlib=True)
    best_unit_cell = uctbx.unit_cell((42, 42, 39, 90, 90, 90))
    d_min = 2
    command = [
        "dials.scale",
        "best_unit_cell=%g,%g,%g,%g,%g,%g" % best_unit_cell.parameters(),
        f"d_min={d_min:g}",
        "unmerged_mtz=unmerged.mtz",
        "merged_mtz=merged.mtz",
    ]
    for i in [1, 2, 3, 4, 5, 7, 10]:
        command.append(location / f"experiments_{i}.json")
        command.append(location / f"reflections_{i}.pickle")
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()
    assert (tmp_path / "unmerged.mtz").is_file()
    assert (tmp_path / "merged.mtz").is_file()
    stats = get_merging_stats(tmp_path / "unmerged.mtz")
    assert stats.overall.d_min >= d_min
    assert stats.crystal_symmetry.unit_cell().parameters() == pytest.approx(
        best_unit_cell.parameters()
    )
    m = iotbx.mtz.object(str(tmp_path / "merged.mtz"))
    for ma in m.as_miller_arrays():
        assert best_unit_cell.parameters() == pytest.approx(ma.unit_cell().parameters())


def test_scale_and_filter_dataset_mode(dials_data, tmp_path):
    """Test the scale and filter command line program."""
    location = dials_data("multi_crystal_proteinase_k", pathlib=True)
    command = [
        "dials.scale",
        "filtering.method=deltacchalf",
        "stdcutoff=1.0",
        "mode=dataset",
        "max_cycles=2",
        "d_min=1.4",
        "output.reflections=filtered.refl",
        "scale_and_filter_results=analysis_results.json",
        "unmerged_mtz=unmerged.mtz",
        "error_model=None",
    ]
    for i in [1, 2, 3, 4, 5, 7, 10]:
        command.append(location / f"experiments_{i}.json")
        command.append(location / f"reflections_{i}.pickle")

    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "filtered.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()
    assert (tmp_path / "analysis_results.json").is_file()

    with open(tmp_path / "analysis_results.json") as fh:
        analysis_results = json.load(fh)
    assert analysis_results["cycle_results"]["1"]["removed_datasets"] == [
        analysis_results["initial_expids_and_image_ranges"][4][0]
    ]
    assert "expids_and_image_ranges" in analysis_results
    refls = flex.reflection_table.from_file(tmp_path / "filtered.refl")
    assert set(refls["id"]) == {0, 1, 2, 3, 5, 6}


def test_scale_array(dials_data, tmp_path):
    """Test a standard dataset - ideally needs a large dataset or full matrix
    round may fail. Currently turning off absorption term to avoid
    overparameterisation and failure of full matrix minimisation."""

    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    refl = data_dir / "20_integrated.pickle"
    expt = data_dir / "20_integrated_experiments.json"
    extra_args = ["model=array", "array.absorption_correction=0", "full_matrix=0"]

    run_one_scaling(tmp_path, [refl, expt] + extra_args)


def test_multi_scale(dials_data, tmp_path):
    """Test standard scaling of two datasets."""

    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    refl_2 = data_dir / "25_integrated.pickle"
    expt_2 = data_dir / "25_integrated_experiments.json"
    extra_args = [
        "unmerged_mtz=unmerged.mtz",
        "error_model=None",
        "intensity_choice=profile",
        "outlier_rejection=simple",
    ]

    run_one_scaling(tmp_path, [refl_1, refl_2, expt_1, expt_2] + extra_args)

    # Now inspect output, check it hasn't changed drastically, or if so verify
    # that the new behaviour is more correct and update test accordingly.
    result = get_merging_stats(tmp_path / "unmerged.mtz")
    expected_nobs = 5526  # 19/06/20
    assert abs(result.overall.n_obs - expected_nobs) < 30
    assert result.overall.r_pim < 0.0221  # at 22/10/18, value was 0.22037
    assert result.overall.cc_one_half > 0.9975  # at 07/08/18, value was 0.99810
    print(result.overall.r_pim)
    print(result.overall.cc_one_half)

    refls = flex.reflection_table.from_file(tmp_path / "scaled.refl")
    n_scaled = refls.get_flags(refls.flags.scaled).count(True)
    assert n_scaled == result.overall.n_obs
    assert n_scaled == refls.get_flags(refls.flags.bad_for_scaling, all=False).count(
        False
    )
    assert len(set(refls["id"])) == 2
    assert len(set(refls["imageset_id"])) == 2
    for id_ in range(2):
        sel = refls["id"] == id_
        assert set(refls["imageset_id"].select(sel)) == {id_}

    # run again, optimising errors, and continuing from where last run left off.
    extra_args = [
        "error_model=basic",
        "unmerged_mtz=unmerged.mtz",
        "check_consistent_indexing=True",
    ]
    run_one_scaling(tmp_path, ["scaled.refl", "scaled.expt"] + extra_args)
    # Now inspect output, check it hasn't changed drastically, or if so verify
    # that the new behaviour is more correct and update test accordingly.
    # Note: error optimisation currently appears to give worse results here!
    result = get_merging_stats(tmp_path / "unmerged.mtz")
    expected_nobs = 5560  # 22/06/20
    print(result.overall.r_pim)
    print(result.overall.cc_one_half)
    assert abs(result.overall.n_obs - expected_nobs) < 100
    assert result.overall.r_pim < 0.016  # at #22/06/20, value was 0.015
    assert result.overall.cc_one_half > 0.997  # at #22/06/20, value was 0.999


def test_multi_scale_individual_error_models(dials_data, tmp_path):
    """Test standard scaling of two datasets."""

    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    refl_2 = data_dir / "25_integrated.pickle"
    expt_2 = data_dir / "25_integrated_experiments.json"
    extra_args = [
        "unmerged_mtz=unmerged.mtz",
        "error_model.grouping=individual",
        "intensity_choice=profile",
    ]

    run_one_scaling(tmp_path, [refl_1, refl_2, expt_1, expt_2] + extra_args)

    # Now inspect output, check it hasn't changed drastically, or if so verify
    # that the new behaviour is more correct and update test accordingly.
    result = get_merging_stats(tmp_path / "unmerged.mtz")
    expected_nobs = 5358  # 04/05/21
    assert abs(result.overall.n_obs - expected_nobs) < 30
    assert result.overall.r_pim < 0.015  # at 04/05/21, value was 0.013
    assert result.overall.cc_one_half > 0.9975  # at 04/05/21, value was 0.999

    # Test that different error models were determined for each sweep.
    scaling_models = load.experiment_list(
        tmp_path / "scaled.expt", check_format=False
    ).scaling_models()
    params_1 = scaling_models[0].configdict["error_model_parameters"]
    params_2 = scaling_models[1].configdict["error_model_parameters"]
    assert params_1[0] != params_2[0]
    assert params_1[1] != params_2[1]


def test_multi_scale_exclude_images(dials_data, tmp_path):
    """Test scaling of multiple dataset with image exclusion."""
    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    refl_2 = data_dir / "25_integrated.pickle"
    expt_2 = data_dir / "25_integrated_experiments.json"
    # Expect this dataset to be given batches 1-1800 and 1901-3600
    # Try excluding last two hundred batches
    extra_args = [
        "error_model=None",
        "intensity_choice=profile",
        "outlier_rejection=simple",
        "exclude_images=0:1601:1800",
        "exclude_images=1:1501:1700",
    ]

    run_one_scaling(tmp_path, [refl_1, refl_2, expt_1, expt_2] + extra_args)

    refls = flex.reflection_table.from_file(tmp_path / "scaled.refl")
    d1 = refls.select(refls["id"] == 0)
    d2 = refls.select(refls["id"] == 1)
    nd1_scaled = d1.get_flags(d1.flags.scaled).count(True)
    # full sweep would have 2312, expect ~2060
    assert nd1_scaled < 2100
    assert nd1_scaled > 2000
    nd2_scaled = d2.get_flags(d2.flags.scaled).count(True)
    # full sweep would have 3210
    assert nd2_scaled < 2900
    assert nd2_scaled > 2800

    scaling_models = load.experiment_list(
        tmp_path / "scaled.expt", check_format=False
    ).scaling_models()
    assert scaling_models[0].configdict["valid_image_range"] == [1, 1600]
    assert scaling_models[1].configdict["valid_image_range"] == [1, 1500]
    assert [0, 160.0] == pytest.approx(scaling_models[0].configdict["valid_osc_range"])
    assert [-145.0, 5.0] == pytest.approx(
        scaling_models[1].configdict["valid_osc_range"]
    )

    # Run again, excluding some more from one run.
    extra_args = [
        "error_model=None",
        "intensity_choice=profile",
        "outlier_rejection=simple",
        "exclude_images=0:1401:1600",
    ]
    run_one_scaling(tmp_path, ["scaled.refl", "scaled.expt"] + extra_args)
    scaling_models = load.experiment_list(
        tmp_path / "scaled.expt", check_format=False
    ).scaling_models()
    assert scaling_models[0].configdict["valid_image_range"] == [1, 1400]
    assert scaling_models[1].configdict["valid_image_range"] == [1, 1500]
    assert [0, 140.0] == pytest.approx(scaling_models[0].configdict["valid_osc_range"])
    assert [-145.0, 5.0] == pytest.approx(
        scaling_models[1].configdict["valid_osc_range"]
    )

    refls = flex.reflection_table.from_file(tmp_path / "scaled.refl")
    d1 = refls.select(refls["id"] == 0)
    d2 = refls.select(refls["id"] == 1)
    nd1_scaled = d1.get_flags(d1.flags.scaled).count(True)
    # full sweep would have 2312, expect 1800
    assert nd1_scaled < 1850
    assert nd1_scaled > 1750
    nd2_scaled = d2.get_flags(d2.flags.scaled).count(True)
    # full sweep would have 3210, expect ~2850
    assert nd2_scaled < 2900
    assert nd2_scaled > 2800


def test_scale_handle_bad_dataset(dials_data, tmp_path):
    """Set command line parameters such that one dataset does not meet the
    criteria for inclusion in scaling. Check that this is excluded and the
    scaling job completes without failure."""
    location = dials_data("multi_crystal_proteinase_k", pathlib=True)
    command = [
        "dials.scale",
        "reflection_selection.method=intensity_ranges",
        "Isigma_range=90.0,1000",
    ]
    for i in range(1, 6):
        command.append(location / f"experiments_{i}.json")
        command.append(location / f"reflections_{i}.pickle")

    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr

    reflections = flex.reflection_table.from_file(tmp_path / "scaled.refl")
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert len(expts) == 4
    assert len(reflections.experiment_identifiers()) == 4


def test_target_scale_handle_bad_dataset(dials_data, tmp_path):
    """Set command line parameters such that some datasets do not meet the
    criteria for inclusion in scaling. Check that these are excluded and the
    scaling job completes without failure."""
    location = dials_data("cunir_serial_processed", pathlib=True)
    pdb = dials_data("cunir_serial", pathlib=True) / "2BW4.pdb"
    command = [
        "dials.scale",
        "reflection_selection.method=intensity_ranges",
        "Isigma_range=20.0,0.0",
        "full_matrix=None",
        os.fspath(location / "integrated.refl"),
        os.fspath(location / "integrated.expt"),
        f"reference={os.fspath(pdb)}",
    ]

    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr

    reflections = flex.reflection_table.from_file(tmp_path / "scaled.refl")
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert len(expts) == 3
    assert len(reflections.experiment_identifiers()) == 3


def test_targeted_scaling(dials_data, tmp_path):
    """Test the targeted scaling workflow."""
    location = dials_data("l_cysteine_4_sweeps_scaled", pathlib=True)
    target_refl = location / "scaled_35.refl"
    target_expt = location / "scaled_35.expt"

    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    refl_2 = data_dir / "25_integrated.pickle"
    expt_2 = data_dir / "25_integrated_experiments.json"

    # Do targeted scaling, use this as a chance to test the KB model as well.
    extra_args = ["model=KB"]
    run_one_scaling(tmp_path, [target_refl, refl_1, target_expt, expt_1] + extra_args)

    scaled_exp = tmp_path / "scaled.expt"
    scaled_refl = tmp_path / "scaled.refl"
    experiments_list = load.experiment_list(scaled_exp, check_format=False)
    assert len(experiments_list.scaling_models()) == 2
    assert experiments_list.scaling_models()[0].id_ == "physical"
    assert experiments_list.scaling_models()[1].id_ == "KB"

    extra_args = ["model=KB", "only_target=True"]
    run_one_scaling(tmp_path, [refl_2, scaled_refl, expt_2, scaled_exp] + extra_args)
    experiments_list = load.experiment_list(
        tmp_path / "scaled.expt", check_format=False
    )
    assert len(experiments_list.scaling_models()) == 1
    assert experiments_list.scaling_models()[0].id_ == "KB"


def test_shared_absorption_surface(dials_data, tmp_path):
    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    refl_2 = data_dir / "25_integrated.pickle"
    expt_2 = data_dir / "25_integrated_experiments.json"

    # Do targeted scaling, use this as a chance to test the KB model as well.
    extra_args = ["share.absorption=True"]
    run_one_scaling(tmp_path, [refl_1, expt_1, refl_2, expt_2] + extra_args)

    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert (
        expts.scaling_models()[0].components["absorption"].parameters
        == expts.scaling_models()[1].components["absorption"].parameters
    )
    assert (
        expts.scaling_models()[0].components["absorption"].parameter_esds
        == expts.scaling_models()[1].components["absorption"].parameter_esds
    )


def test_incremental_scale_workflow(dials_data, tmp_path):
    """Try scale, cosym, scale, cosym, scale."""
    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"

    run_one_scaling(tmp_path, [refl_1, expt_1])

    # test order also - first new file before scaled
    refl_2 = data_dir / "25_integrated.pickle"
    expt_2 = data_dir / "25_integrated_experiments.json"
    command = ["dials.cosym", refl_2, expt_2, "scaled.refl", "scaled.expt"]

    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "symmetrized.expt").is_file()
    assert (tmp_path / "symmetrized.refl").is_file()

    # now try scaling again to check everything is okay
    run_one_scaling(tmp_path, ["symmetrized.refl", "symmetrized.expt"])

    # test order also - first scaled file then new file
    refl_2 = data_dir / "30_integrated.pickle"
    expt_2 = data_dir / "30_integrated_experiments.json"
    command = [
        "dials.cosym",
        "scaled.refl",
        "scaled.expt",
        refl_1,
        expt_2,
        "output.reflections=symmetrized.refl",
        "output.experiments=symmetrized.expt",
    ]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "symmetrized.expt").is_file()
    assert (tmp_path / "symmetrized.refl").is_file()

    # now try scaling again to check everything is okay
    run_one_scaling(tmp_path, ["symmetrized.refl", "symmetrized.expt"])


@pytest.mark.parametrize(
    ("mode", "parameter", "parameter_values"),
    [
        ("single", None, None),
        ("multi", "physical.absorption_correction", None),
        ("multi", "model", "physical array"),
    ],
)
def test_scale_cross_validate(dials_data, tmp_path, mode, parameter, parameter_values):
    """Test standard scaling of one dataset."""
    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    refl = data_dir / "20_integrated.pickle"
    expt = data_dir / "20_integrated_experiments.json"
    extra_args = [
        f"cross_validation_mode={mode}",
        "nfolds=2",
        "full_matrix=0",
        "error_model=None",
    ]
    if parameter:
        extra_args += [f"parameter={parameter}"]
    if parameter_values:
        extra_args += [f"parameter_values={parameter_values}"]
    command = ["dials.scale", refl, expt] + extra_args
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr


def test_few_reflections(dials_data):
    """
    Test that dials.symmetry does something sensible if given few reflections.

    Use some example integrated data generated from two ten-image 1° sweeps.  These
    each contain a few dozen integrated reflections.

    Also test the behaviour of dials.merge and dials.report on the output.

    By suppressing the output from dials.scale and dials.report, we obviate the need to
    run in a temporary directory.

    Args:
        dials_data: DIALS custom Pytest fixture for access to test data.
    """
    # Get the input experiment lists and reflection tables.
    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    experiments = ExperimentList.from_file(data_dir / "11_integrated.expt")
    experiments.extend(ExperimentList.from_file(data_dir / "23_integrated.expt"))
    refls = "11_integrated.refl", "23_integrated.refl"
    reflections = [flex.reflection_table.from_file(data_dir / refl) for refl in refls]

    # Get and use the default parameters for dials.scale, suppressing HTML output.
    scale_params = scale.phil_scope.fetch(
        source=phil.parse("output.html=None")
    ).extract()
    # Does what it says on the tin.  Run scaling.
    scaled_expt, scaled_refl = scale.run_scaling(scale_params, experiments, reflections)

    # Get and use the default parameters for dials.merge.
    merge_params = merge.phil_scope.fetch(source=phil.parse("")).extract()
    # Run dials.merge on the scaling output.
    merge.merge_data_to_mtz(merge_params, scaled_expt, [scaled_refl])

    # Get and use the default parameters for dials.report, suppressing HTML output.
    report_params = report.phil_scope.fetch(
        source=phil.parse("output.html=None")
    ).extract()
    # Get an Analyser object, which does the dials.report stuff.
    analyse = report.Analyser(
        report_params,
        grid_size=report_params.grid_size,
        centroid_diff_max=report_params.centroid_diff_max,
    )
    # Run dials.report on scaling output.
    analyse(scaled_refl, scaled_expt)
