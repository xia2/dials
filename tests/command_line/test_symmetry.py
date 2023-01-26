from __future__ import annotations

import json
import math
import shutil
import subprocess

import procrunner
import pytest

import scitbx.matrix
from cctbx import sgtbx, uctbx
from dxtbx.model import Crystal, Experiment, ExperimentList, Scan
from dxtbx.serialize import load

from dials.algorithms.symmetry.cosym._generate_test_data import (
    generate_experiments_reflections,
)
from dials.array_family import flex
from dials.command_line import symmetry
from dials.command_line.symmetry import (
    apply_change_of_basis_ops,
    change_of_basis_ops_to_minimum_cell,
    eliminate_sys_absent,
    get_subset_for_symmetry,
    median_unit_cell,
)
from dials.util.exclude_images import exclude_image_ranges_from_scans
from dials.util.multi_dataset_handling import assign_unique_identifiers
from dials.util.phil import parse


def test_symmetry_laue_only(dials_data, tmp_path):
    """Simple test to check that dials.symmetry completes"""

    lcyst_data = dials_data("l_cysteine_dials_output", pathlib=True)
    result = procrunner.run(
        [
            "dials.symmetry",
            lcyst_data / "20_integrated_experiments.json",
            lcyst_data / "20_integrated.pickle",
            lcyst_data / "25_integrated_experiments.json",
            lcyst_data / "25_integrated.pickle",
            "systematic_absences.check=False",
        ],
        working_directory=tmp_path,
    )
    assert not result.returncode and not result.stderr
    assert tmp_path.joinpath("symmetrized.refl").is_file()
    assert tmp_path.joinpath("symmetrized.expt").is_file()
    exps = load.experiment_list(tmp_path / "symmetrized.expt", check_format=False)
    assert str(exps[0].crystal.get_space_group().info()) == "P 2 2 2"
    joint_reflections = flex.reflection_table.from_file(tmp_path / "symmetrized.refl")
    # check that there are 2 unique id and imageset_ids, and that these
    # correctly correspond to each experiment
    assert len(set(joint_reflections["id"])) == 2
    assert len(set(joint_reflections["imageset_id"])) == 2
    for id_ in range(2):
        sel = joint_reflections["id"] == id_
        assert set(joint_reflections["imageset_id"].select(sel)) == {id_}


def test_symmetry_basis_changes_for_C2(tmp_path):
    """Test the correctness of change of basis operations in dials.symmetry

    Supply the unit cell of beta-lactamase, which triggers a change of
    basis from input to minimum during symmetry analysis."""
    unit_cell = (53.173, 61.245, 69.292, 90.0, 93.04675, 90.0)
    space_group = sgtbx.space_group_info("C 2").group()
    experiments, reflections, _ = generate_experiments_reflections(
        space_group=space_group,
        unit_cell=unit_cell,
        sample_size=1,
        map_to_minimum=False,
    )
    experiments.as_json(tmp_path / "tmp.expt")
    joint_table = flex.reflection_table()
    for r in reflections:
        joint_table.extend(r)
    joint_table.as_file(tmp_path / "tmp.refl")

    result = procrunner.run(
        [
            "dials.symmetry",
            tmp_path / "tmp.expt",
            tmp_path / "tmp.refl",
            "json=symmetry.json",
        ],
        working_directory=tmp_path,
    )
    assert not result.returncode and not result.stderr
    assert tmp_path.joinpath("symmetrized.refl").is_file()
    symmetrized_expt_file = tmp_path / "symmetrized.expt"
    assert symmetrized_expt_file.is_file()

    expts = load.experiment_list(symmetrized_expt_file, check_format=False)
    for v, expected in zip(expts[0].crystal.get_unit_cell().parameters(), unit_cell):
        assert v == pytest.approx(expected)

    # Using the change of basis ops from the json output we should be able to
    # reindex the input experiments to match the output experiments
    with tmp_path.joinpath("symmetry.json").open() as f:
        d = json.load(f)
    cs = experiments[0].crystal.get_crystal_symmetry()
    cb_op_inp_min = sgtbx.change_of_basis_op(str(d["cb_op_inp_min"][0]))
    cb_op_min_best = sgtbx.change_of_basis_op(str(d["subgroup_scores"][0]["cb_op"]))
    assert cs.change_basis(cb_op_min_best * cb_op_inp_min).is_similar_symmetry(
        expts[0].crystal.get_crystal_symmetry()
    )


@pytest.mark.parametrize("option", ["", "exclude_images=0:1500:1800"])
def test_symmetry_with_absences(dials_data, tmpdir, option):
    """Simple test to check that dials.symmetry, with absences, completes"""

    lcyst_data = dials_data("l_cysteine_dials_output", pathlib=True)
    cmd = [
        "dials.symmetry",
        lcyst_data / "20_integrated_experiments.json",
        lcyst_data / "20_integrated.pickle",
        lcyst_data / "25_integrated_experiments.json",
        lcyst_data / "25_integrated.pickle",
    ]
    if option:
        cmd.append(option)

    result = procrunner.run(cmd, working_directory=tmpdir)
    assert not result.returncode and not result.stderr
    assert tmpdir.join("symmetrized.refl").check()
    assert tmpdir.join("symmetrized.expt").check()
    expts = load.experiment_list(
        tmpdir.join("symmetrized.expt").strpath, check_format=False
    )
    assert str(expts[0].crystal.get_space_group().info()) == "P 21 21 21"


def test_symmetry_with_laue_group_override(dials_data, tmp_path):
    """Simple test to check that dials.symmetry, with overridden laue group, completes"""

    lcyst_data = dials_data("l_cysteine_dials_output", pathlib=True)
    result = procrunner.run(
        [
            "dials.symmetry",
            "laue_group=P121",
            "change_of_basis_op=-b,-a,-c",
            lcyst_data / "20_integrated_experiments.json",
            lcyst_data / "20_integrated.pickle",
            lcyst_data / "25_integrated_experiments.json",
            lcyst_data / "25_integrated.pickle",
        ],
        working_directory=tmp_path,
    )
    assert not result.returncode and not result.stderr
    assert tmp_path.joinpath("symmetrized.refl").is_file()
    assert tmp_path.joinpath("symmetrized.expt").is_file()
    expts = load.experiment_list(tmp_path / "symmetrized.expt", check_format=False)
    assert str(expts[0].crystal.get_space_group().info()) == "P 1 21 1"
    # Verify that the unit cell has been reindexed correctly
    assert expts[0].crystal.get_unit_cell().parameters() == pytest.approx(
        (8.21578444269, 5.4815363434, 12.1457047712, 90.0, 90.0, 90.0)
    )


@pytest.mark.parametrize("method", ["direct", "fourier"])
def test_symmetry_absences_only(dials_data, tmp_path, method):
    """Test the command line script with real data. Proteinase K in P41"""
    location = dials_data("vmxi_proteinase_k_sweeps", pathlib=True)

    command = [
        "dials.symmetry",
        "laue_group=None",
        location / "experiments_0.json",
        location / "reflections_0.pickle",
        f"method={method}",
    ]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert tmp_path.joinpath("dials.symmetry.html").is_file()
    assert tmp_path.joinpath("symmetrized.expt").is_file()
    exps = load.experiment_list(tmp_path / "symmetrized.expt", check_format=False)
    assert str(exps[0].crystal.get_space_group().info()) == "P 41"

    # Now try with a d_min
    command += ["d_min=4.0"]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    exps = load.experiment_list(tmp_path / "symmetrized.expt", check_format=False)
    assert str(exps[0].crystal.get_space_group().info()) == "P 41"


def test_map_to_minimum_cell():
    # Input and expected output
    input_ucs = [
        (39.7413, 183.767, 140.649, 90, 90, 90),
        (40.16, 142.899, 92.4167, 90, 102.48, 90),
        (180.613, 40.1558, 142.737, 90, 90.0174, 90),
    ]
    input_sgs = ["C 2 2 21", "P 1 2 1", "C 1 2 1"]
    input_hkl = [
        [(1, -75, -71), (1, -73, -70), (1, -71, -69)],
        [(14, -37, -36), (-2, -35, -46), (-3, -34, -47)],
        [(-31, -5, -3), (-25, -3, -3), (-42, -8, -2)],
    ]
    expected_ucs = [
        (39.7413, 94.00755450320204, 140.649, 90.0, 90.0, 77.79717980856927),
        (40.16, 92.46399390642911, 142.899, 90.0, 90.0, 77.3882749092846),
        (
            40.1558,
            92.51154528306184,
            142.73699999999997,
            89.9830147351441,
            90.0,
            77.46527404307477,
        ),
    ]
    expected_output_hkl = [
        [(-1, 37, -71), (-1, 36, -70), (-1, 35, -69)],
        [(-14, 22, 37), (2, 48, 35), (3, 50, 34)],
        [(-5, 13, -3), (-3, 11, -3), (-8, 17, -2)],
    ]

    # Setup the input experiments and reflection tables
    expts = ExperimentList()
    reflections = []
    for uc, sg, hkl in zip(input_ucs, input_sgs, input_hkl):
        uc = uctbx.unit_cell(uc)
        sg = sgtbx.space_group_info(sg).group()
        B = scitbx.matrix.sqr(uc.fractionalization_matrix()).transpose()
        expts.append(Experiment(crystal=Crystal(B, space_group=sg, reciprocal=True)))
        refl = flex.reflection_table()
        refl["miller_index"] = flex.miller_index(hkl)
        reflections.append(refl)

    # Actually run the method we are testing
    cb_ops = change_of_basis_ops_to_minimum_cell(
        expts, max_delta=5, relative_length_tolerance=0.05, absolute_angle_tolerance=2
    )
    cb_ops_as_xyz = [cb_op.as_xyz() for cb_op in cb_ops]
    # Actual cb_ops are machine dependent (sigh)
    assert cb_ops_as_xyz == [
        "-x+y,-2*y,z",
        "-x+z,-z,-y",
        "x+y,-2*x,z",
    ] or cb_ops_as_xyz == ["x-y,2*y,z", "x-z,z,-y", "-x-y,2*x,z"]

    expts_min, reflections = apply_change_of_basis_ops(expts, reflections, cb_ops)
    # Verify that the unit cells have been transformed as expected
    for expt, uc in zip(expts, expected_ucs):
        assert expt.crystal.get_unit_cell().parameters() == pytest.approx(uc, abs=4e-2)

    # Space group should be set to P1
    assert [expt.crystal.get_space_group().type().number() for expt in expts_min] == [
        1,
        1,
        1,
    ]

    # Verify that the reflections have been reindexed as expected
    # Because the exact choice of minimum cell can be platform-dependent,
    # compare the magnitude, but not the sign of the output hkl values
    for refl, expected_hkl in zip(reflections, expected_output_hkl):
        for hkl, e_hkl in zip(refl["miller_index"], expected_hkl):
            assert [abs(h) for h in hkl] == [abs(eh) for eh in e_hkl]


def _make_input_for_exclude_tests(exclude_images=True):
    """Generate input data, that upon exclusion should leave only the first
    reflection."""
    if exclude_images:
        exclude_images = [["0:360:720"], ["1:360:720"]]
    expt1 = Experiment(scan=Scan(image_range=(0, 720), oscillation=(0.0, 1.0)))
    expt2 = Experiment(scan=Scan(image_range=(0, 720), oscillation=(0.0, -1.0)))
    refls1 = flex.reflection_table()
    refls2 = flex.reflection_table()
    refls1["xyzobs.mm.value"] = flex.vec3_double(
        [(0.0, 0.0, 10.0 * math.pi / 180.0), (0.0, 0.0, 370.0 * math.pi / 180.0)]
    )
    refls1["xyzobs.px.value"] = flex.vec3_double([(0.0, 0.0, 10.0), (0.0, 0.0, 370.0)])
    refls1["i"] = flex.int([0, 1])
    refls2["xyzobs.mm.value"] = flex.vec3_double(
        [(0.0, 0.0, -10.0 * math.pi / 180.0), (0.0, 0.0, -370.0 * math.pi / 180.0)]
    )
    refls2["xyzobs.px.value"] = flex.vec3_double([(0.0, 0.0, 10.0), (0.0, 0.0, 370.0)])
    refls2["i"] = flex.int([0, 1])
    expts = ExperimentList([expt1, expt2])
    tables = [refls1, refls2]
    expts, tables = assign_unique_identifiers(expts, tables)
    return exclude_images, expts, tables


def test_get_subset_for_symmetry_image_range():
    """Test that first 360 degrees are selected from each sweep with an exclude
    images command."""
    exclude_images, expts, tables = _make_input_for_exclude_tests(exclude_images=True)
    refls = get_subset_for_symmetry(expts, tables, exclude_images)
    assert refls[0]["i"].all_eq(0)
    assert refls[1]["i"].all_eq(0)


def test_get_subset_for_symmetry_prior_image_range():
    """Test that first 360 degrees are selected from each sweep with an exclude
    images command."""
    exclude_images, expts, tables = _make_input_for_exclude_tests(exclude_images=True)

    # Explicitly exclude a different image range
    expts = exclude_image_ranges_from_scans(tables, expts, [["0:1:360"], ["1:1:360"]])
    refls = get_subset_for_symmetry(expts, tables)
    assert refls[0]["i"].all_eq(1)
    assert refls[1]["i"].all_eq(1)

    # Exclude_images should be cumulative, i.e. the range exclude above should
    # be excluded in addition to the new range provided explicitly to the
    # function
    refls = get_subset_for_symmetry(expts, tables, exclude_images)
    assert len(refls[0]) == 0
    assert len(refls[1]) == 0


def test_get_subset_for_symmetry_first_360():
    """Test that first 360 degrees are selected from each sweep"""
    exclude_images, expts, tables = _make_input_for_exclude_tests(exclude_images=False)
    refls = get_subset_for_symmetry(expts, tables, exclude_images)
    assert refls[0]["i"].all_eq(0)
    assert refls[1]["i"].all_eq(0)


def test_change_of_basis_ops_to_minimum_cell_1037(mocker):
    # See https://github.com/dials/dials/issues/1037

    input_ucs = [
        (
            4.805202948916906,
            12.808064769657364,
            16.544899201125446,
            106.45808502003258,
            90.0065567098825,
            100.77735674275475,
        ),
        (
            4.808011343212577,
            12.821894835790472,
            16.557339561965573,
            106.48431244651402,
            90.0252848479048,
            100.77252933676507,
        ),
        (
            4.8096632137789985,
            12.815648858527567,
            16.55931712239122,
            106.48990701341536,
            90.01703141314147,
            100.80397887485773,
        ),
        (
            4.807294085194974,
            12.822386757910516,
            16.560411742466663,
            106.43185845358086,
            90.02067929544215,
            100.79522302759383,
        ),
    ]

    # Setup the input experiments and reflection tables
    expts = ExperimentList()
    for uc in input_ucs:
        uc = uctbx.unit_cell(uc)
        sg = sgtbx.space_group_info("P1").group()
        B = scitbx.matrix.sqr(uc.fractionalization_matrix()).transpose()
        expts.append(Experiment(crystal=Crystal(B, space_group=sg, reciprocal=True)))

    # We want to spy on the return value of this function
    mocker.spy(symmetry, "unit_cells_are_similar_to")

    # Actually run the method we are testing
    cb_ops = change_of_basis_ops_to_minimum_cell(
        expts, max_delta=5, relative_length_tolerance=0.05, absolute_angle_tolerance=2
    )
    import pytest_mock

    if getattr(pytest_mock, "version", "").startswith("1."):
        assert symmetry.unit_cells_are_similar_to.return_value is True
    else:
        assert symmetry.unit_cells_are_similar_to.spy_return is True
    cb_ops_as_xyz = [cb_op.as_xyz() for cb_op in cb_ops]
    assert len(set(cb_ops_as_xyz)) == 1
    # Actual cb_ops are machine dependent (sigh)
    assert cb_ops_as_xyz[0] in ("x,y,z", "-x,y,-z", "x-y,-y,-z")


def test_change_of_basis_ops_to_minimum_cell_mpro():
    input_ucs = [
        (46.023, 55.001, 64.452, 64.744, 78.659, 89.824),
        (44.747, 53.916, 62.554, 114.985, 99.610, 90.736),
    ]

    # Setup the input experiments and reflection tables
    expts = ExperimentList()
    for uc in input_ucs:
        uc = uctbx.unit_cell(uc)
        sg = sgtbx.space_group_info("P1").group()
        B = scitbx.matrix.sqr(uc.fractionalization_matrix()).transpose()
        expts.append(Experiment(crystal=Crystal(B, space_group=sg, reciprocal=True)))

    # Actually run the method we are testing
    cb_ops = change_of_basis_ops_to_minimum_cell(
        expts, max_delta=5, relative_length_tolerance=0.05, absolute_angle_tolerance=2
    )
    expts.change_basis(cb_ops, in_place=True)
    assert symmetry.unit_cells_are_similar_to(
        expts,
        median_unit_cell(expts),
        relative_length_tolerance=0.05,
        absolute_angle_tolerance=2,
    )


from cctbx import crystal


def test_change_of_basis_ops_to_minimum_cell_with_outlier():
    symmetries = [
        crystal.symmetry(unit_cell=uc, space_group="P1")
        for uc in (
            (52.8868, 52.8868, 333.522, 90, 90, 120),
            (52.6503, 53.0292, 333.783, 89.9872, 89.2247, 60.8078),
            (52.9571, 53.0005, 334.255, 90.0493, 90.0042, 119.893),
            (
                54.4465,
                56.5677,
                355.775,
                93.4376,
                90.0999,
                118.256,
            ),  # This is an outlier
            (52.9235, 52.9235, 335.296, 90, 90, 120),
            (53.4531, 53.4531, 322.909, 90, 90, 120),
        )
    ]

    # Setup the input experiments and reflection tables
    expts = ExperimentList()
    for cs in symmetries:
        B = scitbx.matrix.sqr(cs.unit_cell().fractionalization_matrix()).transpose()
        expts.append(
            Experiment(
                crystal=Crystal(B, space_group=cs.space_group(), reciprocal=True)
            )
        )

    # Actually run the method we are testing
    cb_ops = change_of_basis_ops_to_minimum_cell(
        expts, max_delta=5, relative_length_tolerance=0.05, absolute_angle_tolerance=2
    )
    assert cb_ops.count(None) == 1
    assert cb_ops[3] is None
    expts = ExperimentList([expt for expt, cb_op in zip(expts, cb_ops) if cb_op])
    cb_ops = [cb_op for cb_op in cb_ops if cb_op]

    expts.change_basis(cb_ops, in_place=True)
    assert symmetry.unit_cells_are_similar_to(
        expts,
        median_unit_cell(expts),
        relative_length_tolerance=0.05,
        absolute_angle_tolerance=2,
    )


def test_median_cell():
    unit_cells = [
        uctbx.unit_cell(uc)
        for uc in [
            (10, 11, 11.9, 90, 85, 90),
            (10.1, 11.2, 12, 90, 85.5, 90),
            (10.2, 11.1, 12, 90, 84.7, 90),
        ]
    ]
    expts = ExperimentList()
    for uc in unit_cells:
        sg = sgtbx.space_group_info("P1").group()
        B = scitbx.matrix.sqr(uc.fractionalization_matrix()).transpose()
        expts.append(Experiment(crystal=Crystal(B, space_group=sg, reciprocal=True)))

    median = median_unit_cell(expts)
    assert median.parameters() == pytest.approx((10.1, 11.1, 12, 90, 85, 90))


def test_eliminate_sys_absent():
    refl = flex.reflection_table()
    refl["miller_index"] = flex.miller_index(
        [(-31, -5, -3), (-25, -3, -3), (0, 1, 0), (-42, -8, -2)]
    )
    sgi = sgtbx.space_group_info("C121")
    uc = sgi.any_compatible_unit_cell(volume=1000)
    B = scitbx.matrix.sqr(uc.fractionalization_matrix()).transpose()
    expt = Experiment(crystal=Crystal(B, space_group=sgi.group(), reciprocal=True))
    reflections = eliminate_sys_absent([expt], [refl])
    assert list(reflections[0]["miller_index"]) == [
        (-31, -5, -3),
        (-25, -3, -3),
        (-42, -8, -2),
    ]


def test_few_reflections(dials_data, run_in_tmp_path):
    """
    Test that dials.symmetry does something sensible if given few reflections.

    Use some example integrated data generated from a ten-image 1° sweep.  These
    contain a few dozen integrated reflections.

    Args:
        dials_data: DIALS custom Pytest fixture for access to test data.
        run_in_tmpdir: DIALS custom Pytest fixture to run this test in a temporary
                       directory.
    """
    # Get and use the default parameters for dials.symmetry.
    params = symmetry.phil_scope.fetch(source=parse("")).extract()

    # Use the integrated data from the first ten images of the first sweep.
    data_dir = dials_data("l_cysteine_dials_output", pathlib=True)
    experiments = ExperimentList.from_file(data_dir / "11_integrated.expt")
    reflections = [flex.reflection_table.from_file(data_dir / "11_integrated.refl")]

    # Run dials.symmetry on the above data files.
    symmetry.symmetry(experiments, reflections, params)


def test_x4wide(dials_data, tmp_path):
    """Verify reflections are reindexed before systematic absence analysis.

    Exercise a bug fix in https://github.com/dials/dials/pull/2183

    Expected space group is P 43 21 2 (or its enantiomorphic equivalent, P 41 21 2)
    See also https://www.rcsb.org/structure/3QF8
    """
    x4wide = dials_data("x4wide_processed", pathlib=True)
    result = subprocess.run(
        [
            shutil.which("dials.symmetry"),
            x4wide / "AUTOMATIC_DEFAULT_scaled.expt",
            x4wide / "AUTOMATIC_DEFAULT_scaled.refl",
        ],
        cwd=tmp_path,
    )
    assert not result.returncode and not result.stderr
    assert tmp_path.joinpath("symmetrized.refl").is_file()
    assert tmp_path.joinpath("symmetrized.expt").is_file()
    exps = load.experiment_list(tmp_path / "symmetrized.expt", check_format=False)
    assert str(exps[0].crystal.get_space_group().info()) == "P 41 21 2"
