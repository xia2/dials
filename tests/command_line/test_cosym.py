from __future__ import annotations

import os
import pathlib

import pytest

from cctbx import sgtbx, uctbx
from dxtbx.serialize import load
from dxtbx.util import ersatz_uuid4

import dials.command_line.cosym as dials_cosym
from dials.algorithms.symmetry.cosym._generate_test_data import (
    generate_experiments_reflections,
)
from dials.array_family import flex
from dials.util import Sorry


@pytest.mark.parametrize(
    "space_group,engine,weights,cc_weights",
    [
        (None, "scitbx", "count", None),
        ("P 1", "scipy", None, None),
        ("P 4", "scipy", "standard_error", "sigma"),
    ],
)
def test_cosym(dials_data, run_in_tmp_path, space_group, engine, weights, cc_weights):
    mcp = dials_data("multi_crystal_proteinase_k", pathlib=True)
    args = [
        "space_group=" + str(space_group),
        "seed=0",
        "nproc=1",
        f"engine={engine}",
        f"weights={weights}",
        f"cc_weights={cc_weights}",
    ]
    for i in [1, 2, 3, 4, 5, 7, 8, 10]:
        args.append(os.fspath(mcp / f"experiments_{i}.json"))
        args.append(os.fspath(mcp / f"reflections_{i}.pickle"))
    dials_cosym.run(args=args)
    assert pathlib.Path("symmetrized.refl").is_file()
    assert pathlib.Path("symmetrized.expt").is_file()
    experiments = load.experiment_list("symmetrized.expt", check_format=False)
    if space_group is None:
        assert (
            experiments[0].crystal.get_space_group().type().lookup_symbol() == "P 4 2 2"
        )
    else:
        assert (
            experiments[0].crystal.get_space_group().type().lookup_symbol()
            == space_group
        )
    joint_reflections = flex.reflection_table.from_file("symmetrized.refl")
    # check that there are 8 unique id and imageset_ids, and that these
    # correctly correspond to each experiment
    assert len(set(joint_reflections["id"])) == 8
    assert len(set(joint_reflections["imageset_id"])) == 8
    for id_ in range(8):
        sel = joint_reflections["id"] == id_
        assert set(joint_reflections["imageset_id"].select(sel)) == {id_}


def test_cosym_partial_dataset(dials_data, run_in_tmp_path):
    """Test how cosym handles partial/bad datasets."""
    mcp = dials_data("multi_crystal_proteinase_k", pathlib=True)
    args = []
    for i in [1, 2]:
        args.append(os.fspath(mcp / f"experiments_{i}.json"))
        args.append(os.fspath(mcp / f"reflections_{i}.pickle"))
    # Make one dataset that will be removed in prefiltering
    r = flex.reflection_table.from_file(mcp / "reflections_8.pickle")
    r["partiality"] = flex.double(r.size(), 0.1)
    r.as_file("renamed.refl")
    args.append("renamed.refl")
    args.append(os.fspath(mcp / "experiments_8.json"))
    # Add another good dataset at the end of the input list
    args.append(os.fspath(mcp / "experiments_10.json"))
    args.append(os.fspath(mcp / "reflections_10.pickle"))
    args.append("threshold=None")

    dials_cosym.run(args=args)
    assert pathlib.Path("symmetrized.refl").is_file()
    assert pathlib.Path("symmetrized.expt").is_file()
    experiments = load.experiment_list("symmetrized.expt", check_format=False)
    assert len(experiments) == 3


def test_cosym_resolution_filter_excluding_datasets(dials_data, run_in_tmp_path):
    mcp = dials_data("multi_crystal_proteinase_k", pathlib=True)
    args = ["space_group=P4", "seed=0", "d_min=10.0", "min_reflections=15"]
    for i in [1, 2, 3, 4, 5, 7, 8, 10]:
        args.append(os.fspath(mcp / f"experiments_{i}.json"))
        args.append(os.fspath(mcp / f"reflections_{i}.pickle"))
    dials_cosym.run(args=args)
    assert pathlib.Path("symmetrized.refl").is_file()
    assert pathlib.Path("symmetrized.expt").is_file()

    joint_reflections = flex.reflection_table.from_file("symmetrized.refl")
    # check that there are 8 unique id and imageset_ids, and that these
    # correctly correspond to each experiment
    assert set(joint_reflections["id"]) == {0, 1, 2, 3, 4, 5, 6}
    assert len(set(joint_reflections["imageset_id"])) == 7
    for id_ in range(7):
        sel = joint_reflections["id"] == id_
        assert set(joint_reflections["imageset_id"].select(sel)) == {id_}


def test_cosym_partial_dataset_raises_sorry(dials_data, run_in_tmp_path, capsys):
    """Test how cosym handles partial/bad datasets."""
    mcp = dials_data("multi_crystal_proteinase_k", pathlib=True)
    args = ["renamed.refl", os.fspath(mcp / "experiments_8.json")]
    r2 = flex.reflection_table.from_file(mcp / "reflections_10.pickle")
    r2["partiality"] = flex.double(r2.size(), 0.1)
    r2.as_file("renamed2.refl")
    args.append("renamed2.refl")
    args.append(os.fspath(mcp / "experiments_10.json"))

    with pytest.raises(Sorry):
        dials_cosym.run(args=args)


def test_cosym_with_reference(dials_data, run_in_tmp_path):
    """Test indexing ambiguity resolution against a reference. Use
    SSX example data as have reference data for these."""
    ssx = dials_data("cunir_serial_processed", pathlib=True)
    refls = ssx / "integrated.refl"
    expts = ssx / "integrated.expt"
    reference = dials_data("cunir_serial", pathlib=True) / "2bw4-sf.cif"

    args = ["d_min=2.0", str(refls), str(expts), f"reference={reference}"]
    dials_cosym.run(args=args)
    assert pathlib.Path("symmetrized.refl").is_file()
    assert pathlib.Path("symmetrized.expt").is_file()
    experiments = load.experiment_list("symmetrized.expt", check_format=False)
    assert len(experiments) == 5  # check we have the right number of expts


def test_synthetic_map_cell_issue(run_in_tmp_path):
    # Test that the program filters out datasets that cannot be mapped to a consistent
    # minimum cell in the change_of_basis_ops_to_minimum_cell function.
    unit_cell = uctbx.unit_cell((5.46, 9.82, 29.58, 95.24, 94.54, 105.21))
    space_group = sgtbx.space_group_info("P1").group()
    experiments, reflections, _ = generate_experiments_reflections(
        space_group=space_group,
        unit_cell=unit_cell,
        unit_cell_volume=10000,
        sample_size=7,
        map_to_p1=True,
        d_min=2.0,
    )
    # Set a distribution of cells that will cause the set of cells to fail the cell
    # similarity test, and also which don't all find the same best C2 cell.
    cells = [
        (5.4, 9.9, 29.9, 95.6, 94.6, 105.1),
        (5.4, 9.6, 29.6, 97.4, 92.9, 103.1),
        (5.4, 9.9, 29.5, 95.1, 96.6, 106.0),
        (5.5, 9.6, 29.4, 94.1, 94.7, 105.7),
        (5.5, 9.8, 29.6, 96.4, 93.0, 105.2),
        (5.5, 10.1, 29.6, 95.2, 94.5, 105.0),
        (5.5, 9.8, 29.7, 95.8, 94.0, 103.9),
    ]
    for expt, cell in zip(experiments, cells):
        expt.crystal.set_unit_cell(uctbx.unit_cell(cell))

    experiments.as_json("tmp.expt")
    expt_file = "tmp.expt"
    joint_table = flex.reflection_table.concat(reflections)
    for id in set(joint_table["id"]):
        joint_table.experiment_identifiers()[id] = ersatz_uuid4()
    joint_table.as_file("tmp.refl")
    refl_file = "tmp.refl"

    args = [
        expt_file,
        refl_file,
        "output.experiments=symmetrized.expt",
        "output.reflections=symmetrized.refl",
        "output.html=cosym.html",
        "output.json=cosym.json",
        "output.excluded=True",
    ]

    dials_cosym.run(args=args)
    assert pathlib.Path("symmetrized.refl").is_file()
    assert pathlib.Path("symmetrized.expt").is_file()
    expts = load.experiment_list("symmetrized.expt", check_format=False)
    excl = load.experiment_list("excluded.expt", check_format=False)
    assert len(expts) == 3
    assert len(excl) == 4

    # Increase the angle tolerance so that the cells are determined as similar
    # and can therefore be correctly mapped to the same cell setting.
    args.append("absolute_angle_tolerance=3.0")
    args.append("excluded_prefix=excluded2")
    dials_cosym.run(args=args)
    assert pathlib.Path("symmetrized.refl").is_file()
    assert pathlib.Path("symmetrized.expt").is_file()
    assert pathlib.Path("cosym.html").is_file()
    assert pathlib.Path("cosym.json").is_file()
    assert not pathlib.Path("excluded2.expt").is_file()  # Nothing excluded this time
    expts = load.experiment_list("symmetrized.expt", check_format=False)
    assert len(expts) == 7


@pytest.mark.parametrize(
    (
        "space_group",
        "unit_cell",
        "dimensions",
        "sample_size",
        "use_known_space_group",
        "use_known_lattice_group",
    ),
    [
        ("P2", None, None, 10, False, False),
        ("P3", None, None, 20, False, False),
        ("I23", None, 2, 10, False, False),
        ("P422", (79, 79, 37, 90, 90, 90), None, 10, True, False),
        ("P321", (59.39, 59.39, 28.35, 90, 90, 120), None, 5, False, False),
        ("C2", (56.194, 53.224, 32.156, 90.000, 92.277, 90.000), None, 5, False, False),
    ],
)
def test_synthetic(
    space_group,
    unit_cell,
    dimensions,
    sample_size,
    use_known_space_group,
    use_known_lattice_group,
    run_in_tmp_path,
):
    space_group = sgtbx.space_group_info(space_group).group()
    if unit_cell is not None:
        unit_cell = uctbx.unit_cell(unit_cell)
    experiments, reflections, _ = generate_experiments_reflections(
        space_group=space_group,
        unit_cell=unit_cell,
        unit_cell_volume=10000,
        sample_size=sample_size,
        map_to_p1=True,
        d_min=1.5,
    )

    experiments.as_json("tmp.expt")
    expt_file = "tmp.expt"
    joint_table = flex.reflection_table()
    for r in reflections:
        joint_table.extend(r)
    for id in set(joint_table["id"]):
        joint_table.experiment_identifiers()[id] = ersatz_uuid4()
    joint_table.as_file("tmp.refl")
    refl_file = "tmp.refl"

    args = [
        expt_file,
        refl_file,
        "output.experiments=symmetrized.expt",
        "output.reflections=symmetrized.refl",
        "output.html=cosym.html",
        "output.json=cosym.json",
    ]

    if use_known_space_group:
        args.append(f"space_group={space_group.info()}")
    if use_known_lattice_group:
        args.append(f"lattice_group={space_group.info()}")
    if dimensions is not None:
        args.append(f"dimensions={dimensions}")

    dials_cosym.run(args=args)
    assert pathlib.Path("symmetrized.refl").is_file()
    assert pathlib.Path("symmetrized.expt").is_file()
    assert pathlib.Path("cosym.html").is_file()
    assert pathlib.Path("cosym.json").is_file()
    cosym_expts = load.experiment_list("symmetrized.expt", check_format=False)
    assert len(cosym_expts) == len(experiments)
    for expt in cosym_expts:
        if unit_cell is not None:
            assert expt.crystal.get_unit_cell().parameters() == pytest.approx(
                unit_cell.parameters()
            )
        if (
            str(expt.crystal.get_space_group().info()) == "P 6 2 2"
            and str(space_group.info()) == "P 3 2 1"
        ):
            # This is fine
            continue
        assert str(expt.crystal.get_space_group().info()) == str(space_group.info())
        assert expt.crystal.get_space_group() == space_group
