from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from dxtbx.serialize import load

from dials.algorithms.integration.processor import _average_bbox_size
from dials.array_family import flex


def test_basic_integrate(dials_data, tmp_path):
    # Call dials.integrate

    exp = load.experiment_list(
        dials_data("centroid_test_data", pathlib=True) / "experiments.json"
    )
    exp[0].identifier = "foo"
    exp.as_json(tmp_path / "modified_input.json")

    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            "modified_input.json",
            "profile.fitting=False",
            "integration.integrator=3d",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    experiments = load.experiment_list(tmp_path / "integrated.expt")
    assert experiments[0].identifier == "foo"

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    mask = table.get_flags(table.flags.integrated, all=False)
    assert len(table) == 1666
    assert mask.count(True) == 1666
    assert "id" in table
    for row in table.rows():
        assert row["id"] == 0

    assert dict(table.experiment_identifiers()) == {0: "foo"}

    originaltable = table

    (tmp_path / "integrated.refl").unlink()
    for i in range(1, 10):
        source = dials_data("centroid_test_data", pathlib=True) / f"centroid_000{i}.cbf"
        destination = tmp_path / f"centroid_001{i}.cbf"
        try:
            destination.symlink_to(source)
        except OSError:
            shutil.copyfile(source, destination)

    with (
        dials_data("centroid_test_data", pathlib=True)
        .joinpath("experiments.json")
        .open("r") as fh
    ):
        j = json.load(fh)
    assert j["scan"][0]["image_range"] == [1, 9]
    j["scan"][0]["image_range"] = [11, 19]
    assert j["scan"][0]["oscillation"] == [0.0, 0.2]
    j["scan"][0]["oscillation"] = [360.0, 0.2]
    j["experiment"][0]["identifier"] = "bar"
    with (tmp_path / "models.expt").open("w") as fh:
        json.dump(j, fh)

    # Call dials.integrate
    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            "models.expt",
            "profile.fitting=False",
            "integration.integrator=3d",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    experiments = load.experiment_list(tmp_path / "integrated.expt")
    assert experiments[0].identifier == "bar"

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert dict(table.experiment_identifiers()) == {0: "bar"}
    mask1 = table.get_flags(table.flags.integrated, all=False)
    assert len(table) == 1666
    assert mask1.count(True) == 1666
    mask2 = originaltable.get_flags(table.flags.integrated, all=False)
    assert mask1.all_eq(mask2)
    t1 = table.select(mask1)
    t2 = originaltable.select(mask1)
    Cal_P1 = t1["xyzcal.mm"].parts()[2]
    Cal_Z1 = t1["xyzcal.px"].parts()[2]
    Obs_Z1 = t1["xyzobs.px.value"].parts()[2]
    # Obs_P1 = t1['xyzobs.mm.value'].parts()[2]
    Cal_Z2 = t2["xyzcal.px"].parts()[2]
    Cal_P2 = t2["xyzcal.mm"].parts()[2]
    Obs_Z2 = t2["xyzobs.px.value"].parts()[2]
    # Obs_P2 = t2['xyzobs.mm.value'].parts()[2]
    diff_I = t1["intensity.sum.value"] - t2["intensity.sum.value"]
    diff_Cal_Z = Cal_Z1 - (Cal_Z2 + 10)
    diff_Obs_Z = Obs_Z1 - (Obs_Z2 + 10)
    diff_Cal_P = Cal_P1 - (Cal_P2 + 2 * math.pi)
    # diff_Obs_P = Obs_P1 - (Obs_P2 + 2*math.pi)
    assert flex.abs(diff_I).all_lt(1e-7)
    assert flex.abs(diff_Cal_Z).all_lt(1e-7)
    assert flex.abs(diff_Cal_P).all_lt(1e-7)
    assert flex.abs(diff_Obs_Z).all_lt(1e-7)
    # assert(flex.abs(diff_Obs_P).all_lt(1e-7))


@pytest.mark.parametrize(
    ("block_size", "block_units"),
    [(None, None), (1, "degrees"), (2, "frames"), (1, "frames")],
)
def test_basic_blocking_options(dials_data, tmp_path, block_size, block_units):
    exp = load.experiment_list(
        dials_data("centroid_test_data", pathlib=True) / "experiments.json"
    )
    exp[0].identifier = "foo"
    exp.as_json(tmp_path / "modified_input.json")

    args = [shutil.which("dials.integrate"), "modified_input.json", "nproc=2"]
    if block_size:
        args.append(f"block.size={block_size}")
    if block_units:
        args.append(f"block.units={block_units}")

    result = subprocess.run(args, cwd=tmp_path, capture_output=True)
    assert not result.returncode and not result.stderr


@pytest.mark.skip(reason="3d threaded integrator")
def test_basic_threaded_integrate(dials_data, tmp_path):
    """Test the threaded integrator on single imageset data."""

    expts = dials_data("centroid_test_data", pathlib=True) / "indexed.expt"
    refls = dials_data("centroid_test_data", pathlib=True) / "indexed.refl"

    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            "integration.integrator=3d_threaded",
            "background.algorithm=glm",
            "njobs=2",
            "nproc=2",
            refls,
            expts,
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert tmp_path.joinpath("integrated.refl").is_file()
    assert tmp_path.joinpath("integrated.expt").is_file()

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert table.size() == 3526
    assert set(table["id"]) == {0}
    assert table.select(table["id"] == 0).size() == 3526


def test_basic_integrate_output_integrated_only(dials_data, tmp_path):
    exp = load.experiment_list(
        dials_data("centroid_test_data", pathlib=True) / "experiments.json"
    )
    exp[0].identifier = "bar"
    exp.as_json(tmp_path / "modified_input.json")

    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            "modified_input.json",
            "profile.fitting=False",
            "integration.integrator=3d",
            "output_unintegrated_reflections=False",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    experiments = load.experiment_list(tmp_path / "integrated.expt")
    assert experiments[0].identifier == "bar"

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    mask = table.get_flags(table.flags.integrated, all=False)
    assert len(table) == 1666
    assert mask.count(False) == 0
    assert "id" in table
    for row in table.rows():
        assert row["id"] == 0

    assert dict(table.experiment_identifiers()) == {0: "bar"}


def test_integration_with_sampling(dials_data, tmp_path):
    exp = load.experiment_list(
        dials_data("centroid_test_data", pathlib=True) / "experiments.json"
    )
    exp[0].identifier = "foo"
    exp.as_json(tmp_path / "modified_input.json")

    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            "modified_input.json",
            "profile.fitting=False",
            "sampling.integrate_all_reflections=False",
            "sampling.random_seed=42",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    experiments = load.experiment_list(tmp_path / "integrated.expt")
    assert experiments[0].identifier == "foo"

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")

    # account for random number generator in sampling
    assert len(table) == 839

    assert dict(table.experiment_identifiers()) == {0: "foo"}


def test_integration_with_sample_size(dials_data, tmp_path):
    exp = load.experiment_list(
        dials_data("centroid_test_data", pathlib=True) / "experiments.json"
    )
    exp[0].identifier = "foo"
    exp.as_json(tmp_path / "modified_input.json")

    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            "modified_input.json",
            "profile.fitting=False",
            "sampling.integrate_all_reflections=False",
            "sampling.random_seed=42",
            "sampling.minimum_sample_size=500",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    experiments = load.experiment_list(tmp_path / "integrated.expt")
    assert experiments[0].identifier == "foo"
    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert len(table) == 415
    assert dict(table.experiment_identifiers()) == {0: "foo"}


def test_integration_with_image_exclusions(dials_data, tmp_path):
    exp = load.experiment_list(
        dials_data("centroid_test_data", pathlib=True) / "experiments.json"
    )
    exp[0].identifier = "foo"
    exp.as_json(tmp_path / "modified_input.json")

    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            "modified_input.json",
            "profile.fitting=False",
            "exclude_images=4:6",
            "sampling.random_seed=42",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    experiments = load.experiment_list(tmp_path / "integrated.expt")
    assert experiments[0].identifier == "foo"

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")

    assert len(table) == 91

    # NB the excluded range of centroids is actually wider than the
    # exclude_images specification, because reflections that extend onto the
    # excluded images are also rejected.
    _, _, z = table["xyzcal.px"].parts()
    assert len(z.select((z > 3) & (z < 7))) == 0


def test_imageset_id_output_with_multi_sweep(dials_data, tmp_path):
    """Test that imageset ids are correctly output for multi-sweep integration."""
    # Just integrate 15 images for each sweep

    images1 = dials_data("l_cysteine_dials_output", pathlib=True) / "l-cyst_01_000*.cbf"
    images2 = dials_data("l_cysteine_dials_output", pathlib=True) / "l-cyst_02_000*.cbf"

    result = subprocess.run(
        [shutil.which("dials.import"), images1, images2],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    result = subprocess.run(
        [shutil.which("dials.find_spots"), tmp_path / "imported.expt", "nproc=1"],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    result = subprocess.run(
        [
            shutil.which("dials.index"),
            tmp_path / "imported.expt",
            tmp_path / "strong.refl",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            tmp_path / "indexed.expt",
            tmp_path / "indexed.refl",
            "profile.fitting=False",
            "gaussian_rs.min_spots.overall=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert set(table["imageset_id"]) == {0, 1}
    # check that we have approx 50% in each
    n0 = (table["imageset_id"] == 0).count(True)
    n = table.size()
    n1 = (table["imageset_id"] == 1).count(True)
    assert (n0 / n > 0.4) and (n0 / n < 0.6)
    assert (n1 / n > 0.4) and (n1 / n < 0.6)

    # now try again with adding unintegrated reflections
    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            tmp_path / "indexed.expt",
            tmp_path / "indexed.refl",
            "profile.fitting=False",
            "gaussian_rs.min_spots.overall=0",
            "output_unintegrated_reflections=False",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert set(table["imageset_id"]) == {0, 1}
    # check that we have approx 50% in each
    n0 = (table["imageset_id"] == 0).count(True)
    n = table.size()
    n1 = (table["imageset_id"] == 1).count(True)
    assert (n0 / n > 0.4) and (n0 / n < 0.6)
    assert (n1 / n > 0.4) and (n1 / n < 0.6)


def test_basic_integration_with_profile_fitting(dials_data, tmp_path):
    expts = dials_data("centroid_test_data", pathlib=True) / "indexed.expt"
    refls = dials_data("centroid_test_data", pathlib=True) / "indexed.refl"
    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            expts,
            refls,
            "profile.fitting=True",
            "sampling.integrate_all_reflections=False",
            "sampling.random_seed=42",
            "sampling.minimum_sample_size=500",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert len(table) == 436

    prf = table.get_flags(table.flags.integrated_prf)
    zero = table["intensity.prf.value"] == 0.0
    prf_and_zero = prf & zero
    assert prf_and_zero.count(True) == 0


@pytest.mark.xfail(
    sys.platform == "darwin",
    reason="Not understood apparent platform numeric differences",
)
def test_multi_sweep(dials_regression: pathlib.Path, tmp_path):
    expts = os.path.join(
        dials_regression, "integration_test_data", "multi_sweep", "experiments.json"
    )

    experiments = load.experiment_list(expts)
    for i, expt in enumerate(experiments):
        expt.identifier = str(100 + i)

    # Patched data file. Original had trusted_range from -1, but now this range
    # is defined to start from the minimum trusted value. This test should be
    # updated with new data in dials-data (i.e. not dials_regression).
    # https://github.com/dials/dials/issues/2200
    panel = experiments[0].detector[0]
    max_trusted = panel.get_trusted_range()[1]
    panel.set_trusted_range((0, max_trusted))

    experiments.as_json(tmp_path / "modified_input.json")

    refls = os.path.join(
        dials_regression, "integration_test_data", "multi_sweep", "indexed.pickle"
    )

    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            "modified_input.json",
            refls,
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "integrated.refl").is_file()
    assert (tmp_path / "integrated.expt").is_file()

    experiments = load.experiment_list(tmp_path / "integrated.expt")
    for i, expt in enumerate(experiments):
        assert expt.identifier == str(100 + i)

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert len(table) == 3530
    assert dict(table.experiment_identifiers()) == {0: "100", 1: "101"}

    # Check the results
    T1 = table[:1765]
    T2 = table[1765:]
    ID1 = list(set(T1["id"]))
    ID2 = list(set(T2["id"]))
    assert len(ID1) == 1
    assert len(ID2) == 1
    assert ID1[0] == 0
    assert ID2[0] == 1
    I1 = T1["intensity.prf.value"]
    I2 = T2["intensity.prf.value"]
    F1 = T1.get_flags(T1.flags.integrated_prf)
    F2 = T2.get_flags(T2.flags.integrated_prf)
    assert F1 == F2
    I1 = I1.select(F1)
    I2 = I2.select(F2)
    assert flex.abs(I1 - I2) < 1e-6


def test_multi_lattice(dials_regression: pathlib.Path, tmp_path):
    expts = os.path.join(
        dials_regression, "integration_test_data", "multi_lattice", "experiments.json"
    )

    experiments = load.experiment_list(expts)
    for i, expt in enumerate(experiments):
        expt.identifier = str(100 + i)
    experiments.as_json(tmp_path / "modified_input.json")

    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            "modified_input.json",
            os.path.join(
                dials_regression,
                "integration_test_data",
                "multi_lattice",
                "indexed.pickle",
            ),
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "integrated.refl").is_file()
    assert (tmp_path / "integrated.expt").is_file()

    experiments = load.experiment_list(tmp_path / "integrated.expt")
    for i, expt in enumerate(experiments):
        assert expt.identifier == str(100 + i)

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert len(table) == 4962
    assert dict(table.experiment_identifiers()) == {0: "100", 1: "101"}
    # both should only have the imageset_id of zero as they share an imageset
    assert set(table["imageset_id"]) == {0}

    # Check output contains from two lattices
    exp_id = list(set(table["id"]))
    assert len(exp_id) == 2

    # Check both lattices have integrated reflections
    mask = table.get_flags(table.flags.integrated_prf)
    table = table.select(mask)
    exp_id = list(set(table["id"]))
    assert len(exp_id) == 2


def test_output_rubbish(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials.index"),
            dials_data("centroid_test_data", pathlib=True)
            / "imported_experiments.json",
            dials_data("centroid_test_data", pathlib=True) / "strong.pickle",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "indexed.expt").is_file()
    assert (tmp_path / "indexed.refl").is_file()

    # Call dials.integrate
    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            "nproc=1",
            "indexed.expt",
            "indexed.refl",
            "profile.fitting=False",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "integrated.refl").is_file()

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert "id" in table
    for row in table.rows():
        assert row["id"] == 0

    assert list(table.experiment_identifiers().keys()) == [0]
    assert list(table.experiment_identifiers().values())  # not empty


def test_integrate_with_kapton(dials_regression: pathlib.Path, tmp_path):
    pickle_name = "idx-20161021225550223_indexed.pickle"
    json_name = "idx-20161021225550223_refined_experiments.json"
    image_name = "20161021225550223.pickle"
    pickle_path = os.path.join(
        dials_regression, "integration_test_data", "stills_PSII", pickle_name
    )
    json_path = os.path.join(
        dials_regression, "integration_test_data", "stills_PSII", json_name
    )
    image_path = os.path.join(
        dials_regression, "integration_test_data", "stills_PSII", image_name
    )

    assert os.path.exists(pickle_path)
    assert os.path.exists(json_path)
    shutil.copy(pickle_path, tmp_path)
    shutil.copy(image_path, tmp_path)

    with open(json_path) as r:
        (tmp_path / json_name).write_text(
            r.read() % os.fspath(tmp_path).replace("\\", "\\\\")
        )

    templ_phil = """
      output {
        experiments = 'idx-20161021225550223_integrated_experiments_%s.expt'
        reflections = 'idx-20161021225550223_integrated_%s.refl'
      }
      integration {
        lookup.mask = '%s'
        integrator = stills
        profile.fitting = False
        background.algorithm = simple
        debug {
          output = True
          separate_files = False
          split_experiments = False
        }
      }
      profile {
        gaussian_rs.min_spots.overall = 0
      }
      absorption_correction {
        apply = %s
        algorithm = fuller_kapton
        fuller_kapton {
          smart_sigmas = True
        }
      }
"""
    without_kapton_phil = templ_phil % (
        "nokapton",
        "nokapton",
        os.path.join(
            dials_regression, "integration_test_data", "stills_PSII", "mask.pickle"
        ).replace("\\", "\\\\"),
        "False",
    )
    with_kapton_phil = templ_phil % (
        "kapton",
        "kapton",
        os.path.join(
            dials_regression, "integration_test_data", "stills_PSII", "mask.pickle"
        ).replace("\\", "\\\\"),
        "True",
    )

    (tmp_path / "integrate_without_kapton.phil").write_text(without_kapton_phil)

    (tmp_path / "integrate_with_kapton.phil").write_text(with_kapton_phil)

    # Call dials.integrate with and without kapton correction
    for phil in "integrate_without_kapton.phil", "integrate_with_kapton.phil":
        result = subprocess.run(
            ["dials.integrate", "nproc=1", pickle_name, json_name, phil],
            cwd=tmp_path,
            capture_output=True,
        )
        assert not result.returncode and not result.stderr

    results = []
    for mode in "kapton", "nokapton":
        table = flex.reflection_table.from_file(
            tmp_path / f"idx-20161021225550223_integrated_{mode}.refl"
        )
        millers = table["miller_index"]
        test_indices = {"zero": (-5, 2, -6), "low": (-2, -20, 7), "high": (-1, -10, 4)}
        test_rows = {k: millers.first_index(v) for k, v in test_indices.items()}
        test_I_sigsqI = {
            k: (table[v]["intensity.sum.value"], table[v]["intensity.sum.variance"])
            for k, v in test_rows.items()
        }
        results.append(test_I_sigsqI)
    assert results[0]["zero"][0] == results[1]["zero"][0]
    assert results[0]["zero"][1] - results[1]["zero"][1] < 0.0001
    assert False not in [results[0]["low"][i] > results[1]["low"][i] for i in (0, 1)]
    assert False not in [results[0]["high"][i] > results[1]["high"][i] for i in (0, 1)]


def test_average_bbox_size():
    """Test behaviour of function for obtaining average bbox size."""
    reflections = flex.reflection_table()
    reflections["bbox"] = flex.int6(*(flex.int(10, i) for i in range(6)))
    assert _average_bbox_size(reflections) == (1, 1, 1)
