"""Tests for dials.merge command line program."""


from __future__ import annotations

import json

import procrunner
import pytest

from cctbx import uctbx
from dxtbx.serialize import load
from iotbx import mtz

from dials.array_family import flex


def validate_mtz(mtz_file, expected_labels, unexpected_labels=None):
    assert mtz_file.is_file()
    m = mtz.object(str(mtz_file))

    assert m.as_miller_arrays()[0].info().wavelength == pytest.approx(0.6889)
    labels = set()
    for ma in m.as_miller_arrays(merge_equivalents=False):
        labels.update(ma.info().labels)
    for l in expected_labels:
        assert l in labels
    if unexpected_labels:
        for l in unexpected_labels:
            assert l not in labels


@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize(
    "truncate,french_wilson_impl", [(True, "dials"), (True, "cctbx"), (False, None)]
)
def test_merge(dials_data, tmp_path, anomalous, truncate, french_wilson_impl):
    """Test the command line script with LCY data"""
    # Main options: truncate on/off, anomalous on/off
    # french_wilson.implementation dials/cctbx

    mean_labels = ["IMEAN", "SIGIMEAN"]
    anom_labels = ["I(+)", "I(-)", "SIGI(+)", "SIGI(-)"]
    amp_labels = ["F", "SIGF"]
    anom_amp_labels = ["F(+)", "SIGF(+)", "F(-)", "SIGF(-)", "DANO", "SIGDANO"]

    location = dials_data("l_cysteine_4_sweeps_scaled", pathlib=True)
    refls = location / "scaled_20_25.refl"
    expts = location / "scaled_20_25.expt"

    mtz_file = tmp_path / f"merge-{anomalous}-{truncate}.mtz"

    command = [
        "dials.merge",
        refls,
        expts,
        f"truncate={truncate}",
        f"french_wilson.implementation={french_wilson_impl}",
        f"anomalous={anomalous}",
        f"output.mtz={str(mtz_file)}",
        "project_name=ham",
        "crystal_name=jam",
        "dataset_name=spam",
        "json=dials.merge.json",
        "additional_stats=True",
    ]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "dials.merge.html").is_file()
    merge_json = tmp_path / "dials.merge.json"
    assert merge_json.is_file()
    expected_labels = mean_labels
    unexpected_labels = []

    with merge_json.open() as fh:
        json_d = json.load(fh)
        wl = list(json_d.keys())[0]
        for k in {"merging_stats", "merging_stats_anom"}:
            assert k in json_d[wl]
            assert {"d_star_sq_min", "n_obs", "cc_anom", "r_split"} <= json_d[wl][
                k
            ].keys()

    if truncate:
        expected_labels += amp_labels
    else:
        unexpected_labels += amp_labels

    if anomalous:
        expected_labels += anom_labels
    else:
        unexpected_labels += anom_labels

    if anomalous and truncate:
        expected_labels += anom_amp_labels
    else:
        unexpected_labels += anom_amp_labels

    validate_mtz(mtz_file, expected_labels, unexpected_labels)


@pytest.mark.parametrize("best_unit_cell", [None, "5.5,8.1,12.0,90,90,90"])
def test_merge_dmin_dmax(dials_data, tmp_path, best_unit_cell):
    """Test the d_min, d_max"""

    location = dials_data("l_cysteine_4_sweeps_scaled", pathlib=True)
    refls = location / "scaled_20_25.refl"
    expts = location / "scaled_20_25.expt"

    mtz_file = tmp_path / "merge.mtz"

    command = [
        "dials.merge",
        refls,
        expts,
        "truncate=False",
        "anomalous=False",
        "d_min=1.0",
        "d_max=8.0",
        f"output.mtz={str(mtz_file)}",
        "project_name=ham",
        "crystal_name=jam",
        "dataset_name=spam",
        f"best_unit_cell={best_unit_cell}",
        "output.html=None",
    ]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr

    # check the unit cell was correctly set if using best_unit_cell
    m = mtz.object(str(mtz_file))
    if best_unit_cell:
        for ma in m.as_miller_arrays():
            assert uctbx.unit_cell(best_unit_cell).parameters() == pytest.approx(
                ma.unit_cell().parameters()
            )

    # check we only have reflections in range 8 - 1A
    max_min_resolution = m.max_min_resolution()

    assert max_min_resolution[0] <= 8
    assert max_min_resolution[1] >= 1


def test_merge_multi_wavelength(dials_data, tmp_path):
    """Test that merge handles multi-wavelength data suitably - should be
    exported into an mtz with separate columns for each wavelength."""

    mean_labels = [f"{pre}IMEAN_WAVE{i}" for i in [1, 2] for pre in ["", "SIG"]]
    anom_labels = [
        f"{pre}I_WAVE{i}({sgn})"
        for i in [1, 2]
        for pre in ["", "SIG"]
        for sgn in ["+", "-"]
    ]
    amp_labels = [f"{pre}F_WAVE{i}" for i in [1, 2] for pre in ["", "SIG"]]
    anom_amp_labels = [
        f"{pre}F_WAVE{i}({sgn})"
        for i in [1, 2]
        for pre in ["", "SIG"]
        for sgn in ["+", "-"]
    ]

    location = dials_data("l_cysteine_4_sweeps_scaled", pathlib=True)
    refl1 = location / "scaled_30.refl"
    expt1 = location / "scaled_30.expt"
    refl2 = location / "scaled_35.refl"
    expt2 = location / "scaled_35.expt"
    expts1 = load.experiment_list(expt1, check_format=False)
    expts1[0].beam.set_wavelength(0.7)
    expts2 = load.experiment_list(expt2, check_format=False)
    expts1.extend(expts2)

    tmp_expt = tmp_path / "tmp.expt"
    expts1.as_json(tmp_expt)

    reflections1 = flex.reflection_table.from_file(refl1)
    reflections2 = flex.reflection_table.from_file(refl2)
    # first need to resolve identifiers - usually done on loading
    reflections2["id"] = flex.int(reflections2.size(), 1)
    del reflections2.experiment_identifiers()[0]
    reflections2.experiment_identifiers()[1] = "3"
    reflections1.extend(reflections2)

    tmp_refl = tmp_path / "tmp.refl"
    reflections1.as_file(tmp_refl)

    # Can now run after creating our 'fake' multiwavelength dataset
    command = ["dials.merge", tmp_refl, tmp_expt, "truncate=True", "anomalous=True"]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "merged.mtz").is_file()
    assert (tmp_path / "dials.merge.html").is_file()
    m = mtz.object(str(tmp_path / "merged.mtz"))
    labels = []
    for ma in m.as_miller_arrays(merge_equivalents=False):
        labels.extend(ma.info().labels)
    assert all(x in labels for x in mean_labels)
    assert all(x in labels for x in anom_labels)
    assert all(x in labels for x in amp_labels)
    assert all(x in labels for x in anom_amp_labels)

    # 6 miller arrays for each dataset, check the expected number of reflections.
    arrays = m.as_miller_arrays()
    assert len(arrays) == 12
    assert arrays[0].info().wavelength == pytest.approx(0.7)
    assert arrays[6].info().wavelength == pytest.approx(0.6889)
    assert abs(arrays[0].size() - 1223) < 10  # check number of miller indices
    assert abs(arrays[6].size() - 1453) < 10  # check number of miller indices

    # test changing the wavelength tolerance such that data is combined under
    # one wavelength. Check the number of reflections to confirm this.
    command = [
        "dials.merge",
        tmp_refl,
        tmp_expt,
        "truncate=True",
        "anomalous=True",
        "wavelength_tolerance=0.02",
    ]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr
    m = mtz.object(str(tmp_path / "merged.mtz"))
    arrays = m.as_miller_arrays()
    assert arrays[0].info().wavelength == pytest.approx(0.7)
    assert len(arrays) == 6
    assert abs(arrays[0].size() - 1538) < 10


def test_suitable_exit_for_bad_input_from_single_dataset(dials_data, tmp_path):
    location = dials_data("vmxi_proteinase_k_sweeps", pathlib=True)

    command = [
        "dials.merge",
        location / "experiments_0.json",
        location / "reflections_0.pickle",
    ]

    # unscaled data
    result = procrunner.run(command, working_directory=tmp_path)
    assert result.returncode
    assert (
        result.stderr.replace(b"\r", b"")
        == b"""Sorry: intensity.scale.value not found in the reflection table.
Only scaled data can be processed with dials.merge
"""
    )


def test_suitable_exit_for_bad_input_with_more_than_one_reflection_table(
    dials_data, tmp_path
):
    location = dials_data("vmxi_proteinase_k_sweeps", pathlib=True)

    command = [
        "dials.merge",
        location / "experiments_0.json",
        location / "reflections_0.pickle",
        location / "experiments_1.json",
        location / "reflections_1.pickle",
    ]

    # more than one reflection table.
    result = procrunner.run(command, working_directory=tmp_path)
    assert result.returncode
    assert (
        result.stderr.replace(b"\r", b"")
        == b"""Sorry: Only data scaled together as a single reflection dataset
can be processed with dials.merge
"""
    )


def test_merge_exclude_images(dials_data, tmp_path):
    """Test the command line script with LCY data: exclude_images"""

    location = dials_data("l_cysteine_4_sweeps_scaled", pathlib=True)
    refls = location / "scaled_30.refl"
    expts = location / "scaled_30.expt"

    mtz_file = tmp_path / "merge-exclude.mtz"

    command = [
        "dials.merge",
        refls,
        expts,
        f"output.mtz={str(mtz_file)}",
        "exclude_images=0:851:1700",
    ]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr

    # all the data together is 75% complete

    for record in result.stdout.decode().split("\n"):
        if record.startswith("Completeness"):
            assert float(record.split()[1]) < 70
