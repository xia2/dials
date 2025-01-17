from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from iotbx import ccp4_map
from scitbx.array_family import flex


@pytest.mark.skipif(os.name == "nt", reason="does not run on windows")
def test_rs_mapper(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials.rs_mapper"),
            "nproc=1",
            dials_data("centroid_test_data", pathlib=True)
            / "imported_experiments.json",
            'map_file="junk.ccp4"',
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "junk.ccp4").is_file()

    # load results
    m = ccp4_map.map_reader(file_name=str(tmp_path / "junk.ccp4"))
    assert len(m.data) == 7189057
    assert m.header_min == 0.0
    assert flex.min(m.data) == 0.0

    assert m.header_max == 2052.75
    assert flex.max(m.data) == 2052.75

    assert m.header_mean == pytest.approx(0.018924040719866753, abs=1e-6)
    assert flex.mean(m.data) == pytest.approx(0.01892407052218914, abs=1e-6)


def test_multi_panel(dials_data, tmp_path):
    data_dir = dials_data("image_examples", pathlib=True)
    image = data_dir / "DLS_I23_germ_13KeV_0001.cbf"

    result = subprocess.run(
        [
            shutil.which("dials.rs_mapper"),
            "nproc=1",
            image,
            'map_file="junk.ccp4"',
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "junk.ccp4").is_file()

    # load results
    m = ccp4_map.map_reader(file_name=str(tmp_path / "junk.ccp4"))
    assert len(m.data) == 7189057
    assert m.header_min == 0.0
    assert flex.min(m.data) == 0.0

    assert m.header_max == 31342.25
    assert flex.max(m.data) == 31342.25

    assert m.header_mean == pytest.approx(0.05911629647016525, abs=1e-6)
    assert flex.mean(m.data) == pytest.approx(0.05911629647016525, abs=1e-6)


@pytest.mark.skipif(os.name == "nt", reason="does not run on windows")
def test_masked(dials_data, tmp_path):
    subprocess.run(
        [
            shutil.which("dials.import"),
            dials_data("image_examples", pathlib=True) / "dectris_eiger_master.h5",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    result = subprocess.run(
        [
            shutil.which("dials.rs_mapper"),
            "nproc=1",
            "imported.expt",
            "map_file=masked.ccp4",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "masked.ccp4").is_file()

    # Also check the ignore case (regressions here may indicate changes in mask handling)
    result = subprocess.run(
        [
            shutil.which("dials.rs_mapper"),
            "nproc=1",
            "imported.expt",
            "map_file=unmasked.ccp4",
            "ignore_mask=True",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "unmasked.ccp4").is_file()

    # Load results
    masked = ccp4_map.map_reader(file_name=str(tmp_path / "masked.ccp4"))
    unmasked = ccp4_map.map_reader(file_name=str(tmp_path / "unmasked.ccp4"))

    assert masked.header_max < unmasked.header_max
    assert masked.header_max == pytest.approx(289.11111)
    assert unmasked.header_max == pytest.approx(65535.0)
