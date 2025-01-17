from __future__ import annotations

import datetime
import pickle
import shutil
import subprocess
import sys

import pytest

from scitbx.array_family import flex

@pytest.mark.skip(reason="Apparently causes SEGV on some platforms")
@pytest.mark.skipif(
    (sys.platform == "darwin")
    and (datetime.date.today() < datetime.date(2023, 10, 20)),
    reason="Temporary skip to dodge SEGV on Azure pipelines",
)
def test_model_background(dials_data, tmp_path):
    centroid = dials_data("centroid_test_data", pathlib=True)
    expts = centroid / "experiments.json"

    result = subprocess.run(
        [shutil.which("dials.model_background"), expts],
        cwd=tmp_path,
        capture_output=True,
    )
    print(result.stderr.decode())
    result.check_returncode()
    assert not result.stderr
    for filename in (
        "background.pickle",
        "mean_0.png",
        "variance_0.png",
        "dispersion_0.png",
        "mask_0.png",
        "min_0.png",
        "max_0.png",
        "model_0.png",
    ):
        assert (tmp_path / filename).is_file()

    with open((tmp_path / "background.pickle"), "rb") as f:
        background = pickle.load(f)

    panel = 0
    data = background.data(panel)
    assert data.all() == (2527, 2463)
    min_max_mean = flex.min_max_mean_double(data.as_1d())
    assert min_max_mean.max == pytest.approx(5.9114028830604095)
    assert min_max_mean.min == 0.0
    assert min_max_mean.mean == pytest.approx(0.5013730161480899)

    # Test integration using the background model, with robust.algorithm=(True|False)
    refls = centroid / "indexed.refl"

    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            expts,
            refls,
            "background.algorithm=gmodel",
            "gmodel.robust.algorithm=False",
            "gmodel.model=background.pickle",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    result = subprocess.run(
        [
            shutil.which("dials.integrate"),
            expts,
            refls,
            "background.algorithm=gmodel",
            "gmodel.robust.algorithm=True",
            "gmodel.model=background.pickle",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
