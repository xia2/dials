from __future__ import annotations

import os
import shutil
import subprocess


def test(dials_data, tmp_path):

    mtz_file = (
        dials_data("lysozyme_electron_diffraction", pathlib=True) / "refmac_final.mtz"
    )
    env = os.environ.copy()
    env[
        "PYTHONDEVMODE"
    ] = ""  # disable a developermode warning from pyparsing from mathtext in matplotlib
    result = subprocess.run(
        [shutil.which("dials.plot_Fo_vs_Fc"), f"hklin={mtz_file}"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert tmp_path.joinpath("Fo_vs_Fc.pdf").is_file()
    assert "|Fe| = 42.0" in result.stdout.decode()
