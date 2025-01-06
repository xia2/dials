# test running data reduction programs on ssx data
from __future__ import annotations

import shutil
import subprocess

import pytest


@pytest.mark.skipif(
    shutil.which("gemmi") is None, reason="Could not find gemmi executable"
)
def test_ssx_reduction(dials_data, tmp_path):
    """
    Check that dials.cosym, dials.scale, dials.export and dials.merge run
    successfully on ssx data.
    Also test a few smaller analysis programs.
    """
    ssx = dials_data("cunir_serial_processed", pathlib=True)
    ssx_data = dials_data("cunir_serial", pathlib=True)
    refls = ssx / "integrated.refl"
    expts = ssx / "integrated.expt"

    result = subprocess.run(
        [shutil.which("dials.cosym"), expts, refls, "partiality_threshold=0.25"],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    cosym_expts = tmp_path / "symmetrized.expt"
    cosym_refls = tmp_path / "symmetrized.refl"
    assert cosym_expts.is_file()
    assert cosym_refls.is_file()
    assert (tmp_path / "dials.cosym.html").is_file()

    result = subprocess.run(
        [shutil.which("dials.scale"), cosym_expts, cosym_refls],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    scale_expts = tmp_path / "scaled.expt"
    scale_refls = tmp_path / "scaled.refl"
    assert scale_expts.is_file()
    assert scale_refls.is_file()
    assert (tmp_path / "dials.scale.html").is_file()

    # run scaling with reference model / cif
    for reference in [
        ssx_data / "2BW4.pdb",
        ssx_data / "2bw4.cif",
        ssx_data / "2bw4-sf.cif",
    ]:
        result = subprocess.run(
            [
                shutil.which("dials.scale"),
                cosym_expts,
                cosym_refls,
                f"reference={reference}",
                "output.experiments=scaled_ref.expt",
                "output.reflections=scaled_ref.refl",
            ],
            cwd=tmp_path,
            capture_output=True,
        )
    assert not result.returncode and not result.stderr

    result = subprocess.run(
        [shutil.which("dials.export"), scale_expts, scale_refls],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.mtz").is_file()

    # now try running mmcif export
    result = subprocess.run(
        [shutil.which("dials.export"), scale_expts, scale_refls, "format=mmcif"],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "scaled.cif").is_file()
    # check that gemmi can understand the output cif
    cmd = [
        shutil.which("gemmi"),
        "cif2mtz",
        tmp_path / "scaled.cif",
        tmp_path / "test.mtz",
    ]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True)
    assert not result.returncode and not result.stderr
    assert (tmp_path / "test.mtz").is_file()

    result = subprocess.run(
        [shutil.which("dials.merge"), scale_expts, scale_refls],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "merged.mtz").is_file()

    result = subprocess.run(
        [shutil.which("dials.damage_analysis"), scale_expts, scale_refls],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "dials.damage_analysis.html").is_file()

    result = subprocess.run(
        [shutil.which("dials.compute_delta_cchalf"), scale_expts, scale_refls],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "compute_delta_cchalf.html").is_file()

    # will not be able to refine error model due to lack of data, but should rather exit cleanly.
    result = subprocess.run(
        [shutil.which("dials.refine_error_model"), scale_expts, scale_refls],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
