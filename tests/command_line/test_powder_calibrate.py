from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("pyFAI")

from dxtbx.serialize import load

from dials.command_line import powder_calibrate
from dials.command_line.powder_calibrate import (
    Geometry,
    Point,
    PowderCalibrator,
    get_expt_params,
)


@pytest.mark.parametrize(
    "eyeball, starting_geometry", [(True, "imported.expt"), (False, "eyeballed.expt")]
)
def test_calibrate_coarse(dials_data, tmp_path, eyeball, starting_geometry):

    aluminium_powder = dials_data("aluminium_standard", pathlib=True)

    starting_geom_exptlist = load.experiment_list(aluminium_powder / starting_geometry)

    def mocked_eyeball(self):
        """
        Mock the calibrate method to update obj geometry to an eyeballed one
        without calling matplotlib Widget tools
        """
        self.geometry.update_beam_pos(beam_coords_px=Point(1103, 1024))

    with patch.object(
        powder_calibrate.EyeballWidget,
        "eyeball",
        new=mocked_eyeball,
    ):
        test_calibrator = PowderCalibrator(
            expts=starting_geom_exptlist,
            standard="Al",
            eyeball=eyeball,
            calibrated_geometry=str(tmp_path / "test_calibrated.expt"),
            pyfai_improvement=str(tmp_path / "test_pyfai_improvement.png"),
            straight_lines=str(tmp_path / "test_straight_lines.png"),
        )
        test_calibrator.calibrate_with_calibrant(plots=False)

    calibrated_geom = test_calibrator.geometry

    expected_geom_exptlist = load.experiment_list(aluminium_powder / "calibrated.expt")
    expected_expt_params = get_expt_params(expected_geom_exptlist)
    expected_geom = Geometry(expt_params=expected_expt_params)

    assert pytest.approx(calibrated_geom.param, 1e-2) == expected_geom.param


def test_save_geom_to_expt(dials_data, tmp_path):

    aluminium_powder = dials_data("aluminium_standard", pathlib=True)

    imported_exptlist = load.experiment_list(aluminium_powder / "imported.expt")
    imported_expt_params = get_expt_params(imported_exptlist)
    imported_geom = Geometry(expt_params=imported_expt_params)

    outfile = tmp_path / "test_save.expt"
    imported_geom.save_to_expt(output=outfile)
    assert outfile.is_file()

    # test_save_expt, _ = parse_to_tuples(
    #     args=[str(outfile), "standard=Al", "eyeball=False"]
    # )
    # read_from_saved_geom = Geometry(expt_params=test_save_expt)
    #
    # assert imported_geom.param == pytest.approx(read_from_saved_geom.param)
