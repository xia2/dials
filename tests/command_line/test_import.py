from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
from pathlib import Path

import procrunner
import pytest

from dxtbx.imageset import ImageSequence
from dxtbx.serialize import load


@pytest.mark.parametrize("use_beam", ["True", "False"])
@pytest.mark.parametrize("use_gonio", ["True", "False"])
@pytest.mark.parametrize("use_detector", ["True", "False"])
def test_reference_individual(dials_data, tmpdir, use_beam, use_gonio, use_detector):

    expected_beam = {"True": 3, "False": 0.9795}
    expected_gonio = {
        "True": (7, 8, 9, 4, 5, 6, 1, 2, 3),
        "False": (1, 0, 0, 0, 1, 0, 0, 0, 1),
    }
    expected_detector = {"True": "Fake panel", "False": "Panel"}

    # Find the image files
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )

    # Create an experiment with some faked geometry items
    fake_phil = """
      geometry {
        beam.wavelength = 3
        detector.panel.name = "Fake panel"
        goniometer.fixed_rotation = 7,8,9,4,5,6,1,2,3
      }
  """
    tmpdir.join("fake.phil").write(fake_phil)
    fake_result = procrunner.run(
        ["dials.import", "fake.phil", "output.experiments=fake_geometry.expt"]
        + image_files,
        working_directory=tmpdir,
    )
    assert not fake_result.returncode and not fake_result.stderr
    assert tmpdir.join("fake_geometry.expt").check(file=1)

    # Write an import phil file
    import_phil = f"""
      input {{
        reference_geometry = fake_geometry.expt
        check_reference_geometry = False
        use_beam_reference = {use_beam}
        use_gonio_reference = {use_gonio}
        use_detector_reference = {use_detector}
      }}
  """
    tmpdir.join("test_reference_individual.phil").write(import_phil)

    result = procrunner.run(
        [
            "dials.import",
            "test_reference_individual.phil",
            "output.experiments=reference_geometry.expt",
        ]
        + image_files,
        working_directory=tmpdir,
    )
    assert not result.returncode and not result.stderr
    assert tmpdir.join("reference_geometry.expt").check(file=1)

    experiments = load.experiment_list(tmpdir.join("reference_geometry.expt"))
    assert experiments[0].identifier != ""
    imgset = experiments[0].imageset

    beam = imgset.get_beam()
    goniometer = imgset.get_goniometer()
    detector = imgset.get_detector()

    assert beam.get_wavelength() == expected_beam[use_beam]
    assert goniometer.get_fixed_rotation() == expected_gonio[use_gonio]
    assert detector[0].get_name() == expected_detector[use_detector]


def test_multiple_sequence_import_fails_when_not_allowed(dials_data, tmpdir):
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )
    del image_files[4]  # Delete filename to force two sequences

    # run without allowing multiple sequences
    result = procrunner.run(
        [
            "dials.import",
            "output.experiments=experiments_multiple_sequences.expt",
            "allow_multiple_sequence=False",
        ]
        + image_files,
        working_directory=tmpdir,
    )
    assert result.returncode == 1
    assert b"ore than 1 sequence" in result.stderr
    assert not tmpdir.join("experiments_multiple_sequences.expt").check()


def test_can_import_multiple_sequences(dials_data, tmpdir):
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )
    del image_files[4]  # Delete filename to force two sequences

    result = procrunner.run(
        ["dials.import", "output.experiments=experiments_multiple_sequences.expt"]
        + image_files,
        working_directory=tmpdir,
    )
    assert not result.returncode and not result.stderr
    assert tmpdir.join("experiments_multiple_sequences.expt").check(file=1)

    experiments = load.experiment_list(
        tmpdir.join("experiments_multiple_sequences.expt")
    )
    assert len(experiments) == 2
    for experiment in experiments:
        assert experiment.identifier != ""


def test_with_mask(dials_data, tmpdir):
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )
    mask_filename = dials_data("centroid_test_data", pathlib=True) / "mask.pickle"

    result = procrunner.run(
        [
            "dials.import",
            f"mask={mask_filename}",
            "output.experiments=experiments_with_mask.expt",
        ]
        + image_files,
        working_directory=tmpdir,
    )
    assert not result.returncode and not result.stderr
    assert tmpdir.join("experiments_with_mask.expt").check(file=1)

    experiments = load.experiment_list(tmpdir.join("experiments_with_mask.expt"))
    assert experiments[0].identifier != ""
    assert (
        pathlib.Path(experiments[0].imageset.external_lookup.mask.filename)
        == mask_filename
    )


def test_override_geometry(dials_data, tmpdir):
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )

    # Write a geometry phil file
    tmpdir.join("geometry.phil").write(
        """
      geometry {
        beam {
          wavelength = 2
          direction = (-1,0,0)
        }
        detector {
          panel {
            name = "New panel"
            type = "New type"
            pixel_size = 10,20
            image_size = 30,40
            trusted_range = 50,60
            thickness = 70
            material = "Si"
            fast_axis = -1,0,0
            slow_axis = 0,-1,0
            origin = 100,100,100
          }
        }
        goniometer {
          axes = 0,0,-1
          fixed_rotation = 0,1,2,3,4,5,6,7,8
          setting_rotation = 8,7,6,5,4,3,2,1,0
        }
        scan {
          image_range = 1,4
          oscillation = 1,2
        }
      }
  """
    )

    result = procrunner.run(
        ["dials.import", "geometry.phil", "output.experiments=override_geometry.expt"]
        + image_files,
        working_directory=tmpdir,
    )
    assert not result.returncode and not result.stderr
    assert tmpdir.join("override_geometry.expt").check(file=1)

    experiments = load.experiment_list(tmpdir.join("override_geometry.expt"))
    assert experiments[0].identifier != ""
    imgset = experiments[0].imageset

    beam = imgset.get_beam()
    detector = imgset.get_detector()
    goniometer = imgset.get_goniometer()
    scan = imgset.get_scan()

    assert beam.get_wavelength() == 2
    assert beam.get_sample_to_source_direction() == (-1, 0, 0)
    assert detector[0].get_name() == "New panel"
    assert detector[0].get_type() == "New type"
    assert detector[0].get_pixel_size() == (10, 20)
    assert detector[0].get_image_size() == (30, 40)
    assert detector[0].get_trusted_range() == (50, 60)
    assert detector[0].get_thickness() == 70
    assert detector[0].get_material() == "Si"
    assert detector[0].get_fast_axis() == (-1, 0, 0)
    assert detector[0].get_slow_axis() == (0, -1, 0)
    assert detector[0].get_origin() == (100, 100, 100)
    assert goniometer.get_rotation_axis_datum() == (0, 0, -1)
    assert goniometer.get_fixed_rotation() == (0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert goniometer.get_setting_rotation() == (8, 7, 6, 5, 4, 3, 2, 1, 0)
    assert scan.get_image_range() == (1, 4)
    assert scan.get_oscillation() == (1, 2)


def test_import_beam_centre(dials_data, tmpdir):
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )

    # provide mosflm beam centre to dials.import
    result = procrunner.run(
        [
            "dials.import",
            "mosflm_beam_centre=100,200",
            "output.experiments=mosflm_beam_centre.expt",
        ]
        + image_files,
        working_directory=tmpdir,
    )
    assert not result.returncode and not result.stderr
    assert tmpdir.join("mosflm_beam_centre.expt").check(file=1)

    experiments = load.experiment_list(tmpdir.join("mosflm_beam_centre.expt"))
    imgset = experiments[0].imageset
    assert experiments[0].identifier != ""
    beam_centre = imgset.get_detector()[0].get_beam_centre(imgset.get_beam().get_s0())
    assert beam_centre == pytest.approx((200, 100))

    # provide an alternative models.expt to get geometry from
    result = procrunner.run(
        [
            "dials.import",
            "reference_geometry=mosflm_beam_centre.expt",
            "output.experiments=mosflm_beam_centre2.expt",
        ]
        + image_files,
        working_directory=tmpdir,
    )
    assert not result.returncode and not result.stderr
    assert tmpdir.join("mosflm_beam_centre2.expt").check(file=1)
    experiments = load.experiment_list(tmpdir.join("mosflm_beam_centre2.expt"))
    imgset = experiments[0].imageset
    beam_centre = imgset.get_detector()[0].get_beam_centre(imgset.get_beam().get_s0())
    assert beam_centre == pytest.approx((200, 100))


def test_fast_slow_beam_centre(dials_regression, run_in_tmp_path):
    # test slow_fast_beam_centre with a multi-panel CS-PAD image
    impath = os.path.join(
        dials_regression,
        "image_examples",
        "LCLS_cspad_nexus",
        "idx-20130301060858401.cbf",
    )
    result = procrunner.run(
        [
            "dials.import",
            "fast_slow_beam_centre=42,134,18",
            "output.experiments=fast_slow_beam_centre.expt",
            impath,
        ]
    )
    assert not result.returncode and not result.stderr
    assert os.path.exists("fast_slow_beam_centre.expt")

    experiments = load.experiment_list("fast_slow_beam_centre.expt")
    imgset = experiments[0].imageset
    assert experiments[0].identifier != ""
    # beam centre on 18th panel
    s0 = imgset.get_beam().get_s0()
    beam_centre = imgset.get_detector()[18].get_beam_centre_px(s0)
    assert beam_centre == pytest.approx((42, 134))

    # check relative panel positions have not changed
    from scitbx import matrix

    o = matrix.col(imgset.get_detector()[0].get_origin())
    offsets = []
    for p in imgset.get_detector():
        intra_pnl = o - matrix.col(p.get_origin())
        offsets.append(intra_pnl.length())

    result = procrunner.run(
        ["dials.import", "output.experiments=reference.expt", impath]
    )
    assert not result.returncode and not result.stderr
    assert os.path.exists("reference.expt")

    ref_exp = load.experiment_list("reference.expt")
    ref_imset = ref_exp[0].imageset
    o = matrix.col(ref_imset.get_detector()[0].get_origin())
    ref_offsets = []
    for p in ref_imset.get_detector():
        intra_pnl = o - matrix.col(p.get_origin())
        ref_offsets.append(intra_pnl.length())
    assert offsets == pytest.approx(ref_offsets)


def test_from_image_files(dials_data, tmpdir):
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )

    # Import from the image files
    result = procrunner.run(
        ["dials.import", "output.experiments=imported.expt"] + image_files,
        working_directory=tmpdir,
    )
    assert not result.returncode
    assert tmpdir.join("imported.expt").check(file=1)
    # check that an experiment identifier is assigned
    exp = load.experiment_list(tmpdir.join("imported.expt"))
    assert exp[0].identifier != ""


def test_from_template(dials_data, tmpdir):
    # Find the image files
    templates = [
        dials_data("centroid_test_data", pathlib=True) / "centroid_####.cbf",
        dials_data("centroid_test_data", pathlib=True) / "centroid_0001.cbf",
    ]

    for template in templates:
        # Import from the image files
        result = procrunner.run(
            [
                "dials.import",
                f"template={template}",
                "output.experiments=imported.expt",
            ],
            working_directory=tmpdir,
        )

        assert not result.returncode
        assert tmpdir.join("imported.expt").check(file=1)
        # check that an experiment identifier is assigned
        exp = load.experiment_list(tmpdir.join("imported.expt"))
        assert exp[0].identifier != ""


def test_extrapolate_scan(dials_data, tmpdir):
    # First image file
    image = dials_data("centroid_test_data", pathlib=True) / "centroid_0001.cbf"

    result = procrunner.run(
        [
            "dials.import",
            image,
            "output.experiments=import_extrapolate.expt",
            "geometry.scan.image_range=1,900",
            "geometry.scan.extrapolate_scan=True",
        ],
        working_directory=tmpdir,
    )
    assert not result.returncode
    assert tmpdir.join("import_extrapolate.expt").check(file=1)
    # check that an experiment identifier is assigned
    exp = load.experiment_list(tmpdir.join("import_extrapolate.expt"))
    assert exp[0].identifier != ""


@pytest.fixture
def centroid_test_data_with_missing_image(dials_data, tmp_path):
    """
    Provide a testset with a missing image (#4) in a temporary directory
    (symlink if possible, copy if necessary), and clean up test files
    afterwards to conserve disk space.
    """
    images = sorted(
        dials_data("centroid_test_data", pathlib=True).glob(
            "centroid_*[1,2,3,5,6,7,8,9].cbf"
        )
    )
    for image in images:
        try:
            (tmp_path / image.name).symlink_to(image)
        except OSError:
            shutil.copy(image, tmp_path)
    yield tmp_path / "centroid_####.cbf"
    for image in images:
        try:
            (tmp_path / image.name).unlink()
        except PermissionError:
            pass


def test_template_with_missing_image_fails(centroid_test_data_with_missing_image):
    # This should fail because image #4 is missing
    for image_range in (None, (3, 5)):
        result = procrunner.run(
            ["dials.import", f"template={centroid_test_data_with_missing_image}"]
            + (["image_range=%i,%i" % image_range] if image_range else []),
            working_directory=centroid_test_data_with_missing_image.parent,
        )
        assert result.returncode
        assert b"Missing image 4 from imageset" in result.stderr


def test_template_with_missing_image_outside_of_image_range(
    centroid_test_data_with_missing_image,
):
    # Explicitly pass an image_range that doesn't include the missing image
    for image_range in ((1, 3), (5, 9)):
        result = procrunner.run(
            [
                "dials.import",
                f"template={centroid_test_data_with_missing_image}",
                "image_range=%i,%i" % image_range,
                "output.experiments=imported_%i_%i.expt" % image_range,
            ],
            working_directory=centroid_test_data_with_missing_image.parent,
        )
        assert not result.returncode
        expts = load.experiment_list(
            centroid_test_data_with_missing_image.parent.joinpath(
                "imported_%i_%i.expt" % image_range
            )
        )
        assert expts[0].scan.get_image_range() == image_range


def test_import_still_sequence_as_experiments(dials_data, tmp_path):
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )

    out = "experiments_as_still.expt"

    procrunner.run(
        ["dials.import", "scan.oscillation=0,0", f"output.experiments={out}"]
        + image_files,
        working_directory=tmp_path,
    )

    imported_exp = load.experiment_list(tmp_path / out)
    assert len(imported_exp) == len(image_files)
    for exp in imported_exp:
        assert exp.identifier != ""

    iset = {exp.imageset for exp in imported_exp}
    assert len(iset) == 1

    # verify scans, goniometers kept too
    assert all(exp.scan.get_oscillation() == (0.0, 0.0) for exp in imported_exp)
    assert all(exp.goniometer is not None for exp in imported_exp)


def test_import_still_sequence_as_experiments_subset(dials_data, tmpdir):
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )[3:6]

    out = "experiments_as_still.expt"

    procrunner.run(
        ["dials.import", "scan.oscillation=10,0", f"output.experiments={out}"]
        + image_files,
        working_directory=tmpdir,
    )

    imported_exp = load.experiment_list(tmpdir.join(out))
    assert len(imported_exp) == len(image_files)
    for exp in imported_exp:
        assert exp.identifier != ""

    iset = {exp.imageset for exp in imported_exp}
    assert len(iset) == 1

    # verify scans, goniometers kept too
    assert all(exp.scan.get_oscillation() == (10.0, 0.0) for exp in imported_exp)
    assert all(exp.goniometer is not None for exp in imported_exp)


def test_import_still_sequence_as_expts_subset_by_range(dials_data, tmp_path):
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )

    out = tmp_path / "experiments_as_still.expt"

    result = procrunner.run(
        [
            "dials.import",
            "scan.oscillation=10,0",
            "image_range=3,5",
            f"output.experiments={out}",
            *image_files,
        ],
        working_directory=tmp_path,
    )

    assert result.returncode == 0

    imported_exp = load.experiment_list(out)
    assert len(imported_exp) == 3
    for exp in imported_exp:
        assert exp.identifier != ""

    iset = {exp.imageset for exp in imported_exp}
    assert len(iset) == 1
    assert len(imported_exp[0].imageset) == 3

    assert list(iset)[0].get_image_identifier(0) == os.fspath(image_files[2])

    # verify scans, goniometers kept too
    assert all(exp.scan.get_oscillation() == (10.0, 0.0) for exp in imported_exp)
    assert all(exp.goniometer is not None for exp in imported_exp)


def test_import_still_sequence_as_experiments_split_subset(dials_data, tmpdir):
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )
    image_files = image_files[:3] + image_files[6:]

    out = "experiments_as_still.expt"

    procrunner.run(
        ["dials.import", "scan.oscillation=10,0", f"output.experiments={out}"]
        + image_files,
        working_directory=tmpdir,
    )

    imported_exp = load.experiment_list(tmpdir.join(out))
    assert len(imported_exp) == len(image_files)
    for exp in imported_exp:
        assert exp.identifier != ""

    iset = {exp.imageset for exp in imported_exp}
    assert len(iset) == 2


def test_with_convert_sequences_to_stills(dials_data, tmpdir):
    image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )
    result = procrunner.run(
        [
            "dials.import",
            "convert_sequences_to_stills=True",
            "output.experiments=experiments_as_stills.expt",
        ]
        + image_files,
        working_directory=tmpdir,
    )
    assert not result.returncode and not result.stderr
    assert tmpdir.join("experiments_as_stills.expt").check(file=1)

    experiments = load.experiment_list(tmpdir.join("experiments_as_stills.expt"))
    for exp in experiments:
        assert exp.identifier != ""

    # should be no goniometers
    assert experiments.scans() == [None]
    assert experiments.goniometers() == [None]

    # should be same number of imagesets as images
    assert len(experiments.imagesets()) == len(image_files)

    # all should call out as still too
    assert experiments.all_stills()


def test_convert_stills_to_sequences(dials_data, tmp_path):
    """Test with Jungfrau & Sacla data"""
    JF16M = dials_data("lysozyme_JF16M_4img", pathlib=True)
    result = subprocess.run(
        [
            shutil.which("dials.import"),
            "convert_stills_to_sequences=True",
            JF16M / "lyso009a_0087.JF07T32V01_master_4img.h5",
            "output.experiments=experiments_as_seq.expt",
        ],
        cwd=tmp_path,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "experiments_as_seq.expt").is_file()
    experiments = load.experiment_list(tmp_path / "experiments_as_seq.expt")
    for exp in experiments:
        assert exp.identifier != ""

    assert len(experiments.imagesets()) == 1
    assert isinstance(experiments.imagesets()[0], ImageSequence)
    assert len(experiments.scans()) == 4

    # Test with sacla data
    sacla_path = dials_data("image_examples", pathlib=True)
    image_path = sacla_path / "SACLA-MPCCD-run266702-0-subset.h5"
    result = subprocess.run(
        [
            shutil.which("dials.import"),
            "convert_stills_to_sequences=True",
            image_path,
            "output.experiments=experiments_as_seq_sacla.expt",
        ],
        cwd=tmp_path,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "experiments_as_seq_sacla.expt").is_file()
    experiments2 = load.experiment_list(tmp_path / "experiments_as_seq_sacla.expt")
    for exp in experiments2:
        assert exp.identifier != ""

    assert len(experiments2.imagesets()) == 1
    assert isinstance(experiments2.imagesets()[0], ImageSequence)
    assert len(experiments2.scans()) == 4

    # also add in something that is sequences, for completess
    centroid_image_files = sorted(
        dials_data("centroid_test_data", pathlib=True).glob("centroid*.cbf")
    )[
        :3
    ]  # just three images
    result = subprocess.run(
        [
            shutil.which("dials.import"),
            "convert_stills_to_sequences=True",
            image_path,
            "output.experiments=joint.expt",
        ]
        + centroid_image_files,
        cwd=tmp_path,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "joint.expt").is_file()
    experiments3 = load.experiment_list(tmp_path / "joint.expt")
    for exp in experiments3:
        assert exp.identifier != ""

    assert len(experiments3.imagesets()) == 2
    assert isinstance(experiments3.imagesets()[0], ImageSequence)
    assert isinstance(experiments3.imagesets()[1], ImageSequence)
    assert len(experiments3.scans()) == 5  # four for sacla stills, 1 for centroid data


def test_convert_stills_to_sequences_nonh5(dials_regression, tmp_path):
    image_path = Path(
        dials_regression,
        "image_examples",
        "LCLS_cspad_nexus",
        "idx-20130301060858801.cbf",
    )
    assert image_path.is_file()
    result = subprocess.run(
        [
            shutil.which("dials.import"),
            "convert_stills_to_sequences=True",
            image_path,
            "output.experiments=lcls.expt",
        ],
        cwd=tmp_path,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "lcls.expt").is_file()
    experiments = load.experiment_list(tmp_path / "lcls.expt")
    for exp in experiments:
        assert exp.identifier != ""

    assert len(experiments.imagesets()) == 1
    assert isinstance(experiments.imagesets()[0], ImageSequence)
    assert len(experiments.scans()) == 1  # only one image example here
