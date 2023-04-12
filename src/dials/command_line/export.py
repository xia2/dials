from __future__ import annotations

import logging
import sys

from iotbx.phil import parse
from libtbx import Auto

from dials.util import log, show_mail_handle_errors

logger = logging.getLogger("dials.command_line.export")

help_message = """

This program is used to export the results of dials processing in various
formats.

The output formats currently supported are:

MTZ format exports the files as an unmerged mtz file, ready for input to
downstream programs such as Pointless and Aimless. For exporting integrated,
but unscaled data, the required input is a models.expt file and an
integrated.refl file. For exporting scaled data, the required input is a
models.expt file and a scaled.refl file, also passing the option
intensity=scale.

NXS format exports the files as an NXmx file. The required input is a
models.expt file and an integrated.refl file.

MMCIF format exports the files as an mmcif file. The required input is a
models.expt file and an integrated.refl file.

XDS_ASCII format exports intensity data and the experiment metadata in the
same format as used by the output of XDS in the CORRECT step - output can
be scaled with XSCALE.

SADABS format exports intensity data (and geometry by direction cosines)
as an ersatz-SADABS format reverse engineered from the file format used by
EvalCCD for input to SADABS.

MOSFLM format exports the files as an index.mat mosflm-format matrix file and a
mosflm.in file containing basic instructions for input to mosflm. The required
input is an models.expt file.

XDS format exports a models.expt file as XDS.INP and XPARM.XDS files. If a
reflection file is given it will be exported as a SPOT.XDS file.

SHELX format exports intensity data in HKLF 4 format for use in the SHELX suite
of programs. As this file format does not contain unit cell parameters or
symmetry information a minimal instruction file is also written. Optionally the
expected contents of the asymmetric unit can be added to this instruciton file.

PETS format exports intensity data and diffraction data in the CIF format
used by PETS. This is primarily intended to produce files suitable for
dynamic diffraction refinement using Jana2020, which requires this format.

Examples::

  # Export to mtz
  dials.export models.expt integrated.refl
  dials.export models.expt integrated.refl mtz.hklout=integrated.mtz
  dials.export models.expt scaled.refl intensity=scale mtz.hklout=scaled.mtz

  # Export to nexus
  dials.export models.expt integrated.refl format=nxs
  dials.export models.expt integrated.refl format=nxs nxs.hklout=integrated.nxs

  # Export to mmcif
  dials.export models.expt integrated.refl format=mmcif
  dials.export models.expt integrated.refl format=mmcif mmcif.hklout=integrated.mmcif

  # Export to mosflm
  dials.export models.expt integrated.refl format=mosflm

  # Export to xds
  dials.export strong.refl format=xds
  dials.export indexed.refl format=xds
  dials.export models.expt format=xds
  dials.export models.expt indexed.refl format=xds

  # Export to shelx
  dials.export scaled.expt scaled.refl format=shelx
  dials.export scaled.expt scaled.refl format=shelx shelx.hklout=dials.hkl
  dials.export scaled.expt scaled.refl format=shelx composition=C3H7NO2S
"""

phil_scope = parse(
    """

  format = *mtz sadabs nxs mmcif mosflm xds xds_ascii json shelx pets
    .type = choice
    .help = "The output file format"

  intensity = *auto profile sum scale
    .type = choice(multi=True)
    .help = "Choice of which intensities to export. Allowed combinations:
            scale, profile, sum, profile+sum, sum+profile+scale. Auto will
            default to scale or profile+sum depending on if the data are scaled."


  debug = False
    .type = bool
    .help = "Output additional debugging information"

  mtz {

    combine_partials = True
      .type = bool
      .help = "Combine partials that have the same partial id into one
        reflection, with an updated partiality given by the sum of the
        individual partialities."

    partiality_threshold=0.4
      .type = float
      .help = "All reflections with partiality values above the partiality
        threshold will be retained. This is done after any combination of
        partials if applicable."

    min_isigi = -5
      .type = float
      .help = "Exclude reflections with unfeasible values of I/Sig(I)"

    force_static_model = False
      .type = bool
      .help = "Force program to use static model even if scan varying is present"

    filter_ice_rings = False
      .type = bool
      .help = "Filter reflections at ice ring resolutions"

    d_min = None
      .type = float
      .help = "Filter out reflections with d-spacing below d_min"

    hklout = auto
      .type = path
      .help = "The output MTZ filename, defaults to integrated.mtz or scaled_unmerged.mtz"
              "depending on if the input data are scaled."

    crystal_name = XTAL
      .type = str
      .help = "The name of the crystal, for the mtz file metadata"

    project_name = DIALS
      .type = str
      .help = "The project name for the mtz file metadata"

    best_unit_cell = None
    .type = unit_cell
    .help = "Best unit cell value, to use when performing resolution cutting,"
            "and as the overall unit cell in the exported mtz. If None, the median"
            "cell will be used."

    wavelength_tolerance=1e-4
      .type = float
      .help = "An absolute tolerance on the wavelength (in A)"

  }

  sadabs {

    hklout = integrated.sad
      .type = path
      .help = "The output raw sadabs file"
    run = 1
      .type = int
      .help = "Batch number / run number for output file"
    predict = False
      .type = bool
      .help = "Compute centroids with static model, not observations"

  }

  xds_ascii {

    hklout = DIALS.HKL
      .type = path
      .help = "The output raw hkl file"

  }

  nxs {

    hklout = integrated.nxs
      .type = path
      .help = "The output Nexus file"

    instrument_name = Unknown
      .type = str
      .help = "Name of the instrument/beamline"

    instrument_short_name = Unknown
      .type = str
      .help = "Short name for instrument/beamline, perhaps the acronym"

    source_name = Unknown
      .type = str
      .help = "Name of the source/facility"

    source_short_name = Unknown
      .type = str
      .help = "Short name for source, perhaps the acronym"

  }

  mmcif {

    hklout = auto
      .type = path
      .help = "The output CIF file, defaults to integrated.cif or scaled_unmerged.cif
        depending on if the data are scaled."

    compress = gz bz2 xz
      .type = choice
      .help = "Choose compression format (also appended to the file name)"

    pdb_version = *v5 v5_next
      .type = choice
      .help = "This controls which pdb mmcif dictionary version the output"
              "mmcif file should comply with. v5_next adds support for"
              "recording unmerged data as well as additional scan metadata"
              "and statistics, however writing can be slow for large datasets."
    scale = True
      .type = bool
      .help = "If True, apply a scale such that the minimum intensity is greater"
              "than (less negative than) the mmcif.min_scale value below."
    min_scale = -999999.0
      .type = float
      .help = "If mmcif.scale is True, scale all negative intensities such that"
              "they are less negative than this value."
  }

  mosflm {

    directory = mosflm
      .type = path
      .help = "The output directory for mosflm output"

  }

  xds {

    directory = xds
      .type = path
      .help = "The output directory for xds output"

  }

  json {

    filename = rlp.json
      .type = path
    compact = True
      .type = bool
    n_digits = 6
      .type = int(value_min=1)
      .help = "Number of decimal places to be used for representing the"
              "reciprocal lattice points."

  }

  shelx {

    hklout = dials.hkl
      .type = path
      .help = "The output hkl file"
    ins = dials.ins
      .type = path
      .help = "The output ins file"
    composition = CH
      .type = str
      .help = "The chemical composition of the asymmetric unit"
    scale = True
      .type = bool
      .help = "Scale reflections to maximise output precision in SHELX 8.2f format"
    scale_range = -9999.0, 9999.0
      .type = floats(size=2, value_min=-999999., value_max=9999999.)
      .help = "minimum or maximum intensity value after scaling."

  }

  pets {

    filename_prefix = dials_dyn
      .type = str
      .help = "The prefix for output files, where the default will produce"
              "dials_dyn.cif_pets"
    id = None
      .type = int
      .help = "The experiment ID to export from a multi-experiment list"

    partiality_cutoff = 0.99
      .type = float
      .help = "Cutoff for determining which reflections are deemed to be fully"
              "recorded"

    flag_filtering = False
      .type = bool
      .help = "If true, keep only the reflections where the relevant `integrated`"
              "flag is set (either `integrated_sum` or `integrated_prf`). This"
              "seems to be quite restrictive compared to"
              "PETS, so is not set by default."

    virtual_frame {
      excitation_error_cutoff = 0.04
        .type = float
        .help = "Excitation error cutoff determining which reflections are"
                "included in virtual frames"
      n_merged = 1
        .type = int
        .help = "Number of frames to merge in a virtual frame"
      step = 1
        .type = int
        .help = "Step between frames"

    }
  }

  output {
    log = dials.export.log
      .type = path
      .help = "The log filename"
  }
"""
)


def _check_input(experiments, reflections, params=None, check_intensities=False):
    if not experiments:
        raise ValueError("export requires an experiment list")
    if len(reflections) != 1:
        raise ValueError("export requires 1 reflection table")
    if check_intensities:
        if not params:
            raise ValueError("No parameters given")
        if "profile" not in params.intensity and "sum" not in params.intensity:
            raise ValueError(
                "Only intensity options containing sum or profile can be exported in this format"
            )
        if (
            "intensity.sum.value" not in reflections[0]
            and "intensity.prf.value" not in reflections[0]
        ):
            raise ValueError(
                "Unable to find 'intensity.sum.value' or 'intensity.prf.value' columns in reflection table."
            )


def export_mtz(params, experiments, reflections):
    """
    Export reflections in MTZ format

    :param params: The phil parameters
    :param experiments: The experiment list
    :param reflections: The reflection tables
    """

    _check_input(experiments, reflections)

    from dials.util.export_mtz import export_mtz

    # Handle case where user has passed data before integration
    if (
        "intensity.sum.value" not in reflections[0]
        and "intensity.prf.value" not in reflections[0]
    ):
        raise ValueError(
            "Error: No intensity data in reflections; cannot export un-integrated data to MTZ"
        )

    reflection_table = reflections[0]
    filename = params.mtz.hklout
    # if mtz filename is auto, then choose scaled.mtz or integrated.mtz
    if filename in (None, Auto, "auto"):
        if ("intensity.scale.value" in reflection_table) and (
            "intensity.scale.variance" in reflection_table
        ):
            filename = "scaled.mtz"
            logger.info("Data appear to be scaled, setting mtz.hklout = 'scaled.mtz'")
        else:
            filename = "integrated.mtz"
            logger.info(
                "Data appear to be unscaled, setting mtz.hklout = 'integrated.mtz'"
            )

    export_mtz(
        reflection_table,
        experiments,
        intensity_choice=params.intensity,
        filename=filename,
        best_unit_cell=params.mtz.best_unit_cell,
        partiality_threshold=params.mtz.partiality_threshold,
        combine_partials=params.mtz.combine_partials,
        min_isigi=params.mtz.min_isigi,
        filter_ice_rings=params.mtz.filter_ice_rings,
        d_min=params.mtz.d_min,
        force_static_model=params.mtz.force_static_model,
        crystal_name=params.mtz.crystal_name,
        project_name=params.mtz.project_name,
        wavelength_tolerance=params.mtz.wavelength_tolerance,
    )


def export_sadabs(params, experiments, reflections):
    """
    Export data in HKL format

    :param params: The phil parameters
    :param experiments: The experiment list
    :param reflections: The reflection tables
    """

    _check_input(experiments, reflections, params=params, check_intensities=True)

    from dials.util.export_sadabs import export_sadabs

    export_sadabs(reflections[0], experiments, params)


def export_xdsascii(params, experiments, reflections):
    """
    Export data in XDS_ASCII format

    :param params: The phil parameters
    :param experiments: The experiment list
    :param reflections: The reflection tables
    """

    _check_input(experiments, reflections, params=params, check_intensities=True)

    from dials.util.export_xds_ascii import export_xds_ascii

    export_xds_ascii(reflections[0], experiments, params)


def export_nexus(params, experiments, reflections):
    """
    Export data in Nexus format

    :param params: The phil parameters
    :param experiments: The experiment list
    :param reflections: The reflection tables
    """

    _check_input(experiments, reflections)

    from dials.util.nexus import dump

    dump(experiments, reflections[0], params.nxs)


def export_mmcif(params, experiments, reflections):
    """
    Export data in CIF format

    :param params: The phil parameters
    :param experiments: The experiment list
    :param reflections: The reflection tables
    """

    _check_input(experiments, reflections)

    from dials.util.export_mmcif import MMCIFOutputFile

    outfile = MMCIFOutputFile(params)
    outfile.write(experiments, reflections[0])


def export_mosflm(params, experiments, reflections):
    """
    Export stuff in mosflm format

    :param params: The phil parameters
    :param experiments: The experiment list
    :param reflections: The reflection tables
    """

    _check_input(experiments, [None])

    if reflections:
        sys.exit("Mosflm exporter does not need a reflection table")

    from dials.util.mosflm import dump

    dump(experiments, params.mosflm.directory)


def export_xds(params, experiments, reflections):
    """
    Export stuff in xds format

    :param params: The phil parameters
    :param experiments: The experiment list
    :param reflections: The reflection tables
    """

    # Check the input
    if len(reflections) > 1:
        raise ValueError("XDS exporter requires 0 or 1 reflection table")

    if reflections:
        reflections = reflections[0]

    from dials.util.xds import dump

    dump(experiments, reflections, params.xds.directory)


def export_json(params, experiments, reflections):
    """
    Export reflections in json format

    :param params: The phil parameters
    :param experiments: The experiment list
    :param reflections: The reflection tables
    """

    # Check the input
    _check_input(experiments, [None])
    if not reflections:
        raise ValueError("json exporter requires a reflection table")

    from scitbx.array_family import flex

    from dials.util import export_json

    imagesets = [expt.imageset for expt in experiments]

    if len(reflections) != len(imagesets):
        raise ValueError(
            "Mismatch between %d reflections lists and %d imagesets"
            % (len(reflections), len(imagesets))
        )
    selected_reflections = None
    for i, refl in enumerate(reflections):
        refl["imageset_id"] = flex.size_t(refl.size(), i)
        if selected_reflections is None:
            selected_reflections = refl
        else:
            selected_reflections.extend(refl)

    exporter = export_json.ReciprocalLatticeJson(experiments, selected_reflections)
    exporter.as_json(
        filename=params.json.filename,
        compact=params.json.compact,
        n_digits=params.json.n_digits,
        experiments=experiments,
    )


def export_shelx(params, experiments, reflections):
    """
    Export data in SHELX HKL format

    :param params: The phil parameters
    :param experiments: The experiment list
    :param reflections: The reflection tables
    """

    _check_input(experiments, reflections, params=params)

    # check for a single intensity choice
    if len(params.intensity) > 1:
        raise ValueError(
            "Only 1 intensity option can be exported in this format, please choose a single intensity option e.g. intensity=profile"
        )

    from dials.util.export_shelx import export_shelx

    export_shelx(reflections[0], experiments, params)


def export_pets(params, experiments, reflections):
    """
    Export reflections in PETS CIF format

    :param params: The phil parameters
    :param experiments: The experiment list
    :param reflections: The reflection tables
    """

    # Check the input
    _check_input(experiments, reflections, check_intensities=False)

    from dials.util.export_pets import PETSOutput

    pets_output = PETSOutput(experiments, reflections, params)
    pets_output.write_dyn_cif_pets()

    return


@show_mail_handle_errors()
def run(args=None):
    from dials.util.options import (
        ArgumentParser,
        reflections_and_experiments_from_files,
    )
    from dials.util.version import dials_version

    usage = "dials.export models.expt reflections.refl [options]"

    # Create the option parser
    parser = ArgumentParser(
        usage=usage,
        read_experiments=True,
        read_reflections=True,
        check_format=False,
        phil=phil_scope,
        epilog=help_message,
    )

    # Get the parameters
    params, options = parser.parse_args(args, show_diff_phil=False)

    # Configure the logging
    log.config(logfile=params.output.log)

    # Print the version number
    logger.info(dials_version())

    # Log the diff phil
    diff_phil = parser.diff_phil.as_str()
    if diff_phil != "":
        logger.info("The following parameters have been modified:\n")
        logger.info(diff_phil)

    if not params.input.experiments and not params.input.reflections:
        parser.print_help()
        sys.exit()

    # Get the experiments and reflections
    reflections, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )

    try:
        template_list = {
            str(e.imageset.get_template()) + ":{}:{}".format(*e.scan.get_image_range())
            for e in experiments
        }
    except AttributeError:
        # For stills we need a different approach
        template_list = {}

    # Print out some bulk info
    logger.info("-" * 80)
    for f in template_list:
        logger.info("  template: %s", f)
    logger.info("-" * 80)

    # do auto interpreting of intensity choice:
    # note that this may still fail certain checks further down the processing,
    # but these are the defaults to try
    if params.intensity in ([None], [Auto], ["auto"], Auto) and reflections:
        if ("intensity.scale.value" in reflections[0]) and (
            "intensity.scale.variance" in reflections[0]
        ):
            params.intensity = ["scale"]
            logger.info("Data appears to be scaled, setting intensity = scale")
        else:
            params.intensity = []
            if "intensity.sum.value" in reflections[0]:
                params.intensity.append("sum")
            if "intensity.prf.value" in reflections[0]:
                params.intensity.append("profile")
            logger.info(
                "Data appears to be unscaled, setting intensity = "
                + "+".join(params.intensity)
            )

    # Choose the exporter
    exporter = {
        "mtz": export_mtz,
        "sadabs": export_sadabs,
        "xds_ascii": export_xdsascii,
        "nxs": export_nexus,
        "mmcif": export_mmcif,
        "mosflm": export_mosflm,
        "xds": export_xds,
        "json": export_json,
        "shelx": export_shelx,
        "pets": export_pets,
    }.get(params.format)
    if not exporter:
        sys.exit(f"Unknown format: {params.format}")

    # Export the data
    try:
        exporter(params, experiments, reflections)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
