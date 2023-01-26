"""
Command line script to allow merging and truncating of a dials dataset.
"""

from __future__ import annotations

import json
import logging
import sys
from io import StringIO
from typing import List, Tuple

from dxtbx.model import ExperimentList
from iotbx import mtz, phil

from dials.algorithms.merging.merge import (
    MTZDataClass,
    collect_html_data_from_merge,
    make_merged_mtz_file,
    merge,
    process_merged_data,
)
from dials.algorithms.merging.reporting import generate_html_report
from dials.algorithms.scaling.scaling_library import determine_best_unit_cell
from dials.array_family import flex
from dials.util import Sorry, log, show_mail_handle_errors
from dials.util.exclude_images import (
    exclude_image_ranges_from_scans,
    get_selection_for_valid_image_ranges,
)
from dials.util.export_mtz import match_wavelengths
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files
from dials.util.version import dials_version

help_message = """
Merge scaled dials data.

Examples::

  dials.merge scaled.expt scaled.refl

  dials.merge scaled.expt scaled.refl truncate=False
"""

logger = logging.getLogger("dials")
phil_scope = phil.parse(
    """
assess_space_group = True
    .type = bool
    .help = "Option to assess space group by testing presence of axial reflections"
anomalous = True
    .type = bool
    .help = "Output anomalous as well as mean intensities."
truncate = True
    .type = bool
    .help = "Use the French & Wilson (1978) algorithm to correct for negative "
            "intensities when estimating amplitudes."
french_wilson {
    implementation = *dials cctbx
        .type = choice
        .help = "Choice of implementation of the French & Wilson algorithm"
    min_reflections = 200
        .type = int(value_min=1)
        .help = "Only perform French & Wilson procedure if at least this "
                "number of reflections."
    fallback_to_flat_prior = True
        .type = bool
        .help = "If insufficient number of reflections to perform the "
                "French & Wilson procedure, fallback to assumption of a "
                "flat prior, i.e.: "
                "  |F| = sqrt((Io+sqrt(Io**2 +2sigma**2))/2.0)"
}
d_min = None
    .type = float
    .help = "High resolution limit to apply to the data."
d_max = None
    .type = float
    .help = "Low resolution limit to apply to the data."
wavelength_tolerance = 1e-4
    .type = float(value_min=0.0)
    .help = "Absolute tolerance for determining wavelength grouping for merging."
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
best_unit_cell = None
    .type = unit_cell
    .help = "Best unit cell value, to use when performing resolution cutting,"
            "and as the overall unit cell in the merged mtz. If undefined, the median"
            "cell will be used."
n_residues = 200
    .type = int
    .help = "Number of residues to use in Wilson scaling"
merging {
    use_internal_variance = False
        .type = bool
    n_bins = 20
        .type = int(value_min=5)
    anomalous = False
        .type = bool
        .help = "Option to control whether reported merging stats are anomalous."
}
output {
    log = dials.merge.log
        .type = str
    mtz = merged.mtz
        .type = str
        .help = "Filename to use for mtz output."
    html = dials.merge.html
        .type = str
        .help = "Filename for html output report."
    json = None
        .type = str
        .help = "Filename to output data from html report in json format."
    crystal_names = XTAL
        .type = strings
        .help = "Crystal name to be used in MTZ file output (multiple names
            allowed for MAD datasets)"
    project_name = AUTOMATIC
        .type = str
        .help = "Project name to be used in MTZ file output"
    dataset_names = NATIVE
        .type = strings
        .help = "Dataset name to be used in MTZ file output (multiple names
            allowed for MAD datasets)"
    additional_stats = False
       .type = bool
       .help = "Calculate and report the R-split statistic"
}
include scope dials.util.exclude_images.phil_scope
""",
    process_includes=True,
)


def merge_data_to_mtz_with_report_collection(
    params: phil.scope_extract,
    experiments: ExperimentList,
    reflections: List[flex.reflection_table],
) -> Tuple[mtz.object, dict]:
    """Run the merge_data_to_mtz function, also collecting data for json/html output"""
    with collect_html_data_from_merge() as collector:
        mtz = merge_data_to_mtz(params, experiments, reflections)
        json_data = collector.create_json()
    return mtz, json_data


def merge_data_to_mtz(
    params: phil.scope_extract,
    experiments: ExperimentList,
    reflections: List[flex.reflection_table],
) -> mtz.object:
    """Merge data (at each wavelength) and write to an mtz file object.

    reflections can be a list containing a single reflection table, or a list
    of reflection tables.
    """
    wavelengths = match_wavelengths(
        experiments,
        absolute_tolerance=params.wavelength_tolerance,
    )  # wavelengths is an ordered dict
    mtz_datasets = [
        MTZDataClass(wavelength=w, project_name=params.output.project_name)
        for w in wavelengths.keys()
    ]
    dataset_names = params.output.dataset_names
    crystal_names = params.output.crystal_names

    # exclude any images
    if params.exclude_images:
        experiments = exclude_image_ranges_from_scans(
            reflections, experiments, params.exclude_images
        )
        reflections = [
            refl.select(get_selection_for_valid_image_ranges(refl, exp))
            for refl, exp in zip(reflections, experiments)
        ]
    # check if best_unit_cell is set.
    best_unit_cell = params.best_unit_cell
    if not best_unit_cell:
        best_unit_cell = determine_best_unit_cell(experiments)
    for table in reflections:
        table["d"] = best_unit_cell.d(table["miller_index"])
    for expt in experiments:
        expt.crystal.unit_cell = best_unit_cell

    if len(wavelengths) > 1:
        logger.info(
            "Multiple wavelengths found: \n%s",
            "\n".join(
                "  Wavlength: %.5f, experiment numbers: %s "
                % (k, ",".join(map(str, v)))
                for k, v in wavelengths.items()
            ),
        )
        if not dataset_names or len(dataset_names) != len(wavelengths):
            logger.info(
                "Unequal number of dataset names and wavelengths, using default naming."
            )
            dataset_names = [None] * len(wavelengths)
        if not crystal_names or len(crystal_names) != len(wavelengths):
            logger.info(
                "Unequal number of crystal names and wavelengths, using default naming."
            )
            crystal_names = [None] * len(wavelengths)
        experiments_subsets: List[ExperimentList] = []
        reflections_subsets: List[flex.reflection_table] = []
        for dataset, dname, cname in zip(mtz_datasets, dataset_names, crystal_names):
            dataset.dataset_name = dname
            dataset.crystal_name = cname
        for exp_nos in wavelengths.values():
            expids = [experiments[i].identifier for i in exp_nos]
            experiments_subsets.append(
                ExperimentList([experiments[i] for i in exp_nos])
            )
            reflections_subsets.append(
                flex.reflection_table.concat(
                    [r.select_on_experiment_identifiers(expids) for r in reflections]
                )
            )
    else:
        mtz_datasets[0].dataset_name = dataset_names[0]
        mtz_datasets[0].crystal_name = crystal_names[0]
        experiments_subsets: List[ExperimentList] = [experiments]
        if len(reflections) > 1:
            reflections_subsets = [flex.reflection_table.concat(reflections)]
        else:
            reflections_subsets = reflections

    # merge and truncate the data for each wavelength group
    for experimentlist, reflection_table, mtz_dataset in zip(
        experiments_subsets, reflections_subsets, mtz_datasets
    ):
        # First generate two merge_equivalents objects, collect merging stats
        merged, merged_anomalous, stats_summary = merge(
            experimentlist,
            reflection_table,
            d_min=params.d_min,
            d_max=params.d_max,
            combine_partials=params.combine_partials,
            partiality_threshold=params.partiality_threshold,
            best_unit_cell=best_unit_cell,
            anomalous=params.anomalous,
            assess_space_group=params.assess_space_group,
            n_bins=params.merging.n_bins,
            use_internal_variance=params.merging.use_internal_variance,
            show_additional_stats=params.output.additional_stats,
        )
        process_merged_data(
            params, mtz_dataset, merged, merged_anomalous, stats_summary
        )

    # pass the dataclasses to an MTZ writer to generate the mtz file and return.
    return make_merged_mtz_file(mtz_datasets)


@show_mail_handle_errors()
def run(args=None):
    """Run the merging from the command-line."""
    usage = """Usage: dials.merge scaled.refl scaled.expt [options]"""

    parser = ArgumentParser(
        usage=usage,
        read_experiments=True,
        read_reflections=True,
        phil=phil_scope,
        check_format=False,
        epilog=help_message,
    )
    params, options = parser.parse_args(args=args, show_diff_phil=False)

    if not params.input.experiments or not params.input.reflections:
        parser.print_help()
        sys.exit()

    reflections, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )

    log.config(verbosity=options.verbose, logfile=params.output.log)
    logger.info(dials_version())

    diff_phil = parser.diff_phil.as_str()
    if diff_phil != "":
        logger.info("The following parameters have been modified:\n")
        logger.info(diff_phil)

    ### Assert that all data have been scaled with dials - should only be
    # able to input one reflection table and experimentlist that are
    # matching and scaled together.

    if len(reflections) != 1:
        raise Sorry(
            """Only data scaled together as a single reflection dataset
can be processed with dials.merge"""
        )

    for k in [
        "intensity.scale.value",
        "intensity.scale.variance",
        "inverse_scale_factor",
    ]:
        if k not in reflections[0]:
            raise Sorry(
                f"""{k} not found in the reflection table.
Only scaled data can be processed with dials.merge"""
            )

    try:
        if params.output.json or params.output.html:
            mtz_file, json_data = merge_data_to_mtz_with_report_collection(
                params, experiments, reflections
            )
        else:
            mtz_file = merge_data_to_mtz(params, experiments, reflections)
            json_data = {}
    except ValueError as e:
        raise Sorry(e)

    logger.info("\nWriting reflections to %s", (params.output.mtz))
    out = StringIO()
    mtz_file.show_summary(out=out)
    logger.info(out.getvalue())
    mtz_file.write(params.output.mtz)

    if params.output.json:
        with open(params.output.json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
    if params.output.html:
        generate_html_report(json_data, params.output.html)


if __name__ == "__main__":
    run()
