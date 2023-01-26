"""Merging functions for experiment lists and reflection tables."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from io import StringIO
from typing import Optional, Tuple

import numpy as np

from dxtbx.model import ExperimentList
from iotbx import mtz, phil
from mmtbx.scaling import data_statistics

from dials.algorithms.merging.reporting import (
    MergeJSONCollector,
    MergingStatisticsData,
    make_dano_table,
)
from dials.algorithms.scaling.Ih_table import (
    _reflection_table_to_iobs,
    map_indices_to_asu,
)
from dials.algorithms.scaling.scaling_library import (
    merging_stats_from_scaled_array,
    scaled_data_as_miller_array,
)
from dials.algorithms.scaling.scaling_utilities import DialsMergingStatisticsError
from dials.algorithms.symmetry.absences.run_absences_checks import (
    run_systematic_absences_checks,
)
from dials.array_family import flex
from dials.util.export_mtz import MADMergedMTZWriter, MergedMTZWriter
from dials.util.filter_reflections import filter_reflection_table

from .french_wilson import french_wilson

logger = logging.getLogger("dials")


@contextmanager
def collect_html_data_from_merge():
    try:
        html_collector = MergeJSONCollector()
        yield html_collector
    finally:
        html_collector.reset()


def prepare_merged_reflection_table(
    experiments,
    reflection_table,
    d_min=None,
    d_max=None,
    partiality_threshold=0.99,
):
    """Filter the data and prepare a reflection table with merged data."""
    if (
        "inverse_scale_factor" in reflection_table
        and "intensity.scale.value" in reflection_table
    ):
        logger.info("Performing systematic absence checks on scaled data")
        reflections = filter_reflection_table(
            reflection_table,
            intensity_choice=["scale"],
            d_min=d_min,
            d_max=d_max,
            partiality_threshold=partiality_threshold,
        )
        reflections["intensity"] = reflections["intensity.scale.value"]
        reflections["variance"] = reflections["intensity.scale.variance"]
    elif "intensity.prf.value" in reflection_table:
        logger.info(
            "Performing systematic absence checks on unscaled profile-integrated data"
        )
        reflections = filter_reflection_table(
            reflection_table,
            intensity_choice=["profile"],
            d_min=d_min,
            d_max=d_max,
            partiality_threshold=partiality_threshold,
        )
        reflections["intensity"] = reflections["intensity.prf.value"]
        reflections["variance"] = reflections["intensity.prf.variance"]
    else:
        logger.info(
            "Performing systematic absence checks on unscaled summation-integrated data"
        )
        reflections = filter_reflection_table(
            reflection_table,
            intensity_choice=["sum"],
            d_min=d_min,
            d_max=d_max,
            partiality_threshold=partiality_threshold,
        )
        reflections["intensity"] = reflections["intensity.sum.value"]
        reflections["variance"] = reflections["intensity.sum.variance"]
    # now merge
    space_group = experiments[0].crystal.get_space_group()
    reflections["asu_miller_index"] = map_indices_to_asu(
        reflections["miller_index"], space_group
    )
    reflections["inverse_scale_factor"] = flex.double(reflections.size(), 1.0)
    merged = (
        _reflection_table_to_iobs(
            reflections, experiments[0].crystal.get_unit_cell(), space_group
        )
        .merge_equivalents(use_internal_variance=False)
        .array()
    )
    merged_reflections = flex.reflection_table()
    merged_reflections["intensity"] = merged.data()
    merged_reflections["variance"] = flex.pow2(merged.sigmas())
    merged_reflections["miller_index"] = merged.indices()
    return merged_reflections


class MTZDataClass:

    """Container class (i.e. Python 3.7 dataclass) for per-wavelength mtz dataset."""

    def __init__(
        self,
        wavelength=0.0,
        project_name="AUTOMATIC",
        dataset_name="NATIVE",
        crystal_name="XTAL",
        merged_array=None,
        merged_anomalous_array=None,
        amplitudes=None,
        anomalous_amplitudes=None,
        dano=None,
        multiplicities=None,
    ):
        self.wavelength = wavelength
        self.project_name = project_name
        self.dataset_name = dataset_name
        self.crystal_name = crystal_name
        self.merged_array = merged_array
        self.merged_anomalous_array = merged_anomalous_array
        self.amplitudes = amplitudes
        self.anomalous_amplitudes = anomalous_amplitudes
        self.dano = dano
        self.multiplicities = multiplicities


def make_merged_mtz_file(mtz_datasets):
    """
    Make an mtz file object for the data, adding the date, time and program.

    For multi-wavelength data, each wavelength is added as a new crystal.

    Args:
        mtz_datasets: A list of MTZDataClass objects, one per wavelength of the
            experiment.

    Returns:
        An iotbx mtz file object.
    """

    if len(mtz_datasets) > 1:
        writer = MADMergedMTZWriter
    else:
        writer = MergedMTZWriter

    mtz_writer = writer(
        mtz_datasets[0].merged_array.space_group(),
        mtz_datasets[0].merged_array.unit_cell(),
    )

    #### Add each wavelength as a new crystal.
    for dataset in mtz_datasets:
        mtz_writer.add_crystal(
            crystal_name=dataset.crystal_name, project_name=dataset.project_name
        )
        mtz_writer.add_empty_dataset(dataset.wavelength, name=dataset.dataset_name)
        mtz_writer.add_dataset(
            dataset.merged_array,
            dataset.merged_anomalous_array,
            dataset.amplitudes,
            dataset.anomalous_amplitudes,
            dataset.dano,
            dataset.multiplicities,
        )

    return mtz_writer.mtz_file


def merge_scaled_array(
    experiments,
    scaled_array,
    anomalous=True,
    use_internal_variance=False,
    assess_space_group=False,
    n_bins=20,
    show_additional_stats=False,
):
    # assumes filtering already done and converted to combined scaled array

    # Note, merge_equivalents does not raise an error if data is unique.
    merged = scaled_array.merge_equivalents(use_internal_variance=use_internal_variance)
    merged_anom = None

    if anomalous:
        anomalous_scaled = scaled_array.as_anomalous_array()
        merged_anom = anomalous_scaled.merge_equivalents(
            use_internal_variance=use_internal_variance
        )

    # Before merge, do assessment of the space_group
    if assess_space_group:
        merged_reflections = flex.reflection_table()
        merged_reflections["intensity"] = merged.array().data()
        merged_reflections["variance"] = flex.pow2(merged.array().sigmas())
        merged_reflections["miller_index"] = merged.array().indices()
        logger.info("Running systematic absences check")
        run_systematic_absences_checks(experiments, merged_reflections)

    stats_data = MergingStatisticsData(experiments, scaled_array)

    try:
        stats, anom_stats = merging_stats_from_scaled_array(
            scaled_array,
            n_bins,
            use_internal_variance,
            additional_stats=show_additional_stats,
        )
    except DialsMergingStatisticsError as e:
        logger.error(e, exc_info=True)
    else:
        stats_data.merging_statistics_result = stats
        stats_data.anom_merging_statistics_result = anom_stats

    return merged, merged_anom, stats_data


def merge(
    experiments,
    reflections,
    d_min=None,
    d_max=None,
    combine_partials=True,
    partiality_threshold=0.4,
    best_unit_cell=None,
    anomalous=True,
    use_internal_variance=False,
    assess_space_group=False,
    n_bins=20,
    show_additional_stats=False,
):
    """
    Merge reflection table data and generate a summary of the merging statistics.

    This procedure filters the input data, merges the data (normal and optionally
    anomalous), assesses the space group symmetry and generates a summary
    of the merging statistics.

    Returns two merge_equivalents objects and a statistics summary.
    """

    logger.info("\nMerging scaled reflection data\n")
    # first filter bad reflections using dials.util.filter methods
    reflections = filter_reflection_table(
        reflections,
        intensity_choice=["scale"],
        d_min=d_min,
        d_max=d_max,
        combine_partials=combine_partials,
        partiality_threshold=partiality_threshold,
    )
    # ^ scale factor has been applied, so now set to 1.0 - okay as not
    # going to output scale factor in merged mtz.
    reflections["inverse_scale_factor"] = flex.double(reflections.size(), 1.0)
    scaled_array = scaled_data_as_miller_array(
        [reflections], experiments, best_unit_cell
    )
    return merge_scaled_array(
        experiments,
        scaled_array,
        anomalous,
        use_internal_variance,
        assess_space_group,
        n_bins,
        show_additional_stats=show_additional_stats,
    )


def show_wilson_scaling_analysis(merged_intensities, n_residues=200):
    """
    Report the wilson statistics for a merged intensity array

    Args:
        merged_intensities: A merged miller intensity array.
        n_residues: The number of residues to use for the wilson analysis.
    """
    if not merged_intensities.space_group().is_centric():
        try:
            wilson_scaling = data_statistics.wilson_scaling(
                miller_array=merged_intensities, n_residues=n_residues
            )
        except (IndexError, RuntimeError) as e:
            logger.error(
                "\n"
                "Error encountered during Wilson statistics calculation:\n"
                "Perhaps there are too few unique reflections.\n"
                "%s",
                e,
                exc_info=True,
            )
        else:
            # Divert output through logger - do with StringIO rather than
            # info_handle else get way too much whitespace in output.
            out = StringIO()
            wilson_scaling.show(out=out)
            logger.info(out.getvalue())
            return wilson_scaling.iso_b_wilson
    return None


def truncate(
    merged_intensities,
    implementation: str = "dials",
    min_reflections: int = 200,
    fallback_to_flat_prior: bool = True,
):
    """
    Perform French-Wilson truncation procedure on merged intensities.

    Args:
        merged_intensities (miller.array): A merged miller intensity array (normal or anomalous)
        implementation (str): Choice of implementation of French & Wilson algorithm, either
                              "dials" or "cctbx"
        min_reflections (int): Minimum number of reflections to perform the French & Wilson
                               procedure
        fallback_to_flat_prior (bool): Fallback to assumption of a flat, positive prior,
                                       if the number of reflections are fewer than min_reflections,
                                       i.e. |F| = sqrt((Io+sqrt(Io**2 +2sigma**2))/2.0)

    Returns:
        (tuple): tuple containing:
            amplitudes: A normal all-positive miller amplitude array
            anom_amplitudes: An anomalous all-positive amplitude array, if the
                input array has the anomalous_flag set, else None.
            dano: The array of anomalous differences, if the input array has the
                anomalous_flag set, else None.
    """
    logger.info("\nPerforming French-Wilson treatment of scaled intensities")
    out = StringIO()
    n_refl = merged_intensities.size()
    if n_refl < min_reflections and fallback_to_flat_prior:
        logger.info(
            "Insufficient reflections for French & Wilson procedure, "
            "falling back to assumption of a flat, positive prior, i.e.: "
            "  |F| = sqrt((Io+sqrt(Io**2 +2sigma**2))/2.0)"
        )
        do_french_wilson = lambda ma: ma.enforce_positive_amplitudes()
    elif n_refl < min_reflections:
        raise ValueError(
            "Insufficient reflections for French & Wilson procedure. "
            "Either set fallback_to_flat_prior=True or truncate=False."
        )
    elif implementation == "cctbx":
        do_french_wilson = lambda ma: ma.french_wilson(log=out)
    else:
        do_french_wilson = french_wilson

    if merged_intensities.anomalous_flag():
        anom_amplitudes = do_french_wilson(merged_intensities)
        n_removed = merged_intensities.size() - anom_amplitudes.size()
        assert anom_amplitudes.is_xray_amplitude_array()
        amplitudes = anom_amplitudes.as_non_anomalous_array()
        amplitudes = amplitudes.merge_equivalents(use_internal_variance=False).array()
        dano = anom_amplitudes.anomalous_differences()
    else:
        anom_amplitudes = None
        dano = None
        amplitudes = do_french_wilson(merged_intensities)
        n_removed = merged_intensities.size() - amplitudes.size()
    logger.info("Total number of rejected intensities %s", n_removed)
    logger.debug(out.getvalue())
    return amplitudes, anom_amplitudes, dano


def merge_scaled_array_to_mtz_with_report_collection(
    params: phil.scope_extract,
    experiments: ExperimentList,
    scaled_array,
    wavelength: Optional[float] = None,
) -> Tuple[mtz.object, dict]:
    if wavelength is None:
        wavelength = np.mean(
            np.array([expt.beam.get_wavelength() for expt in experiments], dtype=float)
        )
    with collect_html_data_from_merge() as collector:
        mtz_dataset = MTZDataClass(
            wavelength=wavelength,
            project_name=params.output.project_name,
            dataset_name=params.output.dataset_names[0],
            crystal_name=params.output.crystal_names[0],
        )

        merged, merged_anomalous, stats_summary = merge_scaled_array(
            experiments,
            scaled_array,
            anomalous=params.anomalous,
            assess_space_group=params.assess_space_group,
            n_bins=params.merging.n_bins,
            use_internal_variance=params.merging.use_internal_variance,
            show_additional_stats=params.output.additional_stats,
        )
        process_merged_data(
            params, mtz_dataset, merged, merged_anomalous, stats_summary
        )
        mtz = make_merged_mtz_file([mtz_dataset])
        json_data = collector.create_json()
    return mtz, json_data


def process_merged_data(params, mtz_dataset, merged, merged_anomalous, stats_summary):
    merged_array = merged.array()
    # Save the relevant data in the mtz_dataset dataclass
    # This will add the data for IMEAN/SIGIMEAN
    mtz_dataset.merged_array = merged_array
    if merged_anomalous:
        merged_anomalous_array = merged_anomalous.array()
        # This will add the data for I(+), I(-), SIGI(+), SIGI(-), N(+), N(-)
        mtz_dataset.merged_anomalous_array = merged_anomalous_array
        mtz_dataset.multiplicities = merged_anomalous.redundancies()
    else:
        merged_anomalous_array = None
        # This will add the data for N
        mtz_dataset.multiplicities = merged.redundancies()

    if params.anomalous:
        merged_intensities = merged_anomalous_array
    else:
        merged_intensities = merged_array

    anom_amplitudes = None
    if params.truncate:
        amplitudes, anom_amplitudes, dano = truncate(
            merged_intensities,
            implementation=params.french_wilson.implementation,
            min_reflections=params.french_wilson.min_reflections,
            fallback_to_flat_prior=params.french_wilson.fallback_to_flat_prior,
        )
        # This will add the data for F, SIGF
        mtz_dataset.amplitudes = amplitudes
        # This will add the data for F(+), F(-), SIGF(+), SIGF(-)
        mtz_dataset.anomalous_amplitudes = anom_amplitudes
        # This will add the data for DANO, SIGDANO
        mtz_dataset.dano = dano

    # print out analysis statistics
    try:
        B_iso = show_wilson_scaling_analysis(merged_intensities)
    except Exception as e:
        logger.info(e)
    else:
        stats_summary.Wilson_B_iso = B_iso

    if anom_amplitudes:
        logger.info(make_dano_table(anom_amplitudes))

    if stats_summary.merging_statistics_result:
        logger.info(stats_summary)

    if MergeJSONCollector.initiated:
        stats_summary.anomalous_amplitudes = anom_amplitudes
        MergeJSONCollector.data[mtz_dataset.wavelength] = stats_summary
