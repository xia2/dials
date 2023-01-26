"""Calculations relevant to reporting."""

from __future__ import annotations

from cctbx import miller
from scitbx.array_family import flex

from dials.algorithms.scaling.scaling_library import scaled_data_as_miller_array
from dials.util.batch_handling import (
    assign_batches_to_reflections,
    calculate_batch_offsets,
    get_batch_ranges,
)


def batch_dependent_properties(batches, intensities, scales=None):
    """
    Calculate intensity properties as a function of batch.

    Manage individual calculations to ensure that all calculations are
    done on the same batches array and consistent length lists returned.

    Args:
        batches (miller array): the batch numbers for the reflections.
        intensities (miller array): the reflection intensities, with
            sigmas also present.
        scales (miller array, optional): the scale factors of the reflections.

    Returns:
        (tuple): tuple containing:
            binned_batches (list): list of batch number bins
            rmerge (list): rmerge for each bin
            isigi (list): average I/sigma for each bin
            scales (list): average scale for each bin if scales given, else None

    Raises:
        AssertionError: Raised if sigmas not present in intensities, or if
            arrays sizes do not match.
    """
    assert intensities.size() == batches.size()
    if scales:
        assert scales.size() == batches.size()
    assert intensities.sigmas() is not None
    sel = intensities.sigmas() > 0
    intensities = intensities.select(sel)
    batches = batches.select(sel)
    if scales:
        scales = scales.select(sel)

    binned_batches, rmerge = rmerge_vs_batch(intensities, batches)
    _, isigi = i_sig_i_vs_batch(intensities, batches)

    scalesvsbatch = None
    if scales:
        _, scalesvsbatch = scales_vs_batch(scales, batches)
    return binned_batches, rmerge, isigi, scalesvsbatch


def combined_table_to_batch_dependent_properties(
    combined_table, experiments, scaled_array=None
):
    """Extract batch dependent properties from a combined reflection table."""
    tables = []
    for id_ in set(combined_table["id"]).difference({-1}):
        tables.append(combined_table.select(combined_table["id"] == id_))

    return reflection_tables_to_batch_dependent_properties(
        tables, experiments, scaled_array
    )


def reflection_tables_to_batch_dependent_properties(
    reflection_tables, experiments, scaled_array=None
):
    """Extract batch dependent properties from a reflection table list."""
    offsets = calculate_batch_offsets(experiments)
    reflection_tables = assign_batches_to_reflections(reflection_tables, offsets)
    # filter bad refls and negative scales
    batches = flex.int()
    scales = flex.double()
    for r in reflection_tables:
        sel = ~r.get_flags(r.flags.bad_for_scaling, all=False)
        sel &= r["inverse_scale_factor"] > 0
        batches.extend(r["batch"].select(sel))
        scales.extend(r["inverse_scale_factor"].select(sel))
    if not scaled_array:
        scaled_array = scaled_data_as_miller_array(reflection_tables, experiments)
    ms = scaled_array.customized_copy()
    batch_array = miller.array(ms, data=batches)

    batch_ranges = get_batch_ranges(experiments, offsets)
    batch_data = [{"id": i, "range": r} for i, r in enumerate(batch_ranges)]

    properties = batch_dependent_properties(
        batch_array, scaled_array, miller.array(ms, data=scales)
    )

    return properties + (batch_data,)


def rmerge_vs_batch(intensities, batches):
    """Determine batches and Rmerge values per batch."""
    assert intensities.size() == batches.size()
    intensities = intensities.map_to_asu()

    merging = intensities.merge_equivalents()
    merged_intensities = merging.array()

    perm = flex.sort_permutation(batches.data())
    batches = batches.data().select(perm)
    intensities = intensities.select(perm)

    pairs = miller.match_multi_indices(
        merged_intensities.indices(), intensities.indices()
    ).pairs()

    def r_merge_per_batch(pairs):
        """Calculate R_merge for the list of (merged-I, I) pairs."""

        merged_indices, unmerged_indices = zip(*pairs)

        unmerged_Ij = intensities.data().select(flex.size_t(unmerged_indices))
        merged_Ij = merged_intensities.data().select(flex.size_t(merged_indices))

        numerator = flex.sum(flex.abs(unmerged_Ij - merged_Ij))
        denominator = flex.sum(unmerged_Ij)

        if denominator > 0:
            return numerator / denominator
        return 0

    return _batch_bins_and_data(batches, pairs, function_to_apply=r_merge_per_batch)


def i_sig_i_vs_batch(intensities, batches):
    """Determine batches and I/sigma values per batch."""
    assert intensities.size() == batches.size()
    assert intensities.sigmas() is not None
    sel = intensities.sigmas() > 0

    i_sig_i = intensities.data().select(sel) / intensities.sigmas().select(sel)
    batches = batches.select(sel)

    perm = flex.sort_permutation(batches.data())
    batches = batches.data().select(perm)
    i_sig_i = i_sig_i.select(perm)

    return _batch_bins_and_data(batches, i_sig_i, function_to_apply=flex.mean)


def scales_vs_batch(scales, batches):
    """Determine batches and scale values per batch."""
    assert scales.size() == batches.size()

    perm = flex.sort_permutation(batches.data())
    batches = batches.data().select(perm)
    scales = scales.data().select(perm)

    return _batch_bins_and_data(batches, scales, function_to_apply=flex.mean)


def _batch_bins_and_data(batches, values, function_to_apply):
    """Apply function to the data from each batch.

    Return the list of the batch bins and the value for each bin.
    """
    batch_bins = []
    data = []

    i_batch_start = 0
    current_batch = flex.min(batches)
    n_ref = batches.size()
    for i_ref in range(n_ref + 1):
        if i_ref == n_ref or batches[i_ref] != current_batch:
            assert batches[i_batch_start:i_ref].all_eq(current_batch)
            values_for_batch = values[i_batch_start:i_ref]
            data.append(function_to_apply(values_for_batch))
            batch_bins.append(current_batch)
            i_batch_start = i_ref
            if i_ref < n_ref:
                current_batch = batches[i_batch_start]
    return batch_bins, data


formats = {
    "High resolution limit": " %7.2f",
    "Low resolution limit": " %7.2f",
    "Completeness": "%7.1f",
    "Multiplicity": "%7.1f",
    "I/sigma": "%7.1f",
    "Rmerge(I)": "%7.3f",
    "Rmerge(I+/-)": "%7.3f",
    "Rmeas(I)": "%7.3f",
    "Rmeas(I+/-)": "%7.3f",
    "Rpim(I)": "%7.3f",
    "Rpim(I+/-)": "%7.3f",
    "Rsplit(I)": "%7.3f",
    "CC half": "%7.3f",
    "Wilson B factor": "%7.3f",
    "Partial bias": "%7.3f",
    "Anomalous completeness": "%7.1f",
    "Anomalous multiplicity": "%7.1f",
    "Anomalous correlation": "%7.3f",
    "Anomalous slope": "%7.3f",
    "dF/F": "%7.3f",
    "dI/s(dI)": "%7.3f",
    "Total observations": "%7d",
    "Total unique": "%7d",
}


def format_statistics(statistics, caption=None):
    """Format for printing statistics from data processing"""

    available = list(statistics.keys())

    columns = len(statistics.get("Completeness", [1, 2, 3]))
    if caption:
        result = caption.ljust(44)
    else:
        result = "".ljust(44)
    if columns == 3:
        result += " Overall    Low     High"
    elif columns == 4:
        result += "Suggested   Low    High  Overall"
    result += "\n"

    for k, format_str in formats.items():
        if k in available:
            try:
                row_data = statistics[k]
                if (
                    columns == 4 and len(row_data) == 1
                ):  # place value in leftmost overall/suggest column
                    row_data = row_data + [None] * (columns - 1)
                row_format = [format_str] + [format_str.strip()] * (len(row_data) - 1)
                formatted = " ".join(
                    (f % k) if k is not None else (" " * len(f % 0))
                    for f, k in zip(row_format, row_data)
                )
            except TypeError:
                formatted = "(error)"
            result += k.ljust(44) + formatted + "\n"

    return result


def table_1_summary(
    merging_statistics,
    anomalous_statistics=None,
    selected_statistics=None,
    selected_anomalous_statistics=None,
    Wilson_B_iso=None,
):
    """Make a summary table of the merging statistics."""
    text = "\n            -------------Summary of merging statistics--------------           \n\n"
    stats = table_1_stats(
        merging_statistics,
        anomalous_statistics,
        selected_statistics,
        selected_anomalous_statistics,
        Wilson_B_iso,
    )
    text += format_statistics(stats)
    return text


def table_1_stats(
    merging_statistics,
    anomalous_statistics=None,
    selected_statistics=None,
    selected_anomalous_statistics=None,
    Wilson_B_iso=None,
):
    """Extract a statistics dict from merging stats objects"""
    key_to_var = {
        "I/sigma": "i_over_sigma_mean",
        "Completeness": "completeness",
        "Low resolution limit": "d_max",
        "Multiplicity": "mean_redundancy",
        "Rmerge(I)": "r_merge",
        "Rmeas(I)": "r_meas",
        "High resolution limit": "d_min",
        "Total observations": "n_obs",
        "Rpim(I)": "r_pim",
        "CC half": "cc_one_half",
        "Total unique": "n_uniq",
    }
    extra_key_to_var = {}
    if merging_statistics.r_split:
        extra_key_to_var.update(
            {
                "Rsplit(I)": {
                    "overall": "r_split",
                    "binned": "r_split_binned",
                }
            }
        )

    anom_key_to_var = {
        "Rmerge(I+/-)": "r_merge",
        "Rpim(I+/-)": "r_pim",
        "Rmeas(I+/-)": "r_meas",
        "Anomalous completeness": "anom_completeness",
        "Anomalous correlation": "anom_half_corr",
        "Anomalous multiplicity": "mean_redundancy",
    }

    four_column_output = bool(selected_statistics)

    stats = {}

    def generate_stats(d, r, s, e=None):
        if e is None:
            e = {}
        for key, value in d.items():
            if four_column_output:
                values = (
                    getattr(s.overall, value) if s else None,
                    getattr(s.bins[0], value) if s else None,
                    getattr(s.bins[-1], value) if s else None,
                    getattr(r.overall, value) if r else None,
                )
            else:
                values = (
                    getattr(r.overall, value) if r else None,
                    getattr(r.bins[0], value) if r else None,
                    getattr(r.bins[-1], value) if r else None,
                )
            if "completeness" in value:
                values = [v_ * 100 if v_ else None for v_ in values]
            if values[0] is not None:
                stats[key] = values
        for key, value in e.items():
            if four_column_output:
                values = (
                    getattr(s, value["overall"]),
                    getattr(s, value["binned"])[0],
                    getattr(s, value["binned"])[-1],
                    getattr(r, value["overall"]),
                )
            else:
                values = (
                    getattr(r, value["overall"]),
                    getattr(r, value["binned"])[0],
                    getattr(r, value["binned"])[-1],
                )
            if values[0] is not None:
                stats[key] = values

    generate_stats(
        key_to_var, merging_statistics, selected_statistics, extra_key_to_var
    )
    if Wilson_B_iso:
        stats["Wilson B factor"] = [Wilson_B_iso]

    if anomalous_statistics:
        anom_probability_plot = (
            anomalous_statistics.overall.anom_probability_plot_expected_delta
        )
        if anom_probability_plot is not None:
            stats["Anomalous slope"] = [anom_probability_plot.slope]
        stats["dF/F"] = [anomalous_statistics.overall.anom_signal]
        stats["dI/s(dI)"] = [
            anomalous_statistics.overall.delta_i_mean_over_sig_delta_i_mean
        ]
        if selected_anomalous_statistics:
            anom_probability_plot = (
                selected_anomalous_statistics.overall.anom_probability_plot_expected_delta
            )
            if anom_probability_plot is not None:
                stats["Anomalous slope"] = [anom_probability_plot.slope]
            stats["dF/F"] = [selected_anomalous_statistics.overall.anom_signal]
            stats["dI/s(dI)"] = [
                selected_anomalous_statistics.overall.delta_i_mean_over_sig_delta_i_mean
            ]
        generate_stats(
            anom_key_to_var, anomalous_statistics, selected_anomalous_statistics
        )

    return stats


def make_merging_statistics_summary(dataset_statistics):
    """Format merging statistics information into an output string."""

    text = "\n            ----------Merging statistics by resolution bin----------           \n\n"
    r_split_vals = []

    # 14/15 columns in table, need to know max size of the values string in each col to reformat
    names = ["d_max", "d_min", "#obs", "#uniq", "mult.", f"{'%'}comp", "<I>"] + [
        "<I/sI>",
        "r_mrg",
        "r_meas",
        "r_pim",
        "r_anom",
        "cc1/2",
        "cc_ano",
    ]
    if (
        hasattr(dataset_statistics, "r_split")
        and dataset_statistics.r_split is not None
    ):
        r_split_vals = list(dataset_statistics.r_split_binned) + [
            dataset_statistics.r_split
        ]
        names.insert(-2, "r_splt")
    n_headers = len(names)  # 14 or 15
    vals = [[] for _ in range(n_headers)]
    for i, stats in enumerate(
        list(dataset_statistics.bins) + [dataset_statistics.overall]
    ):
        txt = stats.format().split()
        if r_split_vals:
            txt.insert(-2, f"{r_split_vals[i]:.3f}")
            if len(txt) < n_headers:
                txt.insert(-3, "0.000")  # handle null r_anom
        else:
            if len(txt) < n_headers:
                txt.insert(-2, "0.000")  # handle null r_anom
        assert len(txt) == n_headers, "Mismatch between number of headers and values"
        for j, t in enumerate(txt):
            vals[j].append(t)

    max_lengths = [max(len(s) for s in col) for col in vals]
    hasasterisk = [any("*" in v for v in col) for col in vals]
    header = ""
    min_separator = 2  # min whitespace between values in a colum

    # First format the header, adjusting max length if necessary.
    for i, (name, ml, hasaster) in enumerate(zip(names, max_lengths, hasasterisk)):
        width_this = max(ml, len(name))  # longest of name or max value length
        max_lengths[i] = width_this  # update in case name longer
        if hasaster:  # i.e. if any of the values in the column have an asterisk
            header += (
                name.rjust(width_this + min_separator - 1) + " "
            )  # align with last value not asterisk
        else:
            header += name.rjust(width_this + min_separator)

    # Now write the values into the rows with correct separation
    rows = ["" for _ in range(len(dataset_statistics.bins) + 1)]  # +1 for overall
    for ml, col, hasaster in zip(max_lengths, vals, hasasterisk):
        for i, value in enumerate(col):
            if hasaster:  # i.e. if any of the values in the column have an asterisk
                if "*" in value:
                    rows[i] += value.rjust(ml + min_separator)
                else:
                    rows[i] += value.rjust(ml + min_separator - 1) + " "
            else:
                rows[i] += value.rjust(ml + min_separator)
    rows.insert(0, header)
    text = "\n".join(r for r in rows) + "\n\n"

    return (
        "\n            ----------Merging statistics by resolution bin----------           \n\n"
        + text
    )
