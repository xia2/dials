from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Type

from jinja2 import ChoiceLoader, Environment, PackageLoader

from cctbx import miller, uctbx
from dxtbx.model import ExperimentList
from iotbx.merging_statistics import dataset_statistics

from dials.algorithms.clustering import plots as cluster_plotter
from dials.algorithms.clustering.observers import uc_params_from_experiments
from dials.algorithms.scaling.observers import make_merging_stats_plots
from dials.array_family import flex
from dials.report.analysis import (
    format_statistics,
    make_merging_statistics_summary,
    table_1_stats,
    table_1_summary,
)
from dials.report.plots import d_star_sq_to_d_ticks
from dials.util import tabulate

logger = logging.getLogger("dials")


class MergeJSONCollector(object):

    initiated = False
    data = {}

    @classmethod
    def __init__(cls):
        cls.initiated = True

    @classmethod
    def reset(cls):
        cls.data = {}
        cls.initiated = False

    def create_json(self):
        return generate_json_data(self.data)


@dataclass
class MergingStatisticsData:
    experiments: ExperimentList
    scaled_miller_array: miller.array
    reflections: Optional[
        List[flex.reflection_table]
    ] = None  # only needed if using this class like a script when making batch plots
    merging_statistics_result: Optional[Type[dataset_statistics]] = None
    anom_merging_statistics_result: Optional[Type[dataset_statistics]] = None
    anomalous_amplitudes: Optional[miller.array] = None
    Wilson_B_iso: Optional[float] = None

    def __str__(self):
        if not self.merging_statistics_result:
            return ""
        stats_summary = make_merging_statistics_summary(self.merging_statistics_result)
        stats_summary += table_1_summary(
            self.merging_statistics_result,
            self.anom_merging_statistics_result,
            Wilson_B_iso=self.Wilson_B_iso,
        )
        return stats_summary

    def table_1_stats(self):
        return format_statistics(
            table_1_stats(
                self.merging_statistics_result,
                self.anom_merging_statistics_result,
                Wilson_B_iso=self.Wilson_B_iso,
            )
        )

    def __repr__(self):
        return self.__str__()


def generate_json_data(data: dict[float, MergingStatisticsData]) -> dict:
    json_data = {}
    for wl, stats in data.items():
        wl_key = f"{wl:.5f}"
        stats_plots = make_merging_stats_plots(
            stats,
            run_xtriage_analysis=True,
            make_batch_plots=False,
        )
        # remove unneeded items
        for i in ["batch_plots", "anom_plots", "image_range_tables"]:
            del stats_plots[i]
        uc_params = uc_params_from_experiments(stats.experiments)
        stats_plots["unit_cell_plots"] = cluster_plotter.plot_uc_histograms(
            uc_params, scatter_style="heatmap"
        )
        json_data[wl_key] = stats_plots
        if stats.anomalous_amplitudes:
            json_data[wl_key]["resolution_plots"].update(
                make_dano_plots({wl: stats.anomalous_amplitudes})["dF"]
            )
        if stats.merging_statistics_result:
            json_data[wl_key][
                "merging_stats"
            ] = stats.merging_statistics_result.as_dict()
            json_data[wl_key]["table_1_stats"] = stats.table_1_stats()
            if stats.anom_merging_statistics_result:
                json_data[wl_key][
                    "merging_stats_anom"
                ] = stats.anom_merging_statistics_result.as_dict()
    if len(json_data) > 1:
        # create an overall summary table
        headers = [""] + ["Wavelength " + f"{wl:.5f}" + " Å" for wl in data.keys()]
        rows = []
        first_wl = f"{list(data.keys())[0]:.5f}"
        for k, v in json_data[first_wl]["scaling_tables"][
            "overall_summary_data"
        ].items():
            rows.append([k, v])
        for wl in list(data.keys())[1:]:
            for i, v in enumerate(
                json_data[f"{wl:.5f}"]["scaling_tables"][
                    "overall_summary_data"
                ].values()
            ):
                rows[i].append(v)
        json_data["multi_data"] = {"overall_summary": [headers] + rows}
    return json_data


def dano_over_sigdano_stats(anomalous_amplitudes, n_bins=20):
    """Calculate the statistic for resolution bins and overall."""
    vals = flex.double()
    # First calculate dF/s(dF) per resolution bin
    try:
        anomalous_amplitudes.setup_binner_counting_sorted(n_bins=n_bins)
    except Exception as e:
        logger.info(f"Unable to setup binner successfully, error: {e}")
        return None, None
    else:
        resolution_bin_edges = flex.double()
        for i_bin in anomalous_amplitudes.binner().range_used():
            sel = anomalous_amplitudes.binner().selection(i_bin)
            arr = anomalous_amplitudes.select(sel)
            vals.append(dano_over_sigdano(arr))
            resolution_bin_edges.append(anomalous_amplitudes.binner().bin_d_min(i_bin))
        resolution_bin_edges.append(anomalous_amplitudes.binner().d_min())
        return vals, resolution_bin_edges


def dano_over_sigdano(anomalous_amplitudes):
    """Calculate < |F(+) - F(-)| / sigma(F(+) - F(-))> i.e. <DANO/SIGDANO>."""
    diff = anomalous_amplitudes.anomalous_differences()
    if not diff.data() or not diff.sigmas():
        return 0.0
    return flex.mean(flex.abs(diff.data()) / diff.sigmas())


def make_dano_table(anomalous_amplitudes):
    """Calculate <dano/sigdano> in resolution bins and tabulate."""
    dFsdF, resolution_bin_edges = dano_over_sigdano_stats(anomalous_amplitudes)
    if not dFsdF:
        return ""
    logger.info("Size of anomalous differences")
    header = ["d_max", "d_min", "<|ΔF|/σ(ΔF)>"]
    rows = []
    for i, dF in enumerate(dFsdF):
        rows.append(
            [
                f"{resolution_bin_edges[i]:6.2f}",
                f"{resolution_bin_edges[i+1]:6.2f}",
                f"{dF:6.3f}",
            ]
        )
    return tabulate(rows, header)


def make_dano_plots(anomalous_data):
    """
    Make dicts of data for plotting e.g. for plotly.

    Args:
        anomalous_data (dict) : A dict of (wavelength, anomalous array) data.

    Returns:
        dict: A dictionary containing the plotting data.
    """

    data = {
        "dF": {
            "dano": {
                "data": [],
                "help": """\
This plot shows the size of the anomalous differences of F relative to the uncertainties,
(<|F(+)-F(-)|/σ(F(+)-F(-))>). A value of 0.8 is indicative of pure noise, and
a suggested cutoff is when the value falls below 1.2, although these guides require
reliable sigma estimates. For further information see
https://strucbio.biologie.uni-konstanz.de/ccp4wiki/index.php?title=SHELX_C/D/E
""",
                "layout": {"title": "<|ΔF|/σ(ΔF)> vs resolution"},
            },
        },
    }

    for i, (wave, anom) in enumerate(anomalous_data.items()):
        dFsdF, resolution_bin_edges = dano_over_sigdano_stats(anom)
        if not dFsdF:
            continue
        d_star_sq_bins = [
            0.5
            * (
                uctbx.d_as_d_star_sq(resolution_bin_edges[i])
                + uctbx.d_as_d_star_sq(resolution_bin_edges[i + 1])
            )
            for i in range(0, len(resolution_bin_edges[:-1]))
        ]
        d_star_sq_tickvals, d_star_sq_ticktext = d_star_sq_to_d_ticks(
            d_star_sq_bins, nticks=5
        )
        data["dF"]["dano"]["data"].append(
            {
                "x": d_star_sq_bins,
                "y": list(dFsdF),
                "type": "scatter",
                "name": "\u03BB" + f"={wave:.4f}",
            }
        )
    if not data["dF"]["dano"]["data"]:
        return data
    data["dF"]["dano"]["data"].append(
        {
            "x": d_star_sq_bins,
            "y": [0.8] * len(d_star_sq_bins),
            "type": "scatter",
            "mode": "lines",
            "name": "random noise level",
        }
    )
    data["dF"]["dano"]["data"].append(
        {
            "x": d_star_sq_bins,
            "y": [1.2] * len(d_star_sq_bins),
            "type": "scatter",
            "mode": "lines",
            "name": "an approximate <br>threshold for a<br>resolution cutoff",
        }
    )

    data["dF"]["dano"]["layout"] = {
        "title": "<|ΔF|/σ(ΔF)> vs resolution",
        "xaxis": {
            "title": "Resolution (Å)",
            "tickvals": d_star_sq_tickvals,
            "ticktext": d_star_sq_ticktext,
        },
        "yaxis": {"title": "<|ΔF|/σ(ΔF)>", "rangemode": "tozero"},
    }
    return data


def generate_html_report(json_data, filename):
    multi_data = None
    if "multi_data" in json_data:
        multi_data = json_data.pop("multi_data")
    for wl, v in json_data.items():
        str_wl = wl.replace(".", "_")
        for plot_cat in ["resolution_plots", "misc_plots"]:
            for name in list(v[plot_cat].keys()):
                v[plot_cat][name + "_" + str_wl] = v[plot_cat].pop(name)

    loader = ChoiceLoader(
        [
            PackageLoader("dials", "templates"),
            PackageLoader("dials", "static", encoding="utf-8"),
        ]
    )
    env = Environment(loader=loader)
    template = env.get_template("merge_report.html")
    html = template.render(
        page_title="DIALS merge report",
        individual_reports=json_data,
        multi_data=multi_data,
    )
    logger.info("Writing html report to %s", filename)
    with open(filename, "wb") as f:
        f.write(html.encode("utf-8", "xmlcharrefreplace"))
