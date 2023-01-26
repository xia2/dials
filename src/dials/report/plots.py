"""
This module defines a number of general plots, which may be relevant to
for reports of several programs.
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Type

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from cctbx import uctbx
from dxtbx import flumpy
from iotbx.merging_statistics import dataset_statistics
from libtbx.utils import Sorry
from mmtbx.scaling import printed_output
from mmtbx.scaling.absolute_scaling import expected_intensity, scattering_information
from mmtbx.scaling.matthews import matthews_rupp
from scitbx.array_family import flex

logger = logging.getLogger("dials")


def make_image_range_table(experiments, batch_manager):
    """Make a summary table of image ranges."""
    table = [
        [
            "Experiment number",
            "scan image range",
            "image range in use",
            "associated batch range",
            "Image template",
        ]
    ]
    for i, exp in enumerate(experiments):
        if exp.scan:
            valid_image_ranges = ",".join(
                str(j) for j in exp.scan.get_valid_image_ranges(exp.identifier)
            )
            image_range = exp.scan.get_image_range()
            template = exp.imageset.get_template()
            b_0 = batch_manager._batch_increments[i]
            batch_params = batch_manager.batch_params[i]
            batch_range = batch_params["range"]
            batches = (b_0, b_0 + (batch_range[1] - batch_range[0]))
            table.append(
                [
                    str(batch_params["id"]),
                    image_range,
                    valid_image_ranges,
                    batches,
                    template,
                ]
            )
    return table


def scale_rmerge_vs_batch_plot(batch_manager, rmerge_vs_b, scales_vs_b=None):
    reduced_batches = batch_manager.reduced_batches
    shapes, annotations, text = batch_manager.batch_plot_shapes_and_annotations()
    if len(annotations) > 30:
        # at a certain point the annotations become unreadable
        annotations = None

    return {
        "scale_rmerge_vs_batch": {
            "data": [
                (
                    {
                        "x": reduced_batches,
                        "y": scales_vs_b,
                        "type": "scatter",
                        "name": "Scale",
                        "opacity": 0.75,
                        "text": text,
                    }
                    if scales_vs_b is not None
                    else {}
                ),
                {
                    "x": reduced_batches,
                    "y": rmerge_vs_b,
                    "yaxis": "y2",
                    "type": "scatter",
                    "name": "R<sub>merge</sub>",
                    "opacity": 0.75,
                    "text": text,
                },
            ],
            "layout": {
                "title": "Scale and R<sub>merge</sub> vs batch",
                "xaxis": {"title": "N"},
                "yaxis": {"title": "Scale", "rangemode": "tozero"},
                "yaxis2": {
                    "title": "R<sub>merge</sub>",
                    "overlaying": "y",
                    "side": "right",
                    "rangemode": "tozero",
                },
                "shapes": shapes,
                "annotations": annotations,
            },
        }
    }


def i_over_sig_i_vs_batch_plot(batch_manager, i_sig_i_vs_batch):

    reduced_batches = batch_manager.reduced_batches
    shapes, annotations, text = batch_manager.batch_plot_shapes_and_annotations()
    if len(annotations) > 30:
        # at a certain point the annotations become unreadable
        annotations = None

    return {
        "i_over_sig_i_vs_batch": {
            "data": [
                {
                    "x": reduced_batches,
                    "y": i_sig_i_vs_batch,
                    "type": "scatter",
                    "name": "I/sigI vs batch",
                    "opacity": 0.75,
                    "text": text,
                }
            ],
            "layout": {
                "title": "<I/σ(I)> vs batch",
                "xaxis": {"title": "N"},
                "yaxis": {"title": "<I/σ(I)>", "rangemode": "tozero"},
                "shapes": shapes,
                "annotations": annotations,
            },
        }
    }


def i_over_sig_i_vs_i_plot(intensities, sigmas, label=None):
    """Plot unscaled I / sigma_adjusted vs unscaled I."""
    sel = (intensities > 0) & (sigmas > 0)
    intensities = intensities.select(sel)
    sigmas = sigmas.select(sel)
    x = flex.log10(intensities)
    y = intensities / sigmas

    H, xedges, yedges = np.histogram2d(
        x.as_numpy_array(), y.as_numpy_array(), bins=(200, 200)
    )
    nonzeros = np.nonzero(H)
    z = np.empty(H.shape)
    z[:] = np.NAN
    z[nonzeros] = H[nonzeros]
    key = f"i_over_sig_i_vs_i_{label}" if label is not None else "i_over_sig_i_vs_i"
    title = "I/σ(I) vs I"
    title = title + f" (error model {label})" if label is not None else title
    return {
        key: {
            "data": [
                {
                    "x": xedges.tolist(),
                    "y": yedges.tolist(),
                    "z": z.transpose().tolist(),
                    "type": "heatmap",
                    "name": "Isigma distribution",
                    "colorbar": {
                        "title": "Number of reflections",
                        "titleside": "right",
                    },
                    "colorscale": "Viridis",
                }
            ],
            "layout": {
                "title": title,
                "xaxis": {"title": "log I"},
                "yaxis": {"title": "I/σ(I)"},
            },
            "help": """\
This plot shows the distribution of I/σ(I) as a function of I, which can
give indication of the errors within the dataset. The I/σ(I) asymptotic
limit can be seen at the plateau in the top-right of the plot, if the measured
data are strong enough.

[1] Diederichs, K. (2010). Acta Cryst. D, 66(6), 733-740.
https://doi.org/10.1107/S0907444910014836
""",
        }
    }


def d_star_sq_to_d_ticks(d_star_sq, nticks):
    min_d_star_sq = min(d_star_sq)
    dstep = (max(d_star_sq) - min_d_star_sq) / nticks
    tickvals = [min_d_star_sq + (i * dstep) for i in range(nticks)]
    ticktext = [f"{uctbx.d_star_sq_as_d(dsq):.2f}" for dsq in tickvals]
    return tickvals, ticktext


class _xtriage_output(printed_output):
    def __init__(self, out):
        super().__init__(out)
        self.gui_output = True
        self._out_orig = self.out
        self.out = StringIO()
        self._sub_header_to_out = {}

    def show_big_header(self, text):
        pass

    def show_header(self, text):
        self._out_orig.write(self.out.getvalue())
        self.out = StringIO()
        super().show_header(text)

    def show_sub_header(self, title):
        self._out_orig.write(self.out.getvalue())
        self.out = StringIO()
        self._current_sub_header = title
        assert title not in self._sub_header_to_out
        self._sub_header_to_out[title] = self.out

    def flush(self):
        self._out_orig.write(self.out.getvalue())
        self.out.flush()
        self._out_orig.flush()


def xtriage_output(xanalysis):

    with StringIO() as xs:
        xout = _xtriage_output(xs)
        try:
            xanalysis.show(out=xout)
            sub_header_to_out = xout._sub_header_to_out
            issues = xanalysis.summarize_issues()
        except Exception:
            return {}
        finally:
            xout.flush()

    xtriage_success = []
    xtriage_warnings = []
    xtriage_danger = []

    for level, text, sub_header in issues._issues:
        with StringIO() as tmp_out:
            summary = sub_header_to_out.get(sub_header, tmp_out).getvalue()
        d = {"level": level, "text": text, "summary": summary, "header": sub_header}
        if level == 0:
            xtriage_success.append(d)
        elif level == 1:
            xtriage_warnings.append(d)
        elif level == 2:
            xtriage_danger.append(d)

    return {
        "xtriage_success": xtriage_success,
        "xtriage_warnings": xtriage_warnings,
        "xtriage_danger": xtriage_danger,
    }


class IntensityStatisticsPlots:
    """Generate plots for intensity-derived statistics."""

    def __init__(
        self,
        intensities,
        anomalous=False,
        n_resolution_bins=20,
        xtriage_analyses=None,
        run_xtriage_analysis=True,
    ):
        self.n_bins = n_resolution_bins
        self._xanalysis = xtriage_analyses
        if anomalous:
            intensities = intensities.as_anomalous_array()
        intensities.setup_binner(n_bins=self.n_bins)
        merged = intensities.merge_equivalents()
        self.binner = intensities.binner()
        self.merged_intensities = merged.array()
        self.multiplicities = merged.redundancies().complete_array(new_data_value=0)
        intensities.setup_binner_d_star_sq_step(auto_binning=True)
        self.wilson_plot_result = intensities.wilson_plot(use_binning=True)
        mr = matthews_rupp(intensities.crystal_symmetry(), out=StringIO())
        self.n_residues = mr.n_residues
        if not self._xanalysis and run_xtriage_analysis:
            # imports needed here or won't work, unsure why.
            from mmtbx.scaling.xtriage import master_params as xtriage_master_params
            from mmtbx.scaling.xtriage import xtriage_analyses

            xtriage_params = xtriage_master_params.fetch(sources=[]).extract()
            xtriage_params.scaling.input.xray_data.skip_sanity_checks = True
            try:
                self._xanalysis = xtriage_analyses(
                    miller_obs=self.merged_intensities,
                    unmerged_obs=intensities,
                    text_out="silent",
                    params=xtriage_params,
                )
            except (RuntimeError, Sorry):
                logger.warning("Xtriage analysis failed.", exc_info=True)
                self._xanalysis = None

    def generate_xtriage_output(self):
        if not self._xanalysis:
            return {}
        return xtriage_output(self._xanalysis)

    def generate_resolution_dependent_plots(self):
        d = self.second_moments_plot()
        d.update(self.wilson_plot())
        return d

    def generate_miscellanous_plots(self):
        d = self.cumulative_intensity_distribution_plot()
        d.update(self.l_test_plot())
        d.update(self.multiplicity_histogram())
        return d

    def multiplicity_histogram(self):
        """Generate histogram data for acentric and centric multiplicities."""
        mult_acentric = self.multiplicities.select_acentric().data()
        mult_centric = self.multiplicities.select_centric().data()

        multiplicities_acentric = {}
        multiplicities_centric = {}

        for x in sorted(set(mult_acentric)):
            multiplicities_acentric[x] = mult_acentric.count(x)
        for x in sorted(set(mult_centric)):
            multiplicities_centric[x] = mult_centric.count(x)

        return {
            "multiplicities": {
                "data": [
                    {
                        "x": list(multiplicities_acentric.keys()),
                        "y": list(multiplicities_acentric.values()),
                        "type": "bar",
                        "name": "Acentric",
                        "opacity": 0.75,
                    },
                    {
                        "x": list(multiplicities_centric.keys()),
                        "y": list(multiplicities_centric.values()),
                        "type": "bar",
                        "name": "Centric",
                        "opacity": 0.75,
                    },
                ],
                "layout": {
                    "title": "Distribution of multiplicities",
                    "xaxis": {"title": "Multiplicity"},
                    "yaxis": {
                        "title": "Frequency",
                        # 'rangemode': 'tozero'
                    },
                    "bargap": 0,
                    "barmode": "overlay",
                },
            }
        }

    def wilson_plot(self):
        if not self._xanalysis or not self._xanalysis.wilson_scaling:
            return {}

        dstarsq = self.wilson_plot_result.binner.bin_centers(2)
        observed = self.wilson_plot_result.data[1:-1]
        # The binning of the wilson plot can result in some bins with 'None' values
        if None in observed:
            observed = [i for i in observed if i is not None]
            dstarsq = flex.double(
                [i for i, j in zip(dstarsq, observed) if j is not None]
            )
        if not observed:
            return {}
        expected = expected_intensity(
            scattering_information(n_residues=self.n_residues),
            dstarsq,
            b_wilson=self._xanalysis.iso_b_wilson,
            p_scale=self._xanalysis.wilson_scaling.iso_p_scale,
        )

        x1 = observed
        x2 = expected.mean_intensity
        # ignore the start and end of the plot, which may be unreliable
        if len(x1) > 10:
            x1 = x1[1:-3]
            x2 = x2[1:-3]

        def residuals(k):
            """Calculate the residual for an overall scale factor"""
            return x1 - k * x2

        best = least_squares(residuals, 1.0)

        mean_I_obs_theory = expected.mean_intensity * best.x
        tickvals_wilson, ticktext_wilson = d_star_sq_to_d_ticks(dstarsq, nticks=5)

        return {
            "wilson_intensity_plot": {
                "data": (
                    [
                        {
                            "x": list(dstarsq),
                            "y": list(observed),
                            "type": "scatter",
                            "name": "Observed",
                        },
                        {
                            "x": list(dstarsq),
                            "y": list(mean_I_obs_theory),
                            "type": "scatter",
                            "name": "Expected",
                        },
                    ]
                ),
                "layout": {
                    "title": "Wilson intensity plot",
                    "xaxis": {
                        "title": "Resolution (Å)",
                        "tickvals": tickvals_wilson,
                        "ticktext": ticktext_wilson,
                    },
                    "yaxis": {"type": "log", "title": "Mean(I)", "rangemode": "tozero"},
                },
            }
        }

    def cumulative_intensity_distribution_plot(self):
        if not self._xanalysis or not self._xanalysis.twin_results:
            return {}
        nz_test = self._xanalysis.twin_results.nz_test
        return {
            "cumulative_intensity_distribution": {
                "data": [
                    {
                        "x": list(nz_test.z),
                        "y": list(nz_test.ac_obs),
                        "type": "scatter",
                        "name": "Acentric observed",
                        "mode": "lines",
                        "line": {"color": "rgb(31, 119, 180)"},
                    },
                    {
                        "x": list(nz_test.z),
                        "y": list(nz_test.c_obs),
                        "type": "scatter",
                        "name": "Centric observed",
                        "mode": "lines",
                        "line": {"color": "rgb(255, 127, 14)"},
                    },
                    {
                        "x": list(nz_test.z),
                        "y": list(nz_test.ac_untwinned),
                        "type": "scatter",
                        "name": "Acentric theory",
                        "mode": "lines",
                        "line": {"color": "rgb(31, 119, 180)", "dash": "dot"},
                        "opacity": 0.8,
                    },
                    {
                        "x": list(nz_test.z),
                        "y": list(nz_test.c_untwinned),
                        "type": "scatter",
                        "name": "Centric theory",
                        "mode": "lines",
                        "line": {"color": "rgb(255, 127, 14)", "dash": "dot"},
                        "opacity": 0.8,
                    },
                ],
                "layout": {
                    "title": "Cumulative intensity distribution",
                    "xaxis": {"title": "z", "range": (0, 1)},
                    "yaxis": {"title": "P(Z <= Z)", "range": (0, 1)},
                },
            }
        }

    def l_test_plot(self):
        if not self._xanalysis or not self._xanalysis.twin_results:
            return {}
        l_test = self._xanalysis.twin_results.l_test
        return {
            "l_test": {
                "data": [
                    {
                        "x": list(l_test.l_values),
                        "y": list(l_test.l_cumul_untwinned),
                        "type": "scatter",
                        "name": "Untwinned",
                        "mode": "lines",
                        "line": {"color": "rgb(31, 119, 180)", "dash": "dashdot"},
                    },
                    {
                        "x": list(l_test.l_values),
                        "y": list(l_test.l_cumul_perfect_twin),
                        "type": "scatter",
                        "name": "Perfect twin",
                        "mode": "lines",
                        "line": {"color": "rgb(31, 119, 180)", "dash": "dot"},
                        "opacity": 0.8,
                    },
                    {
                        "x": list(l_test.l_values),
                        "y": list(l_test.l_cumul),
                        "type": "scatter",
                        "name": "Observed",
                        "mode": "lines",
                        "line": {"color": "rgb(255, 127, 14)"},
                    },
                ],
                "layout": {
                    "title": "L test (Padilla and Yeates)",
                    "xaxis": {"title": "|l|", "range": (0, 1)},
                    "yaxis": {"title": "P(L >= l)", "range": (0, 1)},
                },
            }
        }

    def second_moments_plot(self):

        acentric = self.merged_intensities.select_acentric()
        centric = self.merged_intensities.select_centric()
        if acentric.size():
            acentric.setup_binner(n_bins=self.n_bins)
            second_moments_acentric = acentric.second_moment_of_intensities(
                use_binning=True
            )
        else:
            second_moments_acentric = None
        if centric.size():
            centric.setup_binner(n_bins=self.n_bins)
            second_moments_centric = centric.second_moment_of_intensities(
                use_binning=True
            )
        else:
            second_moments_centric = None

        second_moment_d_star_sq = []
        if acentric.size():
            second_moment_d_star_sq.extend(
                second_moments_acentric.binner.bin_centers(2)
            )
        if centric.size():
            second_moment_d_star_sq.extend(second_moments_centric.binner.bin_centers(2))
        tickvals_2nd_moment, ticktext_2nd_moment = d_star_sq_to_d_ticks(
            second_moment_d_star_sq, nticks=5
        )

        return {
            "second_moments": {
                "data": [
                    (
                        {
                            "x": list(
                                second_moments_acentric.binner.bin_centers(2)
                            ),  # d_star_sq
                            "y": second_moments_acentric.data[1:-1],
                            "type": "scatter",
                            "name": "<I<sup>2</sub>> acentric",
                        }
                        if acentric.size()
                        else {}
                    ),
                    (
                        {
                            "x": list(
                                second_moments_centric.binner.bin_centers(2)
                            ),  # d_star_sq
                            "y": second_moments_centric.data[1:-1],
                            "type": "scatter",
                            "name": "<I<sup>2</sub>> centric",
                        }
                        if centric.size()
                        else {}
                    ),
                ],
                "layout": {
                    "title": "Second moment of I",
                    "xaxis": {
                        "title": "Resolution (Å)",
                        "tickvals": tickvals_2nd_moment,
                        "ticktext": ticktext_2nd_moment,
                    },
                    "yaxis": {"title": "<I<sup>2</sub>>", "rangemode": "tozero"},
                },
            }
        }


class ResolutionPlotsAndStats:
    """
    Use iotbx dataset statistics objects to make plots and tables for reports.

    This class allows the generation of plots of various properties as a
    function of resolution as well as a statistics table and summary table,
    using the data from two iotbx.dataset_statistics objects, with
    anomalous=False/True.
    """

    def __init__(
        self,
        dataset_statistics: Type[dataset_statistics],
        anomalous_dataset_statistics: Type[dataset_statistics],
        is_centric=False,
    ):
        self.dataset_statistics = dataset_statistics
        self.anomalous_dataset_statistics = anomalous_dataset_statistics
        self.d_star_sq_bins = [
            0.5 * (uctbx.d_as_d_star_sq(b.d_max) + uctbx.d_as_d_star_sq(b.d_min))
            for b in self.dataset_statistics.bins
        ]
        self.d_star_sq_tickvals, self.d_star_sq_ticktext = d_star_sq_to_d_ticks(
            self.d_star_sq_bins, nticks=5
        )
        self.is_centric = is_centric

    def make_all_plots(self, cc_one_half_method=None):
        """Make a dictionary containing all available resolution-dependent plots."""
        d = self.cc_one_half_plot(method=cc_one_half_method)
        d.update(self.i_over_sig_i_plot())
        d.update(self.completeness_plot())
        d.update(self.multiplicity_vs_resolution_plot())
        d.update(self.r_pim_plot())
        d.update(self.additional_stats_plot())
        return d

    def additional_stats_plot(self):
        d = {}
        # 'binner' attribute only exists for ExtendedDatasetStatistics, make sure
        # this function also works with regular iotbx dataset_statistics with this check.
        if (
            not hasattr(self.dataset_statistics, "binner")
            or not self.dataset_statistics.binner
        ):
            return d
        d_star_sq_bins = []
        for bin in self.dataset_statistics.binner.range_used():
            d_max_min = self.dataset_statistics.binner.bin_d_range(bin)
            d_star_sq_bins.append(
                0.5
                * (
                    uctbx.d_as_d_star_sq(d_max_min[0])
                    + uctbx.d_as_d_star_sq(d_max_min[1])
                )
            )
        d_star_sq_tickvals, d_star_sq_ticktext = d_star_sq_to_d_ticks(
            d_star_sq_bins, nticks=5
        )
        if self.dataset_statistics.r_split:

            d.update(
                {
                    "r_split": {
                        "data": [
                            {
                                "x": d_star_sq_bins,  # d_star_sq
                                "y": self.dataset_statistics.r_split_binned,
                                "type": "scatter",
                                "name": "R<sub>split</sub> vs resolution",
                            }
                        ],
                        "layout": {
                            "title": "R<sub>split</sub> vs resolution",
                            "xaxis": {
                                "title": "Resolution (Å)",
                                "tickvals": d_star_sq_tickvals,
                                "ticktext": d_star_sq_ticktext,
                            },
                            "yaxis": {
                                "title": "R<sub>split</sub>",
                                "rangemode": "tozero",
                            },
                        },
                    }
                }
            )
        return d

    def cc_one_half_plot(self, method=None):
        """Make a plot of cc half against resolution."""

        if method == "sigma_tau":
            cc_one_half_bins = [
                bin_stats.cc_one_half_sigma_tau
                if bin_stats.cc_one_half_sigma_tau
                else 0.0
                for bin_stats in self.dataset_statistics.bins
            ]
            cc_one_half_critical_value_bins = [
                bin_stats.cc_one_half_sigma_tau_critical_value
                if bin_stats.cc_one_half_sigma_tau_critical_value
                else 0.0
                for bin_stats in self.dataset_statistics.bins
            ]
        else:
            cc_one_half_bins = [
                bin_stats.cc_one_half if bin_stats.cc_one_half else 0.0
                for bin_stats in self.dataset_statistics.bins
            ]
            cc_one_half_critical_value_bins = [
                bin_stats.cc_one_half_critical_value
                if bin_stats.cc_one_half_critical_value
                else 0.0
                for bin_stats in self.dataset_statistics.bins
            ]
        cc_anom_bins = [
            bin_stats.cc_anom if bin_stats.cc_anom else 0.0
            for bin_stats in self.dataset_statistics.bins
        ]
        cc_anom_critical_value_bins = [
            bin_stats.cc_anom_critical_value
            if bin_stats.cc_anom_critical_value
            else 0.0
            for bin_stats in self.dataset_statistics.bins
        ]

        return {
            "cc_one_half": cc_half_plot(
                d_star_sq=self.d_star_sq_bins,
                cc_half=cc_one_half_bins,
                cc_anom=cc_anom_bins if not self.is_centric else None,
                cc_half_critical_values=cc_one_half_critical_value_bins,
                cc_anom_critical_values=cc_anom_critical_value_bins
                if not self.is_centric
                else None,
                cc_half_fit=None,
                d_min=None,
            )
        }

    def i_over_sig_i_plot(self):
        """Make a plot of <I/sigI> against resolution."""
        i_over_sig_i_bins = [
            bin_stats.i_over_sigma_mean for bin_stats in self.dataset_statistics.bins
        ]

        return {
            "i_over_sig_i": {
                "data": [
                    {
                        "x": self.d_star_sq_bins,  # d_star_sq
                        "y": i_over_sig_i_bins,
                        "type": "scatter",
                        "name": "I/sigI vs resolution",
                    }
                ],
                "layout": {
                    "title": "<I/σ(I)> vs resolution",
                    "xaxis": {
                        "title": "Resolution (Å)",
                        "tickvals": self.d_star_sq_tickvals,
                        "ticktext": self.d_star_sq_ticktext,
                    },
                    "yaxis": {"title": "<I/σ(I)>", "rangemode": "tozero"},
                },
            }
        }

    def r_pim_plot(self):
        """Make a plot of <I/sigI> against resolution."""
        r_pim_bins = [bin_stats.r_pim for bin_stats in self.dataset_statistics.bins]

        return {
            "r_pim": {
                "data": [
                    {
                        "x": self.d_star_sq_bins,  # d_star_sq
                        "y": r_pim_bins,
                        "type": "scatter",
                        "name": "R<sub>pim</sub> vs resolution",
                    }
                ],
                "layout": {
                    "title": "R<sub>pim</sub> vs resolution",
                    "xaxis": {
                        "title": "Resolution (Å)",
                        "tickvals": self.d_star_sq_tickvals,
                        "ticktext": self.d_star_sq_ticktext,
                    },
                    "yaxis": {"title": "R<sub>pim</sub>", "rangemode": "tozero"},
                },
            }
        }

    def completeness_plot(self):
        """Make a plot of completeness against resolution."""
        completeness_bins = [
            bin_stats.completeness for bin_stats in self.dataset_statistics.bins
        ]
        if self.anomalous_dataset_statistics:
            anom_completeness_bins = [
                bin_stats.anom_completeness
                for bin_stats in self.anomalous_dataset_statistics.bins
            ]

        return {
            "completeness": {
                "data": [
                    {
                        "x": self.d_star_sq_bins,
                        "y": completeness_bins,
                        "type": "scatter",
                        "name": "Completeness",
                    },
                    (
                        {
                            "x": self.d_star_sq_bins,
                            "y": anom_completeness_bins,
                            "type": "scatter",
                            "name": "Anomalous completeness",
                        }
                        if not self.is_centric and self.anomalous_dataset_statistics
                        else {}
                    ),
                ],
                "layout": {
                    "title": "Completeness vs resolution",
                    "xaxis": {
                        "title": "Resolution (Å)",
                        "tickvals": self.d_star_sq_tickvals,
                        "ticktext": self.d_star_sq_ticktext,
                    },
                    "yaxis": {"title": "Completeness", "range": (0, 1)},
                },
            }
        }

    def multiplicity_vs_resolution_plot(self):
        """Make a plot of multiplicity against resolution."""
        multiplicity_bins = [
            bin_stats.mean_redundancy for bin_stats in self.dataset_statistics.bins
        ]
        if self.anomalous_dataset_statistics:
            anom_multiplicity_bins = [
                bin_stats.mean_redundancy
                for bin_stats in self.anomalous_dataset_statistics.bins
            ]

        return {
            "multiplicity_vs_resolution": {
                "data": [
                    {
                        "x": self.d_star_sq_bins,
                        "y": multiplicity_bins,
                        "type": "scatter",
                        "name": "Multiplicity",
                    },
                    (
                        {
                            "x": self.d_star_sq_bins,
                            "y": anom_multiplicity_bins,
                            "type": "scatter",
                            "name": "Anomalous multiplicity",
                        }
                        if not self.is_centric and self.anomalous_dataset_statistics
                        else {}
                    ),
                ],
                "layout": {
                    "title": "Multiplicity vs resolution",
                    "xaxis": {
                        "title": "Resolution (Å)",
                        "tickvals": self.d_star_sq_tickvals,
                        "ticktext": self.d_star_sq_ticktext,
                    },
                    "yaxis": {"title": "Multiplicity", "rangemode": "tozero"},
                },
            }
        }

    def merging_statistics_table(self, cc_half_method=None):

        headers = [
            "Resolution (Å)",
            "N(obs)",
            "N(unique)",
            "Multiplicity",
            "Completeness",
            "Mean I",
            "Mean I/σ(I)",
            "R<sub>merge</sub>",
            "R<sub>meas</sub>",
            "R<sub>pim</sub>",
            "R<sub>anom</sub>",
            "CC<sub>½</sub>",
        ]
        if not self.is_centric:
            headers.append("CC<sub>ano</sub>")
        r_split_vals = []
        if (
            hasattr(self.dataset_statistics, "r_split")
            and self.dataset_statistics.r_split_binned
        ):
            headers.insert(-2, "R<sub>split</sub>")
            r_split_vals = self.dataset_statistics.r_split_binned
        rows = []

        def safe_format(format_str, item):
            return format_str % item if item is not None else ""

        for i, bin_stats in enumerate(self.dataset_statistics.bins):
            row = [
                f"{bin_stats.d_max:.2f} - {bin_stats.d_min:.2f}",
                bin_stats.n_obs,
                bin_stats.n_uniq,
                f"{bin_stats.mean_redundancy:.2f}",
                f"{100 * bin_stats.completeness:.2f}",
                f"{bin_stats.i_mean:.1f}",
                f"{bin_stats.i_over_sigma_mean:.1f}",
                safe_format("%.3f", bin_stats.r_merge),
                safe_format("%.3f", bin_stats.r_meas),
                safe_format("%.3f", bin_stats.r_pim),
                safe_format("%.3f", bin_stats.r_anom),
            ]
            if r_split_vals:
                row.append(f"{r_split_vals[i]:.3f}")
            if cc_half_method == "sigma_tau":
                row.append(
                    "%.3f%s"
                    % (
                        bin_stats.cc_one_half_sigma_tau,
                        "*" if bin_stats.cc_one_half_sigma_tau_significance else "",
                    )
                )
            else:
                row.append(
                    "%.3f%s"
                    % (
                        bin_stats.cc_one_half,
                        "*" if bin_stats.cc_one_half_significance else "",
                    )
                )

            if not self.is_centric:
                row.append(
                    "%.3f%s"
                    % (bin_stats.cc_anom, "*" if bin_stats.cc_anom_significance else "")
                )
            rows.append(row)

        merging_stats_table = [headers]
        merging_stats_table.extend(rows)

        return merging_stats_table

    def overall_statistics_table(self, cc_half_method=None):

        headers = ["", "Overall", "Low resolution", "High resolution"]

        stats = (
            self.dataset_statistics.overall,
            self.dataset_statistics.bins[0],
            self.dataset_statistics.bins[-1],
        )

        rows = [
            ["Resolution (Å)"] + [f"{s.d_max:.2f} - {s.d_min:.2f}" for s in stats],
            ["Observations"] + ["%i" % s.n_obs for s in stats],
            ["Unique reflections"] + ["%i" % s.n_uniq for s in stats],
            ["Multiplicity"] + [f"{s.mean_redundancy:.1f}" for s in stats],
            ["Completeness"] + [f"{s.completeness * 100:.2f}%" for s in stats],
            # ['Mean intensity'] + ['%.1f' %s.i_mean for s in stats],
            ["Mean I/σ(I)"] + [f"{s.i_over_sigma_mean:.1f}" for s in stats],
            ["R<sub>merge</sub>"] + [f"{s.r_merge:.3f}" for s in stats],
            ["R<sub>meas</sub>"] + [f"{s.r_meas:.3f}" for s in stats],
            ["R<sub>pim</sub>"] + [f"{s.r_pim:.3f}" for s in stats],
        ]
        if (
            hasattr(self.dataset_statistics, "r_split")
            and self.dataset_statistics.r_split is not None
        ):
            rsplits = (
                self.dataset_statistics.r_split,
                self.dataset_statistics.r_split_binned[0],
                self.dataset_statistics.r_split_binned[-1],
            )
            rows.append(["R<sub>split</sub>"] + [f"{rs:.3f}" for rs in rsplits])

        if cc_half_method == "sigma_tau":
            rows.append(
                ["CC<sub>½</sub>"] + [f"{s.cc_one_half_sigma_tau:.3f}" for s in stats]
            )
        else:
            rows.append(["CC<sub>½</sub>"] + [f"{s.cc_one_half:.3f}" for s in stats])
        rows = [[f"<strong>{r[0]}</strong>"] + r[1:] for r in rows]

        overall_stats_table = [headers]
        overall_stats_table.extend(rows)

        return overall_stats_table

    def overall_statistics_summary_data(self):
        o = self.dataset_statistics.overall
        h = self.dataset_statistics.bins[-1]
        data = {
            "Resolution range (Å)": f"{o.d_max:.2f} - {o.d_min:.2f} ({h.d_max:.2f} - {h.d_min:.2f})",
            "Completeness (%)": f"{(o.completeness * 100):.2f} ({(h.completeness * 100):.2f})",
            "Multiplicity": f"{o.mean_redundancy:.2f} ({h.mean_redundancy:.2f})",
            "CC-half": f"{o.cc_one_half:.4f} ({h.cc_one_half:.4f})",
            "I/sigma": f"{o.i_over_sigma_mean:.2f} ({h.i_over_sigma_mean:.2f})",
            "Rmerge(I)": f"{o.r_merge:.4f} ({h.r_merge:.4f})",
        }
        if self.anomalous_dataset_statistics:
            o_anom = self.anomalous_dataset_statistics.overall
            h_anom = self.anomalous_dataset_statistics.bins[-1]
            data[
                "Anomalous completeness (%)"
            ] = f"{(o_anom.anom_completeness * 100):.2f} ({(h_anom.anom_completeness * 100):.2f})"
            data[
                "Anomalous multiplicity"
            ] = f"{o_anom.mean_redundancy:.2f} ({h_anom.mean_redundancy:.2f})"
        else:
            data["Anomalous completeness (%)"] = "-"
            data["Anomalous multiplicity"] = "-"
        return data

    def statistics_tables(self):
        """Generate the overall and by-resolution tables."""
        return {
            "overall": self.overall_statistics_table(),
            "resolution_binned": self.merging_statistics_table(),
            "cc_half_significance_level": 0.01,  # Note - in dials.scale and dials.merge, this is a hardcoded value
            "overall_summary_data": self.overall_statistics_summary_data(),
        }


class AnomalousPlotter:
    def __init__(self, anomalous_array, strong_cutoff=0.0, n_bins=20):
        self.intensities_anom = anomalous_array.map_to_asu()
        self.merged = self.intensities_anom.merge_equivalents(
            use_internal_variance=False
        ).array()
        self.n_bins = n_bins
        self.strong_cutoff = strong_cutoff
        if strong_cutoff > 0.0:
            self.low_res_intensities_anom = self.intensities_anom.resolution_filter(
                d_min=strong_cutoff
            )
            self.strong_merged = self.low_res_intensities_anom.merge_equivalents(
                use_internal_variance=False
            ).array()

    def make_plots(self):
        d = {}
        if self.strong_cutoff > 0.0:
            d.update(self.del_anom_normal_plot(self.strong_merged, self.strong_cutoff))
            d.update(
                self.del_anom_scatter_plot(
                    self.low_res_intensities_anom, self.strong_cutoff
                )
            )
        else:
            d.update(self.del_anom_normal_plot(self.merged))
        d.update(self.del_anom_correlation_ratio(self.intensities_anom))
        return d

    def del_anom_correlation_ratio(self, unmerged_intensities):

        acentric = unmerged_intensities.select_acentric()
        centric = unmerged_intensities.select_centric()
        correl_ratios_acentric, correl_ratios_centric = ([], [])

        def calc_correl_ratios(data):
            correl_ratios = []
            data.setup_binner(n_bins=self.n_bins)
            for i_bin in data.binner().range_used():
                sel = data.binner().selection(i_bin)
                data_sel = data.select(sel)
                if data_sel.size() > 0:
                    arr1, arr2 = data_sel.half_dataset_anomalous_correlation(
                        return_split_datasets=1
                    )
                    dano1 = arr1.anomalous_differences().data()
                    dano2 = arr2.anomalous_differences().data()
                    if dano1.size() > 0:
                        rmsd_11 = (
                            flex.sum(flex.pow2(dano1 - dano2)) / (2.0 * dano1.size())
                        ) ** 0.5
                        rmsd_1min1 = (
                            flex.sum(flex.pow2(dano1 + dano2)) / (2.0 * dano1.size())
                        ) ** 0.5
                        if abs(rmsd_11) < 1e-6:
                            # possible for sparse data if a few dano values approx equal
                            correl_ratios.append(1e6)
                        else:
                            correl_ratios.append(rmsd_1min1 / rmsd_11)
                    else:
                        correl_ratios.append(0.0)
                else:
                    correl_ratios.append(0.0)
            return correl_ratios

        if acentric.size() > 0:
            correl_ratios_acentric = calc_correl_ratios(acentric)
            if all(list(flex.double(correl_ratios_acentric) == 0.0)):
                correl_ratios_acentric = []
            else:
                d_star_sq_acentric = acentric.binner().bin_centers(2)
                actickvals, acticktext = d_star_sq_to_d_ticks(
                    d_star_sq_acentric, nticks=5
                )
        if centric.size() > 0:
            correl_ratios_centric = calc_correl_ratios(centric)
            if all(list(flex.double(correl_ratios_centric) == 0.0)):
                correl_ratios_centric = []
            else:
                d_star_sq_centric = centric.binner().bin_centers(2)
                ctickvals, cticktext = d_star_sq_to_d_ticks(
                    d_star_sq_acentric, nticks=5
                )

        if not (correl_ratios_acentric or correl_ratios_centric):
            return {}
        if correl_ratios_acentric:
            tickvals = actickvals
            ticktext = acticktext
        else:
            tickvals = ctickvals
            ticktext = cticktext
        return {
            "anom_correl_plot": {
                "data": [
                    (
                        {
                            "x": list(d_star_sq_acentric),
                            "y": correl_ratios_acentric,
                            "type": "lines",
                            "name": "Anomalous correlation ratio (acentric)",
                        }
                        if correl_ratios_acentric
                        else {}
                    ),
                    (
                        {
                            "x": list(d_star_sq_centric),
                            "y": correl_ratios_centric,
                            "type": "lines",
                            "name": "Anomalous correlation ratio (centric)",
                        }
                        if correl_ratios_centric
                        else {}
                    ),
                ],
                "layout": {
                    "title": "Anomalous R.M.S. correlation ratio (acentric reflections)",
                    "xaxis": {
                        "title": "Resolution (Å)",
                        "tickvals": tickvals,
                        "ticktext": ticktext,
                    },
                    "yaxis": {"anchor": "x", "title": "rms correlation ratio"},
                },
                "help": """\
This plot shows the significance of the anomalous signal, as shown in the
anomalous scatter plot, by calculating the ratio of the width of the signal along
the diagonal (a measure of the anomalous signal) over the width of the signal
perpendicular to the diagonal (a measure of the error).

[1] P. Evans, Acta Cryst. (2006). D62, 72-82
https://doi.org/10.1107/S0907444905036693
""",
            }
        }

    def del_anom_scatter_plot(self, unmerged_intensities, strong_cutoff=0.0):
        """Make a scatter plot of the anomalous differences of half sets."""

        acentric = unmerged_intensities.select_acentric()
        if acentric.size() == 0:
            return {}
        arr1, arr2 = acentric.half_dataset_anomalous_correlation(
            return_split_datasets=1
        )
        dano1 = arr1.anomalous_differences()
        dano2 = arr2.anomalous_differences()
        assert dano1.indices().all_eq(dano2.indices())
        if dano1.size() == 0:
            return {}
        max_val = max(flex.max(dano1.data()), flex.max(dano2.data()))
        min_val = min(flex.min(dano1.data()), flex.min(dano2.data()))

        title = "Correlation of half-set differences"
        plotname = "anom_scatter_plot"
        if strong_cutoff > 0.0:
            title += f" (d > {strong_cutoff:.2f})"
            plotname += "_lowres"
        else:
            title += " (all data)"
        return {
            plotname: {
                "data": [
                    {
                        "x": list(dano1.data()),
                        "y": list(dano2.data()),
                        "type": "scatter",
                        "mode": "markers",
                        "size": 1,
                        "name": "half-set anomalous differences (acentrics)",
                    },
                    {
                        "x": [min_val - 1, max_val + 1],
                        "y": [min_val - 1, max_val + 1],
                        "type": "scatter",
                        "mode": "lines",
                        "name": "D1 = D2",
                        "color": "rgb(0,0,0)",
                    },
                ],
                "layout": {
                    "title": title,
                    "xaxis": {"anchor": "y", "title": "Delta I1"},
                    "yaxis": {"anchor": "x", "title": "Delta I2"},
                },
                "help": """\
This plot shows the correlation of the anomalous differences for the data divided
into two half sets. For each reflection, the I+ and I- observations are divided
into two sets, and two differences are calculated; Delta I1 = I+(1) - I-(1),
Delta I2 = I+(2) - I-(2). Perfect data would therefore have all points along
the diagonal, in reality an elliptical distribution is seen in the presence of
anomalous signal, or a spherical distribution for data with no anomalous signal.

[1] P. Evans, Acta Cryst. (2006). D62, 72-82
https://doi.org/10.1107/S0907444905036693
""",
            }
        }

    @staticmethod
    def del_anom_normal_plot(intensities, strong_cutoff=0.0):
        """Make a normal probability plot of the normalised anomalous differences."""
        diff_array = intensities.anomalous_differences()
        if not diff_array.data().size():
            return {}
        delta = diff_array.data() / diff_array.sigmas()

        n = delta.size()
        y = np.sort(flumpy.to_numpy(delta))
        d = 0.5 / n
        v = np.linspace(start=d, stop=1.0 - d, endpoint=True, num=n)
        x = norm.ppf(v)

        H, xedges, yedges = np.histogram2d(x, y, bins=(200, 200))
        nonzeros = np.nonzero(H)
        z = np.empty(H.shape)
        z[:] = np.NAN
        z[nonzeros] = H[nonzeros]

        # also make a histogram
        histy = flex.histogram(flumpy.from_numpy(y), n_slots=100)
        # make a gaussian for reference also
        n = y.size
        width = histy.slot_centers()[1] - histy.slot_centers()[0]
        gaussian = []
        from math import exp, pi

        for x in histy.slot_centers():
            gaussian.append(n * width * exp(-(x**2) / 2.0) / ((2.0 * pi) ** 0.5))

        title = "Normal probability plot of anomalous differences"
        plotname = "normal_distribution_plot"
        if strong_cutoff > 0.0:
            title += f" (d > {strong_cutoff:.2f})"
            plotname += "_lowres"
        else:
            title += " (all data)"
            plotname += "_highres"
        return {
            plotname: {
                "data": [
                    {
                        "x": xedges.tolist(),
                        "y": yedges.tolist(),
                        "z": z.transpose().tolist(),
                        "type": "heatmap",
                        "name": "normalised deviations",
                        "colorbar": {
                            "title": "Number of reflections",
                            "titleside": "right",
                        },
                        "colorscale": "Viridis",
                    },
                    {
                        "x": [-5, 5],
                        "y": [-5, 5],
                        "type": "scatter",
                        "mode": "lines",
                        "name": "z = m",
                        "color": "rgb(0,0,0)",
                    },
                ],
                "layout": {
                    "title": title,
                    "xaxis": {
                        "anchor": "y",
                        "title": "expected delta",
                        "range": [-4, 4],
                    },
                    "yaxis": {
                        "anchor": "x",
                        "title": "observed delta",
                        "range": [-5, 5],
                    },
                },
                "help": """\
    This plot shows the normalised anomalous differences, sorted in order and
    plotted against the expected order based on a normal distribution model.
    A true normal distribution of deviations would give the straight line indicated.

    [1] P. L. Howell and G. D. Smith, J. Appl. Cryst. (1992). 25, 81-86
    https://doi.org/10.1107/S0021889891010385
    [2] P. Evans, Acta Cryst. (2006). D62, 72-82
    https://doi.org/10.1107/S0907444905036693
    """,
            }
        }


def cc_half_plot(
    d_star_sq,
    cc_half,
    cc_anom=None,
    cc_half_critical_values=None,
    cc_anom_critical_values=None,
    cc_half_fit=None,
    d_min=None,
):
    d_star_sq_tickvals, d_star_sq_ticktext = d_star_sq_to_d_ticks(d_star_sq, nticks=5)
    min_y = min([cc if cc is not None else 0 for cc in cc_half] + [0])
    if cc_anom:
        min_y = min([min_y, min([cc if cc is not None else 0 for cc in cc_anom])])
    return {
        "data": [
            {
                "x": list(d_star_sq),
                "y": list(cc_half),
                "type": "scatter",
                "name": "CC<sub>½</sub>",
                "mode": "lines",
                "line": {"color": "rgb(31, 119, 180)"},
            },
            (
                {
                    "x": list(d_star_sq),
                    "y": list(cc_half_critical_values),
                    "type": "scatter",
                    "name": "CC<sub>½</sub> critical value (p=0.01)",
                    "line": {"color": "rgb(31, 119, 180)", "dash": "dot"},
                }
                if cc_half_critical_values
                else {}
            ),
            (
                {
                    "x": list(d_star_sq),
                    "y": list(cc_anom),
                    "type": "scatter",
                    "name": "CC-anom",
                    "mode": "lines",
                    "line": {"color": "rgb(255, 127, 14)"},
                }
                if cc_anom
                else {}
            ),
            (
                {
                    "x": list(d_star_sq),
                    "y": list(cc_anom_critical_values),
                    "type": "scatter",
                    "name": "CC-anom critical value (p=0.01)",
                    "mode": "lines",
                    "line": {"color": "rgb(255, 127, 14)", "dash": "dot"},
                }
                if cc_anom_critical_values
                else {}
            ),
            (
                {
                    "x": list(d_star_sq),
                    "y": list(cc_half_fit),
                    "type": "scatter",
                    "name": "CC<sub>½</sub> fit",
                    "line": {"color": "rgb(47, 79, 79)"},
                }
                if cc_half_fit
                else {}
            ),
            (
                {
                    "x": [uctbx.d_as_d_star_sq(d_min)] * 2,
                    "y": [0, 1],
                    "type": "scatter",
                    "name": f"d_min = {d_min:.2f} Å",
                    "mode": "lines",
                    "line": {"color": "rgb(169, 169, 169)", "dash": "dot"},
                }
                if d_min
                else {}
            ),
        ],
        "layout": {
            "title": "CC<sub>½</sub> vs resolution",
            "xaxis": {
                "title": "Resolution (Å)",
                "tickvals": d_star_sq_tickvals,
                "ticktext": d_star_sq_ticktext,
            },
            "yaxis": {
                "title": "CC<sub>½</sub>",
                "range": [min_y, 1],
            },
        },
        "help": """\
The correlation coefficients, CC<sub>½</sub>, between random half-datasets. A correlation
coefficient of +1 indicates good correlation, and 0 indicates no correlation.
CC<sub>½</sub> is typically close to 1 at low resolution, falling off to close to zero at
higher resolution. A typical resolution cutoff based on CC<sub>½</sub> is around 0.3-0.5.

[1] Karplus, P. A., & Diederichs, K. (2012). Science, 336(6084), 1030-1033.
    https://doi.org/10.1126/science.1218231
[2] Diederichs, K., & Karplus, P. A. (2013). Acta Cryst D, 69(7), 1215-1222.
    https://doi.org/10.1107/S0907444913001121
[3] Evans, P. R., & Murshudov, G. N. (2013). Acta Cryst D, 69(7), 1204-1214.
    https://doi.org/10.1107/S0907444913000061
""",
    }
