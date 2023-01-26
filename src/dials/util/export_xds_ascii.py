from __future__ import annotations

import copy
import logging
import os

import dxtbx.model
import libtbx.phil
from cctbx.miller import map_to_asu
from rstbx.cftbx.coordinate_frame_helpers import align_reference_frame
from scitbx import matrix

from dials.array_family import flex
from dials.util import Sorry
from dials.util.filter_reflections import (
    FilteringReductionMethods,
    filter_reflection_table,
)

try:
    from typing import Tuple
except ImportError:
    pass

logger = logging.getLogger(__name__)


def export_xds_ascii(integrated_data, experiment_list, params, var_model=(1, 0)):
    """Export data from integrated_data corresponding to experiment_list to
    an XDS_ASCII.HKL formatted text file."""

    if len(experiment_list) == 1:
        experiment_data = integrated_data.select(integrated_data["id"] >= 0)
        _export_experiment(
            params.xds_ascii.hklout,
            experiment_data,
            experiment_list[0],
            params,
            var_model,
        )
    else:
        for i, experiment in enumerate(experiment_list):
            experiment_data = integrated_data.select(integrated_data["id"] == i)
            name, ext = os.path.splitext(params.xds_ascii.hklout)
            filename = name + f"_{i}" + ext
            _export_experiment(filename, experiment_data, experiment, params, var_model)


def _export_experiment(
    filename: str,
    integrated_data: flex.reflection_table,
    experiment: dxtbx.model.Experiment,
    params: libtbx.phil.scope_extract,
    var_model: Tuple = (1, 0),
):
    """Export a single experiment to an XDS_ASCII.HKL format file.

    Args:
        filename: The file to write to
        integrated_data: The reflection table, pre-selected to one experiment
        experiment: The experiment list entry to export
        params: The PHIL configuration object
        var_model:
    """
    # export for xds_ascii should only be for non-scaled reflections
    assert any(
        i in integrated_data for i in ["intensity.sum.value", "intensity.prf.value"]
    )
    # Handle requesting profile intensities (default via auto) but no column
    if "profile" in params.intensity and "intensity.prf.value" not in integrated_data:
        raise Sorry(
            "Requested profile intensity data but only summed present. Use intensity=sum."
        )

    integrated_data = filter_reflection_table(
        integrated_data,
        intensity_choice=params.intensity,
        partiality_threshold=params.mtz.partiality_threshold,
        combine_partials=params.mtz.combine_partials,
        min_isigi=params.mtz.min_isigi,
        filter_ice_rings=params.mtz.filter_ice_rings,
        d_min=params.mtz.d_min,
    )

    # calculate the scl = lp/dqe correction for outputting but don't apply it as
    # it has already been applied in filter_reflection_table
    (
        integrated_data,
        scl,
    ) = FilteringReductionMethods.calculate_lp_qe_correction_and_filter(integrated_data)

    # sort data before output
    nref = len(integrated_data["miller_index"])
    indices = flex.size_t_range(nref)

    unique = copy.deepcopy(integrated_data["miller_index"])

    map_to_asu(experiment.crystal.get_space_group().type(), False, unique)

    perm = sorted(indices, key=lambda k: unique[k])
    integrated_data = integrated_data.select(flex.size_t(perm))

    if experiment.goniometer is None:
        print("Warning: No goniometer. Experimentally exporting with (1 0 0) axis")

    unit_cell = experiment.crystal.get_unit_cell()

    if experiment.scan is None:
        print("Warning: No Scan. Experimentally exporting no-oscillation values")
        image_range = (1, 1)
        phi_start, phi_range = 0.0, 0.0
    else:
        image_range = experiment.scan.get_image_range()
        phi_start, phi_range = experiment.scan.get_image_oscillation(image_range[0])

    # gather the required information for the reflection file

    nref = len(integrated_data["miller_index"])

    miller_index = integrated_data["miller_index"]

    # profile correlation
    if "profile.correlation" in integrated_data:
        prof_corr = 100.0 * integrated_data["profile.correlation"]
    else:
        prof_corr = flex.double(nref, 100.0)

    # partiality
    if "partiality" in integrated_data:
        partiality = 100 * integrated_data["partiality"]
    else:
        prof_corr = flex.double(nref, 100.0)

    if "intensity.sum.value" in integrated_data:
        I = integrated_data["intensity.sum.value"]
        V = integrated_data["intensity.sum.variance"]
        assert V.all_gt(0)
        V = var_model[0] * (V + var_model[1] * I * I)
        sigI = flex.sqrt(V)
    else:
        I = integrated_data["intensity.prf.value"]
        V = integrated_data["intensity.prf.variance"]
        assert V.all_gt(0)
        V = var_model[0] * (V + var_model[1] * I * I)
        sigI = flex.sqrt(V)

    fout = open(filename, "w")

    # first write the header - in the "standard" coordinate frame...

    panel = experiment.detector[0]
    fast = panel.get_fast_axis()
    slow = panel.get_slow_axis()
    Rd = align_reference_frame(fast, (1, 0, 0), slow, (0, 1, 0))
    print("Coordinate change:")
    print("%5.2f %5.2f %5.2f\n%5.2f %5.2f %5.2f\n%5.2f %5.2f %5.2f\n" % Rd.elems)

    fast = Rd * fast
    slow = Rd * slow

    qx, qy = panel.get_pixel_size()
    nx, ny = panel.get_image_size()
    distance = matrix.col(Rd * panel.get_origin()).dot(
        matrix.col(Rd * panel.get_normal())
    )
    org = Rd * (
        matrix.col(panel.get_origin()) - distance * matrix.col(panel.get_normal())
    )
    orgx = -org.dot(fast) / qx
    orgy = -org.dot(slow) / qy

    UB = Rd * matrix.sqr(experiment.crystal.get_A())
    real_space_ABC = UB.inverse().elems

    if experiment.goniometer is not None:
        axis = Rd * experiment.goniometer.get_rotation_axis()
    else:
        axis = Rd * (1, 0, 0)

    beam = Rd * experiment.beam.get_s0()
    cell_fmt = "%9.3f %9.3f %9.3f %7.3f %7.3f %7.3f"
    axis_fmt = "%9.3f %9.3f %9.3f"

    fout.write(
        "\n".join(
            [
                "!FORMAT=XDS_ASCII    MERGE=FALSE    FRIEDEL'S_LAW=TRUE",
                "!Generated by dials.export",
                "!DATA_RANGE= %d %d" % image_range,
                "!ROTATION_AXIS= %9.6f %9.6f %9.6f" % axis.elems,
                "!OSCILLATION_RANGE= %f" % phi_range,
                "!STARTING_ANGLE= %f" % phi_start,
                "!STARTING_FRAME= %d" % image_range[0],
                "!SPACE_GROUP_NUMBER= %d"
                % experiment.crystal.get_space_group().type().number(),
                f"!UNIT_CELL_CONSTANTS= {cell_fmt % unit_cell.parameters()}",
                f"!UNIT_CELL_A-AXIS= {axis_fmt % real_space_ABC[0:3]}",
                f"!UNIT_CELL_B-AXIS= {axis_fmt % real_space_ABC[3:6]}",
                f"!UNIT_CELL_C-AXIS= {axis_fmt % real_space_ABC[6:9]}",
                f"!X-RAY_WAVELENGTH= {experiment.beam.get_wavelength():f}",
                "!INCIDENT_BEAM_DIRECTION= %f %f %f" % beam.elems,
                "!NX= %d NY= %d QX= %f QY= %f" % (nx, ny, qx, qy),
                f"!ORGX= {orgx:9.2f} ORGY= {orgy:9.2f}",
                f"!DETECTOR_DISTANCE= {distance:8.3f}",
                "!DIRECTION_OF_DETECTOR_X-AXIS= %9.5f %9.5f %9.5f" % fast.elems,
                "!DIRECTION_OF_DETECTOR_Y-AXIS= %9.5f %9.5f %9.5f" % slow.elems,
                "!VARIANCE_MODEL= %7.3e %7.3e" % var_model,
                "!NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD=12",
                "!ITEM_H=1",
                "!ITEM_K=2",
                "!ITEM_L=3",
                "!ITEM_IOBS=4",
                "!ITEM_SIGMA(IOBS)=5",
                "!ITEM_XD=6",
                "!ITEM_YD=7",
                "!ITEM_ZD=8",
                "!ITEM_RLP=9",
                "!ITEM_PEAK=10",
                "!ITEM_CORR=11",
                "!ITEM_PSI=12",
                "!END_OF_HEADER",
                "",
            ]
        )
    )

    # then write the data records

    s0 = Rd * matrix.col(experiment.beam.get_s0())

    for j in range(nref):
        x, y, z = integrated_data["xyzcal.px"][j]
        phi = phi_start + z * phi_range
        h, k, l = miller_index[j]
        X = (UB * (h, k, l)).rotate(axis, phi, deg=True)
        s = s0 + X
        g = s.cross(s0).normalize()

        # find component of beam perpendicular to f, e
        e = -(s + s0).normalize()
        if h == k and k == l:
            u = (h, -h, 0)
        else:
            u = (k - l, l - h, h - k)
        q = (
            (matrix.col(u).transpose() * UB.inverse())
            .normalize()
            .transpose()
            .rotate(axis, phi, deg=True)
        )

        psi = q.angle(g, deg=True)
        if q.dot(e) < 0:
            psi *= -1

        fout.write(
            "%d %d %d %f %f %f %f %f %f %.1f %.1f %f\n"
            % (
                h,
                k,
                l,
                I[j],
                sigI[j],
                x,
                y,
                z,
                scl[j],
                partiality[j],
                prof_corr[j],
                psi,
            )
        )

    fout.write("!END_OF_DATA\n")
    fout.close()
    logger.info("Output %d reflections to %s", nref, filename)
