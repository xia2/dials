# LIBTBX_PRE_DISPATCHER_INCLUDE_SH export PHENIX_GUI_ENVIRONMENT=1


from __future__ import annotations

import functools
import os

import iotbx.mtz
import iotbx.phil
from cctbx import crystal
from dxtbx.model import ExperimentList

import dials.util
from dials.algorithms.clustering.unit_cell import cluster_unit_cells
from dials.array_family import flex
from dials.util.multi_dataset_handling import (
    assign_unique_identifiers,
    parse_multiple_datasets,
)
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files

help_message = """
"""

phil_scope = iotbx.phil.parse(
    """
threshold = 5000
  .type = float(value_min=0)
  .help = 'Threshold value for the clustering'
plot {
  show = False
    .type = bool
  name = 'cluster_unit_cell.png'
    .type = path
  log = False
    .type = bool
    .help = 'Display the dendrogram with a log scale'
}
output.clusters = False
    .type = bool
    .help = "If True, clusters will be split at the threshold value and a pair"
            "of output files will be created for each cluster"
"""
)


@dials.util.show_mail_handle_errors()
def run(args=None):
    usage = "dials.cluster_unit_cell [options] models.expt"

    parser = ArgumentParser(
        usage=usage,
        phil=phil_scope,
        read_experiments=True,
        read_reflections=True,
        check_format=False,
        epilog=help_message,
    )

    params, _, args = parser.parse_args(
        args, show_diff_phil=True, return_unhandled=True
    )
    reflections, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )
    crystal_symmetries = []

    if len(experiments) == 0:
        if not args:
            parser.print_help()
            exit(0)
        for arg in args:
            assert os.path.isfile(arg), arg
            mtz_object = iotbx.mtz.object(file_name=arg)
            arrays = mtz_object.as_miller_arrays(
                merge_equivalents=False, anomalous=False
            )
            crystal_symmetries.append(arrays[0].crystal_symmetry())
    else:
        crystal_symmetries = [
            crystal.symmetry(
                unit_cell=expt.crystal.get_unit_cell(),
                space_group=expt.crystal.get_space_group(),
            )
            for expt in experiments
        ]
    if len(crystal_symmetries) <= 1:
        print(f"Cannot cluster only {len(crystal_symmetries)} crystals, exiting.")
        exit(0)
    clusters = do_cluster_analysis(crystal_symmetries, params)

    if params.output.clusters:
        if len(experiments) == 0:
            print("Clustering output can only be generated for input .expt files")
            return
        # Possibilities: either same number of experiments and reflection files,
        # or just one reflection file containing multiple sequences, or no
        # reflections given
        # Want a combined table to work on with experiment identifiers set.

        def _assign_and_return_joint(experiments, reflections):
            experiments, reflections = assign_unique_identifiers(
                experiments, reflections
            )
            joint_table = flex.reflection_table()
            for refls in reflections:
                joint_table.extend(refls)
            return joint_table

        if len(reflections) == 1:
            reflections = reflections[0]
            if len(set(reflections["id"])) != len(experiments):
                raise ValueError(
                    f"Mismatched number of reflection tables (f{len(set(reflections['id']))}) and experiments (f{len(experiments)})"
                )
            if not dict(reflections.experiment_identifiers()):
                reflections = reflections.split_by_experiment_id()
                reflections = _assign_and_return_joint(experiments, reflections)

        elif len(reflections) > 1:
            if not len(reflections) == len(experiments):
                reflections = parse_multiple_datasets(reflections)
                if len(reflections) != len(experiments):
                    raise ValueError(
                        f"Mismatched number of reflection tables (f{len(reflections)}) and experiments (f{len(experiments)})"
                    )
            reflections = _assign_and_return_joint(experiments, reflections)
        # else: no reflections given, continue and just split experiments

        template = "{prefix}_{index:0{maxindexlength:d}d}.{extension}"
        experiments_template = functools.partial(
            template.format,
            prefix="cluster",
            maxindexlength=len(str(len(clusters) - 1)),
            extension="expt",
        )
        reflections_template = functools.partial(
            template.format,
            prefix="cluster",
            maxindexlength=len(str(len(clusters) - 1)),
            extension="refl",
        )

        clusters.sort(key=len, reverse=True)
        for j, cluster in enumerate(clusters):
            sub_expt = ExperimentList([experiments[i] for i in cluster.lattice_ids])
            expt_filename = experiments_template(index=j)
            print(
                f"Saving {len(sub_expt)} lattices from cluster {j+1} to {expt_filename}"
            )
            sub_expt.as_file(expt_filename)
            if reflections:
                identifiers = sub_expt.identifiers()
                sub_refl = reflections.select_on_experiment_identifiers(identifiers)
                # renumber the ids to go from 0->n-1
                sub_refl.reset_ids()
                refl_filename = reflections_template(index=j)
                print(f"Saving reflections from cluster {j+1} to {refl_filename}")
                sub_refl.as_file(refl_filename)


def do_cluster_analysis(crystal_symmetries, params):
    lattice_ids = list(range(len(crystal_symmetries)))

    if params.plot.show or params.plot.name:
        if not params.plot.show:
            import matplotlib

            # http://matplotlib.org/faq/howto_faq.html#generate-images-without-having-a-window-appear
            matplotlib.use("Agg")  # use a non-interactive backend
        import matplotlib.pyplot as plt

        plt.figure("Andrews-Bernstein distance dendogram", figsize=(12, 8))
        ax = plt.gca()
        no_plot = False
    else:
        ax = None
        no_plot = True

    clustering = cluster_unit_cells(
        crystal_symmetries,
        lattice_ids=lattice_ids,
        threshold=params.threshold,
        ax=ax,
        no_plot=no_plot,
    )
    print(clustering)

    if params.plot.show or params.plot.name:

        if params.plot.log:
            ax.set_yscale("symlog", linthresh=1)
        else:
            ax.set_ylim(-ax.get_ylim()[1] / 100, ax.get_ylim()[1])

        plt.tight_layout()
        if params.plot.name:
            plt.savefig(params.plot.name)
        if params.plot.show:
            plt.show()

    return clustering.clusters


if __name__ == "__main__":
    run()
