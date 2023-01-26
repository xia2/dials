"""
Definitions of the scaling algorithm.
"""

from __future__ import annotations

import itertools
import json
import logging
import time

from libtbx import Auto

from dials.algorithms.scaling.observers import (
    ScalingHTMLContextManager,
    ScalingSummaryContextManager,
)
from dials.algorithms.scaling.scale_and_filter import AnalysisResults, log_cycle_results
from dials.algorithms.scaling.scaler_factory import MultiScalerFactory, create_scaler
from dials.algorithms.scaling.scaling_library import (
    create_datastructures_for_reference_file,
    create_scaling_model,
    determine_best_unit_cell,
    merging_stats_from_scaled_array,
    scaled_data_as_miller_array,
    set_image_ranges_in_scaling_models,
)
from dials.algorithms.scaling.scaling_utilities import (
    DialsMergingStatisticsError,
    log_memory_usage,
)
from dials.algorithms.statistics.cc_half_algorithm import (
    CCHalfFromDials as deltaccscript,
)
from dials.array_family import flex
from dials.command_line.compute_delta_cchalf import phil_scope as deltacc_phil_scope
from dials.command_line.cosym import cosym
from dials.command_line.cosym import phil_scope as cosym_phil_scope
from dials.util.exclude_images import (
    exclude_image_ranges_for_scaling,
    get_valid_image_ranges,
)
from dials.util.multi_dataset_handling import (
    assign_unique_identifiers,
    parse_multiple_datasets,
    select_datasets_on_ids,
    update_imageset_ids,
)

logger = logging.getLogger("dials")


def prepare_input(params, experiments, reflections):
    """Perform checks on the data and prepare the data for scaling.

    Raises:
        ValueError - a range of checks are made, a ValueError may be raised
            for a number of reasons.

    """

    #### First exclude any datasets, before the dataset is split into
    #### individual reflection tables and expids set.
    if (
        params.dataset_selection.exclude_datasets
        or params.dataset_selection.use_datasets
    ):
        experiments, reflections = select_datasets_on_ids(
            experiments,
            reflections,
            params.dataset_selection.exclude_datasets,
            params.dataset_selection.use_datasets,
        )
        ids = flex.size_t()
        for r in reflections:
            ids.extend(r.experiment_identifiers().keys())
        logger.info(
            "\nDataset ids for retained datasets are: %s \n",
            ",".join(str(i) for i in ids),
        )

    #### Split the reflections tables into a list of reflection tables,
    #### with one table per experiment.
    logger.info(
        "Checking for the existence of a reflection table \n"
        "containing multiple datasets \n"
    )
    reflections = parse_multiple_datasets(reflections)
    logger.info(
        "Found %s reflection tables & %s experiments in total.",
        len(reflections),
        len(experiments),
    )

    if len(experiments) != len(reflections):
        raise ValueError(
            "Mismatched number of experiments and reflection tables found."
        )

    #### Sort out deprecated options
    if params.scaling_options.target_model or params.scaling_options.target_mtz:
        if params.scaling_options.reference:
            raise ValueError(
                "Can't specify reference in addition to target_mtz/target_model"
            )
        if params.scaling_options.target_model and params.scaling_options.target_mtz:
            raise ValueError(
                "Can only specify one of target_mtz/target_model (both deprecated, use reference=)"
            )
        if params.scaling_options.target_model:
            logger.warning(
                "Warning: target_model option is deprecated and will be removed, please use reference="
            )
            params.scaling_options.reference = params.scaling_options.target_model
            params.scaling_options.target_model = None
        elif params.scaling_options.target_mtz:
            logger.warning(
                "Warning: target_mtz option is deprecated and will be removed, please use reference="
            )
            params.scaling_options.reference = params.scaling_options.target_mtz
            params.scaling_options.target_mtz = None

    #### Assign experiment identifiers.
    experiments, reflections = assign_unique_identifiers(experiments, reflections)
    ids = itertools.chain.from_iterable(
        r.experiment_identifiers().keys() for r in reflections
    )
    logger.info("\nDataset ids are: %s \n", ",".join(str(i) for i in ids))

    for r in reflections:
        r.unset_flags(flex.bool(len(r), True), r.flags.bad_for_scaling)
        r.unset_flags(flex.bool(r.size(), True), r.flags.scaled)

    reflections, experiments = exclude_image_ranges_for_scaling(
        reflections, experiments, params.exclude_images
    )

    #### Allow checking of consistent indexing, useful for
    #### targeted / incremental scaling.
    if params.scaling_options.check_consistent_indexing:
        logger.info("Running dials.cosym to check consistent indexing:\n")
        cosym_params = cosym_phil_scope.extract()
        cosym_params.nproc = params.scaling_options.nproc
        cosym_instance = cosym(experiments, reflections, cosym_params)
        cosym_instance.run()
        experiments = cosym_instance.experiments
        reflections = cosym_instance.reflections
        logger.info("Finished running dials.cosym, continuing with scaling.\n")

    #### Make sure all experiments in same space group
    sgs = [expt.crystal.get_space_group().type().number() for expt in experiments]
    if len(set(sgs)) > 1:
        raise ValueError(
            """The experiments have different space groups:
            space group numbers found: %s
            Please reanalyse the data so that space groups are consistent,
            (consider using dials.reindex, dials.symmetry or dials.cosym) or
            remove incompatible experiments (using the option exclude_datasets=)"""
            % ", ".join(map(str, set(sgs)))
        )
    logger.info(
        "Space group being used during scaling is %s",
        experiments[0].crystal.get_space_group().info(),
    )

    #### If doing targeted scaling, extract data and append an experiment
    #### and reflection table to the lists
    if params.scaling_options.reference:
        # Set a suitable d_min in the case when we might have a model file
        d_min_for_structure_model = 2.0
        if params.cut_data.d_min not in (None, Auto):
            d_min_for_structure_model = params.cut_data.d_min
        expt, reflection_table = create_datastructures_for_reference_file(
            experiments,
            params.scaling_options.reference,
            params.anomalous,
            d_min=d_min_for_structure_model,
        )
        experiments.append(expt)
        reflections.append(reflection_table)

    #### Perform any non-batch cutting of the datasets, including the target dataset
    best_unit_cell = params.reflection_selection.best_unit_cell
    if best_unit_cell is None:
        best_unit_cell = determine_best_unit_cell(experiments)
    for reflection in reflections:
        if params.cut_data.d_min or params.cut_data.d_max:
            d = best_unit_cell.d(reflection["miller_index"])
            if params.cut_data.d_min:
                sel = d < params.cut_data.d_min
                reflection.set_flags(sel, reflection.flags.user_excluded_in_scaling)
            if params.cut_data.d_max:
                sel = d > params.cut_data.d_max
                reflection.set_flags(sel, reflection.flags.user_excluded_in_scaling)
        if params.cut_data.partiality_cutoff and "partiality" in reflection:
            reflection.set_flags(
                reflection["partiality"] < params.cut_data.partiality_cutoff,
                reflection.flags.user_excluded_in_scaling,
            )
    return params, experiments, reflections


class ScalingAlgorithm:
    def __init__(self, params, experiments, reflections):
        self.scaler = None
        self.scaled_miller_array = None
        self.merging_statistics_result = None
        self.anom_merging_statistics_result = None
        self.filtering_results = None
        self.params, self.experiments, self.reflections = prepare_input(
            params, experiments, reflections
        )
        self._create_model_and_scaler()
        logger.debug("Initialised scaling script object")
        log_memory_usage()

    def _create_model_and_scaler(self):
        """Create the scaling models and scaler."""
        self.experiments = create_scaling_model(
            self.params, self.experiments, self.reflections
        )
        logger.info("\nScaling models have been initialised for all experiments.")
        logger.info("%s%s%s", "\n", "=" * 80, "\n")

        self.experiments = set_image_ranges_in_scaling_models(self.experiments)

        self.scaler = create_scaler(self.params, self.experiments, self.reflections)

    def run(self):
        """Run the scaling script."""
        with ScalingHTMLContextManager(self), ScalingSummaryContextManager(self):
            start_time = time.time()
            self.scale()
            self.remove_bad_data()
            if not self.experiments:
                raise ValueError("All data sets have been rejected as bad.")
            for table in self.reflections:
                bad = table.get_flags(table.flags.bad_for_scaling, all=False)
                table.unset_flags(flex.bool(table.size(), True), table.flags.scaled)
                table.set_flags(~bad, table.flags.scaled)
            self.scaled_miller_array = scaled_data_as_miller_array(
                self.reflections,
                self.experiments,
                anomalous_flag=False,
                best_unit_cell=self.params.reflection_selection.best_unit_cell,
            )
            try:
                self.calculate_merging_stats()
            except DialsMergingStatisticsError as e:
                logger.info(e)

            # All done!
            logger.info("\nTotal time taken: %.4fs ", time.time() - start_time)
            logger.info("%s%s%s", "\n", "=" * 80, "\n")

    def scale(self):
        """The main scaling algorithm."""

        if self.scaler.id_ == "target":
            ### FIXME add in quick prescaling round if large scale difference?
            self.scaler.perform_scaling()

            if (
                self.params.scaling_options.only_target
                or self.params.scaling_options.reference
            ):

                self.scaler = targeted_scaling_algorithm(self.scaler)
                return
            # Now pass to a multiscaler ready for next round of scaling.
            self.scaler.expand_scales_to_all_reflections()
            self.scaler = MultiScalerFactory.create_from_targetscaler(self.scaler)

        # From here onwards, scaler should only be a SingleScaler
        # or MultiScaler (not TargetScaler).
        self.scaler = scaling_algorithm(self.scaler)

    def remove_bad_data(self):
        """Remove any target model/mtz data and any datasets which were removed
        from the scaler during scaling."""
        # first remove target refl/exps
        if (
            self.params.scaling_options.reference
            or self.params.scaling_options.only_target
        ):
            # now remove things that were used as the target:
            n_target = len(self.experiments) - self.scaler.n_initial_active_scalers
            self.experiments = self.experiments[:-n_target]
            self.reflections = self.reflections[:-n_target]
        # remove any bad datasets:
        removed_ids = self.scaler.removed_datasets
        if removed_ids:
            logger.info("deleting removed datasets from memory: %s", removed_ids)
            expids = list(self.experiments.identifiers())
            locs_in_list = [expids.index(expid) for expid in removed_ids]
            self.experiments, self.reflections = select_datasets_on_ids(
                self.experiments, self.reflections, exclude_datasets=locs_in_list
            )
        # also remove negative scales (or scales below 0.001)
        n = 0
        for table in self.reflections:
            bad_sf = (
                table["inverse_scale_factor"] < self.params.cut_data.small_scale_cutoff
            )
            n += bad_sf.count(True)
            table.set_flags(bad_sf, table.flags.excluded_for_scaling)
        if n > 0:
            logger.info(
                f"{n} reflections excluded: scale factor < {self.params.cut_data.small_scale_cutoff}"
            )

    def calculate_merging_stats(self):
        try:
            (
                self.merging_statistics_result,
                self.anom_merging_statistics_result,
            ) = merging_stats_from_scaled_array(
                self.scaled_miller_array,
                self.params.output.merging.nbins,
                self.params.output.use_internal_variance,
                additional_stats=self.params.output.additional_stats,
            )
        except DialsMergingStatisticsError as e:
            logger.warning(e, exc_info=True)

    def finish(self):
        """Save the experiments json and scaled pickle file."""

        # Now create a joint reflection table. Delete all other data before
        # joining reflection tables - just need experiments for mtz export
        # and a reflection table.
        del self.scaler
        cols_to_del = [
            "variance",
            "intensity",
            "s0",
            "s0c",
            "s1c",
            "prescaling_correction",
            "batch",
        ]
        for table in self.reflections:
            for col in cols_to_del:
                try:
                    del table[col]
                except KeyError:
                    pass

        # update imageset ids before combining reflection tables.
        self.reflections = update_imageset_ids(self.experiments, self.reflections)
        # Note, we don't use flex.reflection_table.concat below on purpose, so
        # that the dataset ids in the table are consistent from input to output
        # when datasets are removed, e.g. by filtering, exclude_datasets= etc.
        joint_table = flex.reflection_table()
        for i in range(len(self.reflections)):
            joint_table.extend(self.reflections[i])
            self.reflections[i] = 0  # del reference from initial list

        # remove reflections with very low scale factors
        sel = (
            joint_table["inverse_scale_factor"]
            < self.params.cut_data.small_scale_cutoff
        )
        good_sel = ~joint_table.get_flags(joint_table.flags.bad_for_scaling, all=False)
        n_low = (good_sel & sel).count(True)
        if n_low > 0:
            logger.warning(
                f"""{n_low} non-excluded reflections were assigned scale factors < {self.params.cut_data.small_scale_cutoff} during scaling.
These will be excluded in the output reflection table. It may be best to rerun
scaling from this point for an improved model."""
            )
            joint_table.set_flags(sel, joint_table.flags.excluded_for_scaling)

        return self.experiments, joint_table


class ScaleAndFilterAlgorithm(ScalingAlgorithm):
    def __init__(self, params, experiments, reflections):
        super().__init__(params, experiments, reflections)
        if (
            params.filtering.deltacchalf.mode == "dataset"
            and self.scaler.id_ != "multi"
        ):
            raise ValueError(
                """\
Whole dataset deltacchalf scaling and filtering can only be performed in
multi-dataset scaling mode (not single dataset or scaling against a reference)"""
            )

    def run(self):
        """Run cycles of scaling and filtering."""
        with ScalingHTMLContextManager(self):
            start_time = time.time()
            results = AnalysisResults()

            for counter in range(1, self.params.filtering.deltacchalf.max_cycles + 1):
                self.run_scaling_cycle()

                if counter == 1:
                    results.initial_expids_and_image_ranges = [
                        (exp.identifier, exp.scan.get_image_range())
                        if exp.scan
                        else None
                        for exp in self.experiments
                    ]

                delta_cc_params = deltacc_phil_scope.extract()
                delta_cc_params.mode = self.params.filtering.deltacchalf.mode
                delta_cc_params.group_size = (
                    self.params.filtering.deltacchalf.group_size
                )
                delta_cc_params.stdcutoff = self.params.filtering.deltacchalf.stdcutoff
                logger.info("\nPerforming a round of filtering.\n")

                # need to reduce to single table.
                joined_reflections = flex.reflection_table()
                for table in self.reflections:
                    joined_reflections.extend(table)

                script = deltaccscript(
                    delta_cc_params, self.experiments, joined_reflections
                )
                script.run()

                valid_image_ranges = get_valid_image_ranges(self.experiments)
                results.expids_and_image_ranges = [
                    (exp.identifier, valid_image_ranges[i]) if exp.scan else None
                    for i, exp in enumerate(self.experiments)
                ]

                self.experiments = script.experiments
                self.params.dataset_selection.use_datasets = None
                self.params.dataset_selection.exclude_datasets = None

                results = log_cycle_results(results, self, script)
                logger.info(
                    "Cycle %s of filtering, n_reflections removed this cycle: %s",
                    counter,
                    results.get_last_cycle_results()["n_removed"],
                )

                # Test termination conditions
                latest_results = results.get_last_cycle_results()
                if latest_results["n_removed"] == 0:
                    logger.info(
                        "Finishing scaling and filtering as no data removed in this cycle."
                    )
                    self.reflections = parse_multiple_datasets(
                        [script.filtered_reflection_table]
                    )
                    if self.params.scaling_options.full_matrix:
                        results = self._run_final_scale_cycle(results)
                    results.finish(termination_reason="no_more_removed")
                    break

                # Need to split reflections for further processing.
                self.reflections = parse_multiple_datasets(
                    [script.filtered_reflection_table]
                )

                if (
                    latest_results["cumul_percent_removed"]
                    > self.params.filtering.deltacchalf.max_percent_removed
                ):
                    logger.info(
                        "Finishing scale and filtering as have now removed more than the limit."
                    )
                    results = self._run_final_scale_cycle(results)
                    results.finish(termination_reason="max_percent_removed")
                    break

                if self.params.filtering.deltacchalf.min_completeness:
                    if (
                        latest_results["merging_stats"]["completeness"]
                        < self.params.filtering.deltacchalf.min_completeness
                    ):
                        logger.info(
                            "Finishing scaling and filtering as completeness now below cutoff."
                        )
                        results = self._run_final_scale_cycle(results)
                        results.finish(termination_reason="below_completeness_limit")
                        break

                if counter == self.params.filtering.deltacchalf.max_cycles:
                    logger.info("Finishing as reached max number of cycles.")
                    results = self._run_final_scale_cycle(results)
                    results.finish(termination_reason="max_cycles")
                    break

                # If not finished then need to create new scaler to try again
                self._create_model_and_scaler()
            self.filtering_results = results
            # Print summary of results
            logger.info(results)
            with open(self.params.filtering.output.scale_and_filter_results, "w") as f:
                json.dump(self.filtering_results.to_dict(), f, indent=2)
            # All done!
            logger.info("\nTotal time taken: %.4fs ", time.time() - start_time)
            logger.info("%s%s%s", "\n", "=" * 80, "\n")

    def run_scaling_cycle(self):
        """Do a round of scaling for scaling and filtering."""
        # Turn off the full matrix round, all else is the same.

        initial_full_matrix = self.params.scaling_options.full_matrix
        self.scaler.params.scaling_options.full_matrix = False
        self.scaler = scaling_algorithm(self.scaler)
        self.scaler.params.scaling_options.full_matrix = initial_full_matrix
        self.remove_bad_data()
        for table in self.reflections:
            bad = table.get_flags(table.flags.bad_for_scaling, all=False)
            table.unset_flags(flex.bool(table.size(), True), table.flags.scaled)
            table.set_flags(~bad, table.flags.scaled)
        self.scaled_miller_array = scaled_data_as_miller_array(
            self.reflections,
            self.experiments,
            anomalous_flag=False,
            best_unit_cell=self.params.reflection_selection.best_unit_cell,
        )
        try:
            self.calculate_merging_stats()
        except DialsMergingStatisticsError as e:
            logger.info(e)
        logger.info("Performed cycle of scaling.")

    def _run_final_scale_cycle(self, results):
        self._create_model_and_scaler()
        super().run()
        results.add_final_stats(self.merging_statistics_result)
        for table in self.reflections:
            bad = table.get_flags(table.flags.bad_for_scaling, all=False)
            table.unset_flags(flex.bool(table.size(), True), table.flags.scaled)
            table.set_flags(~bad, table.flags.scaled)
        return results


def expand_and_do_outlier_rejection(scaler, calc_cov=False):
    """Calculate scales for all reflections and do outlier rejection."""
    scaler.expand_scales_to_all_reflections(calc_cov=calc_cov)
    if scaler.params.scaling_options.outlier_rejection:
        scaler.round_of_outlier_rejection()


def do_intensity_combination(scaler, reselect=True):
    """
    Do prf/sum intensity combination.

    Optionally reselect reflections to prepare for another minimisation round.
    """
    if scaler.params.reflection_selection.intensity_choice == "combine":
        scaler.combine_intensities()
        if scaler.params.scaling_options.outlier_rejection:
            scaler.round_of_outlier_rejection()
    if reselect:
        scaler.make_ready_for_scaling()


def do_error_analysis(scaler, reselect=True):
    """
    Do error model analysis.

    Optionally reselect reflections to prepare for another minimisation round.
    """
    if scaler.params.weighting.error_model.error_model:
        scaler.perform_error_optimisation()
    if reselect:
        scaler.make_ready_for_scaling()


def scaling_algorithm(scaler):
    """Main algorithm for scaling."""
    scaler.perform_scaling()
    need_to_rescale = False

    if (
        scaler.params.reflection_selection.intensity_choice == "combine"
        or scaler.params.scaling_options.outlier_rejection
    ):

        expand_and_do_outlier_rejection(scaler)

        do_intensity_combination(scaler, reselect=True)

        need_to_rescale = True

    if (
        scaler.params.weighting.error_model.error_model
        or scaler.params.scaling_options.outlier_rejection
    ):
        if need_to_rescale:
            scaler.perform_scaling()

        expand_and_do_outlier_rejection(scaler)

        do_error_analysis(scaler, reselect=True)

        need_to_rescale = True

    if scaler.params.scaling_options.full_matrix:

        scaler.perform_scaling(
            engine=scaler.params.scaling_refinery.full_matrix_engine,
            max_iterations=scaler.params.scaling_refinery.full_matrix_max_iterations,
        )
        # check if we're fixing a parameter, if so, redo full matrix with
        # smaller tolerance for one cycle.
        need_to_scale = scaler.fix_initial_parameter()
        if need_to_scale:
            scaler.perform_scaling(
                engine=scaler.params.scaling_refinery.full_matrix_engine,
                max_iterations=1,
                tolerance=scaler.params.scaling_refinery.rmsd_tolerance / 4.0,
            )
    elif need_to_rescale:
        scaler.perform_scaling()

    # The minimisation has only been done on a subset on the data, so apply the
    # scale factors to the whole reflection table.

    scaler.clear_Ih_table()
    expand_and_do_outlier_rejection(scaler, calc_cov=True)
    do_error_analysis(scaler, reselect=False)

    scaler.prepare_reflection_tables_for_output()
    return scaler


def targeted_scaling_algorithm(scaler):
    """Main algorithm for targeted scaling."""

    if scaler.params.scaling_options.outlier_rejection:
        expand_and_do_outlier_rejection(scaler)
        scaler.make_ready_for_scaling()
        scaler.perform_scaling()

    if scaler.params.scaling_options.full_matrix and (
        scaler.params.scaling_refinery.engine == "SimpleLBFGS"
    ):
        scaler.perform_scaling(
            engine=scaler.params.scaling_refinery.full_matrix_engine,
            max_iterations=scaler.params.scaling_refinery.full_matrix_max_iterations,
        )

    expand_and_do_outlier_rejection(scaler, calc_cov=True)
    # do_error_analysis(scaler, reselect=False)

    scaler.prepare_reflection_tables_for_output()
    return scaler
