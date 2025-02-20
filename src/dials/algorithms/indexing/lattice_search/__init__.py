from __future__ import annotations

import importlib.metadata
import itertools
import logging
import math
from io import StringIO

import libtbx.phil
import scitbx.matrix
from dxtbx.model.experiment_list import Experiment, ExperimentList
from scitbx.array_family import flex

from dials.algorithms.indexing import indexer
from dials.algorithms.indexing.basis_vector_search import combinations, optimise

from .ffb_indexer import FfbIndexer
from .low_res_spot_match import LowResSpotMatch
from .pinkindexer import PinkIndexer
from .strategy import Strategy

__all__ = ["Strategy", "LowResSpotMatch", "PinkIndexer", "FfbIndexer"]


logger = logging.getLogger(__name__)


basis_vector_search_phil_str = """\
basis_vector_combinations
    .expert_level = 1
{
    max_combinations = None
        .type = int(value_min=1)
        .help = "Maximum number of basis vector combinations to test for agreement"
                "with input symmetry."
    max_refine = Auto
        .type = int(value_min=1)
        .help = "Maximum number of putative crystal models to test. Default"
                "for rotation sequences: 50, for still images: 5"
        .expert_level = 1
    sys_absent_threshold = 0.9
        .type = float(value_min=0.0, value_max=1.0)
    solution_scorer = filter *weighted
        .type = choice
        .expert_level = 1
    filter
        .expert_level = 1
    {
        check_doubled_cell = True
            .type = bool
        likelihood_cutoff = 0.8
            .type = float(value_min=0, value_max=1)
        volume_cutoff = 1.25
            .type = float(value_min=1)
        n_indexed_cutoff = 0.9
            .type = float(value_min=0, value_max=1)
    }
    weighted
        .expert_level = 1
    {
        power = 1
            .type = int(value_min=1)
        volume_weight = 1
            .type = float(value_min=0)
        n_indexed_weight = 1
            .type = float(value_min=0)
        rmsd_weight = 1
            .type = float(value_min=0)
    }
}

method = None
    .type = choice

optimise_initial_basis_vectors = False
    .type = bool
    .expert_level = 2
"""

basis_vector_search_phil_scope = libtbx.phil.parse(basis_vector_search_phil_str)

methods = []
for entry_point in itertools.chain(
    importlib.metadata.entry_points(group="dials.index.basis_vector_search"),
    importlib.metadata.entry_points(group="dials.index.lattice_search"),
):
    ext_master_scope = libtbx.phil.parse(
        f"""
{entry_point.name}
.expert_level=1
.help={entry_point.load().phil_help}
{{}}
        """
    )
    ext_phil_scope = ext_master_scope.get_without_substitution(entry_point.name)
    assert len(ext_phil_scope) == 1
    ext_phil_scope = ext_phil_scope[0]
    ext_phil_scope.adopt_scope(entry_point.load().phil_scope)
    basis_vector_search_phil_scope.adopt_scope(ext_master_scope)
    methods.append(entry_point.name)
basis_vector_search_phil_scope.adopt_scope(
    libtbx.phil.parse(
        "method = "
        + " ".join("*" + m if m == "fft3d" else m for m in methods)
        + "\n    .type = choice"
    )
)


class LatticeSearch(indexer.Indexer):
    def __init__(self, reflections, experiments, params):
        super().__init__(reflections, experiments, params)

        self._lattice_search_strategy = None
        for entry_point in importlib.metadata.entry_points(
            group="dials.index.lattice_search"
        ):
            if entry_point.name == self.params.method:
                strategy_class = entry_point.load()
                self._lattice_search_strategy = strategy_class(
                    target_symmetry_primitive=self._symmetry_handler.target_symmetry_primitive,
                    max_lattices=self.params.basis_vector_combinations.max_refine,
                    params=getattr(self.params, entry_point.name, None),
                )
                break

    def find_candidate_crystal_models(self):
        candidate_crystal_models = []
        if self._lattice_search_strategy:
            candidate_crystal_models = (
                self._lattice_search_strategy.find_crystal_models(
                    self.reflections, self.experiments
                )
            )
        return candidate_crystal_models

    def find_lattices(self):
        self.candidate_crystal_models = self.find_candidate_crystal_models()

        crystal_model, n_indexed = self.choose_best_orientation_matrix(
            self.candidate_crystal_models
        )
        if crystal_model is not None:
            crystal_models = [crystal_model]
        else:
            crystal_models = []
        experiments = ExperimentList()
        for cm in crystal_models:
            for expt in self.experiments:
                experiments.append(
                    Experiment(
                        imageset=expt.imageset,
                        beam=expt.beam,
                        detector=expt.detector,
                        goniometer=expt.goniometer,
                        scan=expt.scan,
                        crystal=cm,
                    )
                )
        return experiments

    def choose_best_orientation_matrix(self, candidate_orientation_matrices):
        from dials.algorithms.indexing import model_evaluation

        solution_scorer = self.params.basis_vector_combinations.solution_scorer
        if solution_scorer == "weighted":
            weighted_params = self.params.basis_vector_combinations.weighted
            solutions = model_evaluation.ModelRankWeighted(
                power=weighted_params.power,
                volume_weight=weighted_params.volume_weight,
                n_indexed_weight=weighted_params.n_indexed_weight,
                rmsd_weight=weighted_params.rmsd_weight,
            )
        else:
            filter_params = self.params.basis_vector_combinations.filter
            solutions = model_evaluation.ModelRankFilter(
                check_doubled_cell=filter_params.check_doubled_cell,
                likelihood_cutoff=filter_params.likelihood_cutoff,
                volume_cutoff=filter_params.volume_cutoff,
                n_indexed_cutoff=filter_params.n_indexed_cutoff,
            )

        args = []

        for cm in candidate_orientation_matrices:
            sel = self.reflections["id"] == -1
            if self.d_min is not None:
                sel &= 1 / self.reflections["rlp"].norms() > self.d_min
            xo, yo, zo = self.reflections["xyzobs.mm.value"].parts()
            imageset_id = self.reflections["imageset_id"]
            experiments = ExperimentList()
            for i_expt, expt in enumerate(self.experiments):
                # XXX Not sure if we still need this loop over self.experiments
                if expt.scan is not None and expt.scan.has_property("oscillation"):
                    start, end = expt.scan.get_oscillation_range()
                    if (end - start) > 360:
                        # only use reflections from the first 360 degrees of the scan
                        sel.set_selected(
                            (imageset_id == i_expt)
                            & (zo > ((start * math.pi / 180) + 2 * math.pi)),
                            False,
                        )
                experiments.append(
                    Experiment(
                        imageset=expt.imageset,
                        beam=expt.beam,
                        detector=expt.detector,
                        goniometer=expt.goniometer,
                        scan=expt.scan,
                        crystal=cm,
                    )
                )
            refl = self.reflections.select(sel)
            self.index_reflections(experiments, refl)
            if refl.get_flags(refl.flags.indexed).count(True) == 0:
                continue

            from rstbx.dps_core.cell_assessment import SmallUnitCellVolume

            from dials.algorithms.indexing import non_primitive_basis

            threshold = self.params.basis_vector_combinations.sys_absent_threshold
            if threshold and (
                self._symmetry_handler.target_symmetry_primitive is None
                or self._symmetry_handler.target_symmetry_primitive.unit_cell() is None
            ):
                try:
                    non_primitive_basis.correct(
                        experiments, refl, self._assign_indices, threshold
                    )
                    if refl.get_flags(refl.flags.indexed).count(True) == 0:
                        continue
                except SmallUnitCellVolume:
                    logger.debug(
                        "correct_non_primitive_basis SmallUnitCellVolume error for unit cell %s:",
                        experiments[0].crystal.get_unit_cell(),
                    )
                    continue
                except RuntimeError as e:
                    if "Krivy-Gruber iteration limit exceeded" in str(e):
                        logger.debug(
                            "correct_non_primitive_basis Krivy-Gruber iteration limit exceeded error for unit cell %s:",
                            experiments[0].crystal.get_unit_cell(),
                        )
                        continue
                    raise
                if (
                    experiments[0].crystal.get_unit_cell().volume()
                    < self.params.min_cell_volume
                ):
                    continue

            if self.params.known_symmetry.space_group is not None:
                new_crystal, _ = self._symmetry_handler.apply_symmetry(
                    experiments[0].crystal
                )
                if new_crystal is None:
                    continue
                experiments[0].crystal.update(new_crystal)

            args.append((experiments, refl))
            if len(args) == self.params.basis_vector_combinations.max_refine:
                break

        from libtbx import easy_mp

        evaluator = model_evaluation.ModelEvaluation(self.all_params)
        results = easy_mp.parallel_map(
            evaluator.evaluate,
            args,
            iterable_type=easy_mp.posiargs,
            processes=self.params.nproc,
            preserve_exception_message=True,
        )

        for soln in results:
            if soln is None:
                continue
            solutions.append(soln)

        if len(solutions):
            logger.info("Candidate solutions:")
            logger.info(str(solutions))
            best_model = solutions.best_model()
            logger.debug("best model_likelihood: %.2f", best_model.model_likelihood)
            logger.debug("best n_indexed: %i", best_model.n_indexed)
            self.hkl_offset = best_model.hkl_offset
            return best_model.crystal, best_model.n_indexed
        else:
            return None, None


class BasisVectorSearch(LatticeSearch):
    def __init__(self, reflections, experiments, params):
        super().__init__(reflections, experiments, params)

        strategy_class = None
        for entry_point in importlib.metadata.entry_points(
            group="dials.index.basis_vector_search"
        ):
            if entry_point.name == params.indexing.method:
                strategy_class = entry_point.load()
                break
        if not strategy_class:
            raise RuntimeError(
                f"Unknown basis vector search strategy: {params.indexing.method}"
            )

        target_unit_cell = None
        if (
            self._symmetry_handler.target_symmetry_primitive is not None
            and self._symmetry_handler.target_symmetry_primitive.unit_cell() is not None
        ):
            target_unit_cell = (
                self._symmetry_handler.target_symmetry_primitive.unit_cell()
            )
        self._basis_vector_search_strategy = strategy_class(
            max_cell=self.params.max_cell,
            min_cell=self.params.min_cell,
            target_unit_cell=target_unit_cell,
            params=getattr(self.params, entry_point.name),
        )

    def find_candidate_basis_vectors(self):
        self.d_min = self.params.refinement_protocol.d_min_start
        sel = self.reflections["id"] == -1
        if self.d_min is not None:
            sel &= 1 / self.reflections["rlp"].norms() > self.d_min
        reflections = self.reflections.select(sel)
        (
            self.candidate_basis_vectors,
            used_in_indexing,
        ) = self._basis_vector_search_strategy.find_basis_vectors(reflections["rlp"])
        self._used_in_indexing = sel.iselection().select(used_in_indexing)
        if self.d_min is None:
            self.d_min = flex.min(
                1 / self.reflections["rlp"].select(self._used_in_indexing).norms()
            )

        self.debug_show_candidate_basis_vectors()
        return self.candidate_basis_vectors

    def find_candidate_crystal_models(self):
        self.find_candidate_basis_vectors()
        if self.params.optimise_initial_basis_vectors:
            self.optimise_basis_vectors()
        candidate_crystal_models = self.find_candidate_orientation_matrices(
            self.candidate_basis_vectors
        )
        return candidate_crystal_models

    def find_candidate_orientation_matrices(self, candidate_basis_vectors):
        candidate_crystal_models = combinations.candidate_orientation_matrices(
            candidate_basis_vectors,
            max_combinations=self.params.basis_vector_combinations.max_combinations,
        )
        if self._symmetry_handler.target_symmetry_reference_setting is not None:
            target_symmetry = self._symmetry_handler.target_symmetry_reference_setting
        elif self._symmetry_handler.target_symmetry_primitive is not None:
            target_symmetry = self._symmetry_handler.target_symmetry_primitive
        else:
            target_symmetry = None
        if target_symmetry is not None:
            candidate_crystal_models = combinations.filter_known_symmetry(
                candidate_crystal_models,
                target_symmetry,
                relative_length_tolerance=self.params.known_symmetry.relative_length_tolerance,
                absolute_angle_tolerance=self.params.known_symmetry.absolute_angle_tolerance,
                max_delta=self.params.known_symmetry.max_delta,
            )

        if self.refined_experiments is not None and len(self.refined_experiments) > 0:
            candidate_crystal_models = combinations.filter_similar_orientations(
                candidate_crystal_models,
                self.refined_experiments.crystals(),
                minimum_angular_separation=self.params.multiple_lattice_search.minimum_angular_separation,
            )

        return candidate_crystal_models

    def optimise_basis_vectors(self):
        optimised_basis_vectors = optimise.optimise_basis_vectors(
            self.reflections["rlp"].select(self._used_in_indexing),
            self.candidate_basis_vectors,
        )
        self.candidate_basis_vectors = [
            scitbx.matrix.col(v) for v in optimised_basis_vectors
        ]

    def debug_show_candidate_basis_vectors(self):
        vectors = self.candidate_basis_vectors

        logger.debug("Candidate basis vectors:")
        for i, v in enumerate(vectors):
            logger.debug(f"{i} {v.length()}")  # , vector_heights[i]

        if self.params.debug:
            # print a table of the angles between each pair of vectors
            s = StringIO()

            angles = flex.double(len(vectors) ** 2)
            angles.reshape(flex.grid(len(vectors), len(vectors)))

            for i in range(len(vectors)):
                v_i = vectors[i]
                for j in range(i + 1, len(vectors)):
                    v_j = vectors[j]
                    angles[i, j] = v_i.angle(v_j, deg=True)

            print((" " * 7), end=" ", file=s)
            for i in range(len(vectors)):
                print(f"{vectors[i].length():7.3f}", end=" ", file=s)
            print(file=s)
            for i in range(len(vectors)):
                print(f"{vectors[i].length():7.3f}", end=" ", file=s)
                for j in range(len(vectors)):
                    if j <= i:
                        print((" " * 7), end=" ", file=s)
                    else:
                        print(f"{angles[i, j]:5.1f}  ", end=" ", file=s)
                print(file=s)

            logger.debug(s.getvalue())
