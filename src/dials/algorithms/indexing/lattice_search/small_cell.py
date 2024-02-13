from __future__ import annotations

import logging

import iotbx.phil
from dxtbx.model import Crystal
from scitbx import matrix
from scitbx.array_family import flex
from scitbx.math import superpose
from xfel.small_cell.small_cell import small_cell_index_lattice_detail

from dials.algorithms.indexing import DialsIndexError

from .strategy import Strategy

logger = logging.getLogger(__name__)

from xfel.small_cell.command_line.small_cell_index import small_cell_phil_str


class SmallCell(Strategy):
    """
    A lattice search strategy using the small_cell algorithm.

    """

    phil_help = (
        "A lattice search strategy that matches low resolution spots to candidate "
        "indices based on a known unit cell and space group."
    )

    phil_scope = iotbx.phil.parse(small_cell_phil_str)

    def __init__(
        self, target_symmetry_primitive, max_lattices, params=None, *args, **kwargs
    ):
        """Construct a LowResSpotMatch object.

        Args:
            target_symmetry_primitive (cctbx.crystal.symmetry): The target
                crystal symmetry and unit cell

            max_lattices (int): The maximum number of lattice models to find
        """
        super().__init__(params=None, *args, **kwargs)
        self._target_symmetry_primitive = target_symmetry_primitive
        self._max_lattices = max_lattices

        if target_symmetry_primitive is None:
            raise DialsIndexError(
                "Target unit cell and space group must be provided for small_cell"
            )

        self._params.small_cell.powdercell = (
            target_symmetry_primitive.unit_cell().parameters()
        )
        self._params.small_cell.spacegroup = str(
            target_symmetry_primitive.space_group_info()
        )

    def find_crystal_models(self, reflections, experiments):
        """Find a list of candidate crystal models.

        Args:
            reflections (dials.array_family.flex.reflection_table):
                The found spots centroids and associated data

            experiments (dxtbx.model.experiment_list.ExperimentList):
                The experimental geometry models
        """

        small_cell_result = small_cell_index_lattice_detail(
            experiments, reflections, self._params
        )
        if not small_cell_result:
            return []

        ori = small_cell_result[2]
        direct_matrix = ori.direct_matrix()
        real_a = direct_matrix[0:3]
        real_b = direct_matrix[3:6]
        real_c = direct_matrix[6:9]
        crystal = Crystal(real_a, real_b, real_c, self._params.small_cell.spacegroup)
        candidate_crystal_models = [
            crystal,
        ]

        # Reset id column to ensure that index assignment later will work properly.
        # This is a workaround for https://github.com/dials/dials/issues/2485
        reflections["id"] *= 0
        reflections["id"] -= 1

        # Additional refinement of the crystal orientation using the spots indexed
        # by small_cell by superposing ideal relp positions over the observed relps
        indexed = small_cell_result[3]
        Bmat = matrix.sqr(crystal.get_B())
        observed_relps = [(0.0, 0.0, 0.0)]
        predicted_relps = [(0.0, 0.0, 0.0)]
        for spot in indexed:
            predicted_relps.append(Bmat * spot.hkl.ohkl)
            observed_relps.append(spot.xyz)
        fit = superpose.least_squares_fit(
            flex.vec3_double(observed_relps), flex.vec3_double(predicted_relps)
        )
        Umat = fit.r
        crystal.set_A(Umat * Bmat)

        self.candidate_crystal_models = candidate_crystal_models
        return self.candidate_crystal_models
