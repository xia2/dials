from __future__ import annotations

import logging
from collections import namedtuple

from scitbx import matrix, sparse

from dials.algorithms.refinement import DialsRefineConfigError
from dials.array_family import flex

"""The PredictionParameterisation class ties together parameterisations for
individual experimental models: beam, crystal orientation, crystal unit cell
detector and goniometer into a parameterisation for the reflection prediction
equation. Methods are provided to get and set the global list of parameters in
a defined order and calculate the gradients of the predicted centroids with
respect to the global parameters. A concrete version of this class is provided
for scan-static reflection prediction for rotation data, where the predicted
centroid is expressed as X, Y, phi. Other versions of the class are defined
elsewhere."""

logger = logging.getLogger(__name__)

ParamSet = namedtuple(
    "ParamSet",
    ["beam_param", "xl_ori_param", "xl_uc_param", "det_param", "gonio_param"],
)


class PredictionParameterisation:
    """
    Abstract interface for a class that groups together model parameterisations
    relating to diffraction geometry and provides:

    * A list of all free parameters concatenated from each of the models, with a
      getter and setter method that delegates to the contained models
    * Derivatives of the reflection prediction equation with respect to each of
      these free parameters

    Derived classes determine whether the reflection prediction equation is
    expressed in detector space (X, Y, phi) or orthogonalised reciprocal space.

    It is assumed that the provided model parameterisations will be one of five
    types:

    * Detector parameterisation
    * Beam parameterisation
    * Crystal orientation parameterisation
    * Crystal unit cell parameterisation
    * Goniometer setting matrix parameterisation

    No more than one of each type must be supplied for each Experiment. Omitting
    a parameterisation for any of the types is allowed, which implies that the
    attached model is fixed in refinement of that Experiment.

    A class implementing PredictionParameterisation is used by a Refinery
    object directly, which takes the list of parameters, and indirectly via a
    Target function object, which takes the list of derivatives and composes the
    derivatives of a Target function from them.
    """

    def __init__(
        self,
        experiments,
        detector_parameterisations=None,
        beam_parameterisations=None,
        xl_orientation_parameterisations=None,
        xl_unit_cell_parameterisations=None,
        goniometer_parameterisations=None,
    ):
        if detector_parameterisations is None:
            detector_parameterisations = []
        if beam_parameterisations is None:
            beam_parameterisations = []
        if xl_orientation_parameterisations is None:
            xl_orientation_parameterisations = []
        if xl_unit_cell_parameterisations is None:
            xl_unit_cell_parameterisations = []
        if goniometer_parameterisations is None:
            goniometer_parameterisations = []

        # References to the underlying models
        self._experiments = experiments

        # Keep references to all parameterised models
        self._detector_parameterisations = detector_parameterisations
        self._beam_parameterisations = beam_parameterisations
        self._xl_orientation_parameterisations = xl_orientation_parameterisations
        self._xl_unit_cell_parameterisations = xl_unit_cell_parameterisations
        self._goniometer_parameterisations = goniometer_parameterisations

        self._update()

    def _update(self):
        """Update information derived from the parameterisations"""

        # Check there are free parameters to refine
        self._length = self._len()
        if self._length == 0:
            raise DialsRefineConfigError("There are no free parameters for refinement")

        # Calculate Experiment to parameterisation mapping
        e2bp = {
            ids: i
            for i, p in enumerate(self._beam_parameterisations)
            for ids in p.get_experiment_ids()
        }
        e2xop = {
            ids: i
            for i, p in enumerate(self._xl_orientation_parameterisations)
            for ids in p.get_experiment_ids()
        }
        e2xucp = {
            ids: i
            for i, p in enumerate(self._xl_unit_cell_parameterisations)
            for ids in p.get_experiment_ids()
        }
        e2dp = {
            ids: i
            for i, p in enumerate(self._detector_parameterisations)
            for ids in p.get_experiment_ids()
        }
        e2gp = {
            ids: i
            for i, p in enumerate(self._goniometer_parameterisations)
            for ids in p.get_experiment_ids()
        }

        self._exp_to_param = {
            i: ParamSet(
                e2bp.get(i), e2xop.get(i), e2xucp.get(i), e2dp.get(i), e2gp.get(i)
            )
            for i, _ in enumerate(self._experiments)
        }

    # accessors for the lists of parameterisations of different types
    def get_detector_parameterisations(self):
        return self._detector_parameterisations

    def get_beam_parameterisations(self):
        return self._beam_parameterisations

    def get_crystal_orientation_parameterisations(self):
        return self._xl_orientation_parameterisations

    def get_crystal_unit_cell_parameterisations(self):
        return self._xl_unit_cell_parameterisations

    def get_goniometer_parameterisations(self):
        return self._goniometer_parameterisations

    def _len(self):
        length = 0
        for model in self._detector_parameterisations:
            length += model.num_free()
        for model in self._beam_parameterisations:
            length += model.num_free()
        for model in self._xl_orientation_parameterisations:
            length += model.num_free()
        for model in self._xl_unit_cell_parameterisations:
            length += model.num_free()
        for model in self._goniometer_parameterisations:
            length += model.num_free()

        return length

    def __len__(self):
        return self._length

    def get_param_vals(self):
        """Return a concatenated list of parameters from each of the components
        in the global model"""

        global_p_list = []
        if self._detector_parameterisations:
            det_plists = [x.get_param_vals() for x in self._detector_parameterisations]
            params = [x for l in det_plists for x in l]
            global_p_list.extend(params)

        if self._beam_parameterisations:
            src_plists = [x.get_param_vals() for x in self._beam_parameterisations]
            params = [x for l in src_plists for x in l]
            global_p_list.extend(params)

        if self._xl_orientation_parameterisations:
            xlo_plists = [
                x.get_param_vals() for x in self._xl_orientation_parameterisations
            ]
            params = [x for l in xlo_plists for x in l]
            global_p_list.extend(params)

        if self._xl_unit_cell_parameterisations:
            xluc_plists = [
                x.get_param_vals() for x in self._xl_unit_cell_parameterisations
            ]
            params = [x for l in xluc_plists for x in l]
            global_p_list.extend(params)

        if self._goniometer_parameterisations:
            gon_plists = [
                x.get_param_vals() for x in self._goniometer_parameterisations
            ]
            params = [x for l in gon_plists for x in l]
            global_p_list.extend(params)

        return global_p_list

    def get_param_names(self):
        """Return a list of the names of parameters in the order they are
        concatenated. Use 1-based indexing for indices in the names."""

        param_names = []
        for p in self._detector_parameterisations:
            prefix = p.model_identifier
            param_names.extend([prefix + x for x in p.get_param_names()])

        for p in self._beam_parameterisations:
            prefix = p.model_identifier
            param_names.extend([prefix + x for x in p.get_param_names()])

        for p in self._xl_orientation_parameterisations:
            prefix = p.model_identifier
            param_names.extend([prefix + x for x in p.get_param_names()])

        for p in self._xl_unit_cell_parameterisations:
            prefix = p.model_identifier
            param_names.extend([prefix + x for x in p.get_param_names()])

        for p in self._goniometer_parameterisations:
            prefix = p.model_identifier
            param_names.extend([prefix + x for x in p.get_param_names()])

        return param_names

    def _modify_parameters(self, vals, set_vals=False, set_esds=False, set_fix=False):
        assert [set_vals, set_esds, set_fix].count(True) == 1
        assert len(vals) == len(self)
        it = iter(vals)

        for model in (
            self._detector_parameterisations
            + self._beam_parameterisations
            + self._xl_orientation_parameterisations
            + self._xl_unit_cell_parameterisations
            + self._goniometer_parameterisations
        ):
            tmp = [next(it) for i in range(model.num_free())]
            if set_esds:
                model.set_param_esds(tmp)
            elif set_vals:
                model.set_param_vals(tmp)
            elif set_fix:
                current_fixes = model.get_fixed()
                free_indices = [i for i, e in enumerate(current_fixes) if not e]
                if len(free_indices) != model.num_free():
                    logger.warning(
                        "Unable to fix parameters for parameterisation with prefix %s",
                        model.model_identifier,
                    )
                    continue
                for i, fix in zip(free_indices, tmp):
                    if fix:
                        current_fixes[i] = True
                model.set_fixed(current_fixes)

    def set_param_vals(self, vals):
        """Set the parameter values of the contained models to the values in
        vals. This list must be of the same length as the result of get_param_vals
        and must contain the parameter values in the same order."""

        self._modify_parameters(vals, set_vals=True)

    def set_param_esds(self, esds):
        """Set the estimated standard deviations of parameter values of the
        contained models to the values in esds. This list must be of the same length
        as the result of get_param_vals and must contain the parameter values in the
        same order."""

        self._modify_parameters(esds, set_esds=True)

    def fix_params(self, fix):
        """Fix the parameters according to the boolean values in fix. This list
        must be of the same length as the result of get_param_vals and will
        fix parameters at the positions that are True within the list"""

        self._modify_parameters(fix, set_fix=True)
        self._update()

    def calculate_model_state_uncertainties(self, var_cov):
        """Take a variance-covariance matrix of all free parameters (probably
        calculated by a minimisation engine). For each parameterisation in the
        global model, extract the subset of this matrix for the associated block
        of parameters. Pass this on to the relevant model parameterisation to
        calculate its own uncertainty of state."""

        i = 0
        for model in (
            self._detector_parameterisations
            + self._beam_parameterisations
            + self._xl_orientation_parameterisations
            + self._xl_unit_cell_parameterisations
            + self._goniometer_parameterisations
        ):
            n = model.num_free()
            sub = var_cov.matrix_copy_block(i, i, n, n)
            state_covs = model.calculate_state_uncertainties(sub)
            if state_covs is None:
                continue
            if len(state_covs) == 1:
                model.set_state_uncertainties(state_covs[0])
            else:
                for i_state, state_cov in enumerate(state_covs):
                    model.set_state_uncertainties(state_cov, multi_state_elt=i_state)
            i += n

    def get_gradients(self, reflections, callback=None):
        """Calculate gradients of the prediction formula with respect to each
        of the parameters of the contained models, for all of the reflections.
        This method sets up required quantities relevant to the current step of
        refinement and then loops over the parameterisations of each type extending
        a results list each time."""

        # Set up arrays of quantities of interest for each reflection
        self._nref = len(reflections)
        self._D = flex.mat3_double(self._nref)
        self._s0 = flex.vec3_double(self._nref)
        self._U = flex.mat3_double(self._nref)
        self._B = flex.mat3_double(self._nref)
        self._axis = flex.vec3_double(self._nref)
        self._fixed_rotation = flex.mat3_double(self._nref)
        self._setting_rotation = flex.mat3_double(self._nref)

        # Set up experiment to index mapping
        self._experiment_to_idx = []

        # Populate values in these arrays
        for iexp, exp in enumerate(self._experiments):
            sel = reflections["id"] == iexp
            isel = sel.iselection()
            self._experiment_to_idx.append(isel)
            subref = reflections.select(sel)
            states = self._get_model_data_for_experiment(exp, subref)

            self._D.set_selected(sel, states["D"])
            self._s0.set_selected(sel, states["s0"])
            self._U.set_selected(sel, states["U"])
            self._B.set_selected(sel, states["B"])
            if exp.goniometer:
                self._setting_rotation.set_selected(sel, states["S"])
                self._axis.set_selected(sel, exp.goniometer.get_rotation_axis_datum())
                self._fixed_rotation.set_selected(
                    sel, exp.goniometer.get_fixed_rotation()
                )

        # Other derived values
        self._h = reflections["miller_index"].as_vec3_double()
        self._UB = self._U * self._B
        self._s1 = reflections["s1"]
        self._pv = self._D * self._s1  # 'projection vector' for the ray along s1.

        # Quantities derived from pv, precalculated for efficiency
        u, v, w = self._pv.parts()
        assert w.all_ne(0)
        self._w_inv = 1.0 / w
        self._u_w_inv = u * self._w_inv
        self._v_w_inv = v * self._w_inv

        # Reset a pointer to the parameter number
        self._iparam = 0

        # Do additional setup specified by derived classes
        self._local_setup(reflections)

        # Set up empty list in which to store gradients
        results = []

        # loop over detector parameterisations and extend results
        results = self._grads_detector_loop(reflections, results, callback=callback)

        # loop over the beam parameterisations and extend results
        results = self._grads_beam_loop(reflections, results, callback=callback)

        # loop over the crystal orientation parameterisations and extend results
        results = self._grads_xl_orientation_loop(
            reflections, results, callback=callback
        )

        # loop over the crystal unit cell parameterisations and extend results
        results = self._grads_xl_unit_cell_loop(reflections, results, callback=callback)

        # loop over the goniometer parameterisations and extend results
        results = self._grads_goniometer_loop(reflections, results, callback=callback)

        return results

    @staticmethod
    def _extend_gradient_vectors(results, m, n, keys=("dX_dp", "dY_dp", "dZ_dp")):
        """Extend results list by n empty results. These will each be a dictionary
        indexed by the given keys. The value for each key will be an empty vector of
        size m, to store the derivatives of n parameters, for m reflections. This
        method may be overridden by a derived class to e.g. use sparse vectors"""

        new_results = []
        for i in range(n):
            result = {}
            for key in keys:
                result[key] = flex.double(m, 0.0)
            new_results.append(result)
        results.extend(new_results)

        return results

    def _get_model_data_for_experiment(self, experiment, reflections):
        """Helper function to return model data s0, U, B, D and S for a particular
        experiment. D is always returned as an array the same length as the
        reflections for the experiment, whereas here s0, U, B and S are returned as
        single matrices or vectors. In the scan-varying overload these will all be
        arrays."""

        # D matrix array
        D = flex.mat3_double(len(reflections))
        panels = reflections["panel"]
        for ipanel, D_mat in enumerate([p.get_D_matrix() for p in experiment.detector]):
            sel = panels == ipanel
            D.set_selected(sel, D_mat)

        if "s0" in reflections:
            s0 = reflections["s0"]
        else:
            s0 = experiment.beam.get_s0()

        result = {
            "s0": s0,
            "U": matrix.sqr(experiment.crystal.get_U()),
            "B": matrix.sqr(experiment.crystal.get_B()),
            "D": D,
        }

        # If a goniometer is present, get the setting matrix too
        if experiment.goniometer:
            result["S"] = matrix.sqr(experiment.goniometer.get_setting_rotation())

        return result

    # The detector derivatives calculation is shared by scans and stills type
    # prediction, so this method is here, in the base class.
    def _detector_derivatives(
        self, isel, panel_id, parameterisation=None, dd_ddet_p=None, reflections=None
    ):
        """Helper function to convert derivatives of the detector state to
        derivatives of the vector pv. Derivatives that would all be null vectors
        are replaced with None"""

        # Get required data
        pv = self._pv.select(isel)
        D = self._D.select(isel)

        if dd_ddet_p is None:
            # get the derivatives of detector d matrix for this panel
            dd_ddet_p = parameterisation.get_ds_dp(
                multi_state_elt=panel_id, use_none_as_null=True
            )

            # replace explicit null derivatives with None
            dd_ddet_p = [
                None if e is None else flex.mat3_double(len(D), e.elems)
                for e in dd_ddet_p
            ]

        # calculate the derivative of pv for this parameter
        dpv_ddet_p = [
            der if der is None else (D * (der * -1.0)) * pv for der in dd_ddet_p
        ]

        return dpv_ddet_p

    # The other model derivatives are different for scans and stills. Empty
    # implementations are declared here
    def _beam_derivatives(
        self, isel, parameterisation=None, ds0_dbeam_p=None, reflections=None
    ):
        pass

    def _xl_orientation_derivatives(
        self, isel, parameterisation=None, dU_dxlo_p=None, reflections=None
    ):
        pass

    def _xl_unit_cell_derivatives(
        self, isel, parameterisation=None, dB_dxluc_p=None, reflections=None
    ):
        pass

    def _goniometer_derivatives(
        self, isel, parameterisation=None, dS_dgon_p=None, reflections=None
    ):
        pass

    def _grads_detector_loop(self, reflections, results, callback=None):
        """Loop over all detector parameterisations, calculate gradients and extend
        the results"""

        # loop over the detector parameterisations
        for dp in self._detector_parameterisations:
            # Determine (sub)set of reflections affected by this parameterisation
            isel = flex.size_t()
            for exp_id in dp.get_experiment_ids():
                isel.extend(self._experiment_to_idx[exp_id])

            # Access the detector model being parameterised
            detector = dp.get_model()

            # Get panel numbers of the affected reflections
            panel = reflections["panel"].select(isel)

            # Extend derivative vectors for this detector parameterisation
            results = self._extend_gradient_vectors(
                results, self._nref, dp.num_free(), keys=self._grad_names
            )

            # loop through the panels in this detector
            for panel_id, _ in enumerate(detector):
                # get the right subset of array indices to set for this panel
                sub_isel = isel.select(panel == panel_id)
                if len(sub_isel) == 0:
                    # if no reflections intersect this panel, skip calculation
                    continue

                dpv_ddet_p = self._detector_derivatives(
                    sub_isel, panel_id, parameterisation=dp, reflections=reflections
                )

                # convert to dX/dp, dY/dp and assign the elements of the vectors
                # corresponding to this experiment and panel
                sub_w_inv = self._w_inv.select(sub_isel)
                sub_u_w_inv = self._u_w_inv.select(sub_isel)
                sub_v_w_inv = self._v_w_inv.select(sub_isel)
                dX_ddet_p, dY_ddet_p = self._calc_dX_dp_and_dY_dp_from_dpv_dp(
                    sub_w_inv, sub_u_w_inv, sub_v_w_inv, dpv_ddet_p
                )

                # use a local parameter index pointer because we set all derivatives
                # for this panel before moving on to the next
                iparam = self._iparam
                for dX, dY in zip(dX_ddet_p, dY_ddet_p):
                    if dX is not None:
                        try:
                            dX, indices = dX.data_and_indices
                            indices = sub_isel.select(indices)
                        except AttributeError:
                            indices = sub_isel
                        results[iparam][self._grad_names[0]].set_selected(indices, dX)
                    if dY is not None:
                        try:
                            dY, indices = dY.data_and_indices
                            indices = sub_isel.select(indices)
                        except AttributeError:
                            indices = sub_isel
                        results[iparam][self._grad_names[1]].set_selected(indices, dY)
                    # increment the local parameter index pointer
                    iparam += 1

            if callback is not None:
                iparam = self._iparam
                for i in range(dp.num_free()):
                    results[iparam] = callback(results[iparam])
                    iparam += 1

            # increment the parameter index pointer to the last detector parameter
            self._iparam += dp.num_free()

        return results

    def _grads_model_loop(
        self,
        parameterisations,
        reflections,
        results,
        callback=None,
        derivatives_fn=None,
    ):
        # loop over the parameterisations
        for p in parameterisations:
            # Determine (sub)set of reflections affected by this parameterisation
            isel = flex.size_t()
            for exp_id in p.get_experiment_ids():
                isel.extend(self._experiment_to_idx[exp_id])

            # Extend derivative vectors for this parameterisation
            results = self._extend_gradient_vectors(
                results, self._nref, p.num_free(), keys=self._grad_names
            )

            if len(isel) == 0:
                # if no reflections are in this experiment, skip calculation of
                # gradients, but must still process null gradients by a callback
                if callback:
                    for _ in range(p.num_free()):
                        results[self._iparam] = callback(results[self._iparam])
                        self._iparam += 1
                else:
                    self._iparam += p.num_free()
                continue

            w_inv = self._w_inv.select(isel)
            u_w_inv = self._u_w_inv.select(isel)
            v_w_inv = self._v_w_inv.select(isel)

            dpv_dp, dAngle_dp = derivatives_fn(
                isel, parameterisation=p, reflections=reflections
            )

            # convert to dX/dp, dY/dp and assign the elements of the vectors
            # corresponding to this experiment
            dX_dp, dY_dp = self._calc_dX_dp_and_dY_dp_from_dpv_dp(
                w_inv, u_w_inv, v_w_inv, dpv_dp
            )

            # Loop over gradients and paste them into the correct rows in the
            # results arrays, taking care of whether the gradients are flex
            # or SparseFlex arrays.
            for dX, dY, dAngle in zip(dX_dp, dY_dp, dAngle_dp):
                if dX is not None:
                    try:
                        dX, indices = dX.data_and_indices
                        indices = isel.select(indices)
                    except AttributeError:
                        indices = isel
                    results[self._iparam][self._grad_names[0]].set_selected(indices, dX)
                if dY is not None:
                    try:
                        dY, indices = dY.data_and_indices
                        indices = isel.select(indices)
                    except AttributeError:
                        indices = isel
                    results[self._iparam][self._grad_names[1]].set_selected(indices, dY)
                if dAngle is not None:
                    try:
                        dAngle, indices = dAngle.data_and_indices
                        indices = isel.select(indices)
                    except AttributeError:
                        indices = isel
                    results[self._iparam][self._grad_names[2]].set_selected(
                        indices, dAngle
                    )
                if callback is not None:
                    results[self._iparam] = callback(results[self._iparam])
                # increment the parameter index pointer
                self._iparam += 1

        return results

    def _grads_beam_loop(self, reflections, results, callback=None):
        """Loop over all beam parameterisations, calculate gradients and extend
        the results"""

        return self._grads_model_loop(
            self._beam_parameterisations,
            reflections,
            results,
            derivatives_fn=self._beam_derivatives,
            callback=callback,
        )

    def _grads_xl_orientation_loop(self, reflections, results, callback=None):
        """Loop over all crystal orientation parameterisations, calculate gradients
        and extend the results"""

        return self._grads_model_loop(
            self._xl_orientation_parameterisations,
            reflections,
            results,
            derivatives_fn=self._xl_orientation_derivatives,
            callback=callback,
        )

    def _grads_xl_unit_cell_loop(self, reflections, results, callback=None):
        """Loop over all crystal unit cell parameterisations, calculate gradients
        and extend the results"""

        return self._grads_model_loop(
            self._xl_unit_cell_parameterisations,
            reflections,
            results,
            derivatives_fn=self._xl_unit_cell_derivatives,
            callback=callback,
        )

    def _grads_goniometer_loop(self, reflections, results, callback=None):
        """Loop over all goniometer parameterisations, calculate gradients
        and extend the results"""

        return self._grads_model_loop(
            self._goniometer_parameterisations,
            reflections,
            results,
            derivatives_fn=self._goniometer_derivatives,
            callback=callback,
        )


class SparseGradientVectorMixin:
    """Mixin class to use sparse vectors for storage of gradients of the
    prediction formula"""

    @staticmethod
    def _extend_gradient_vectors(results, m, n, keys=("dX_dp", "dY_dp", "dZ_dp")):
        """Extend results list by n empty results. These will each be a dictionary
        indexed by the given keys. The value for each key will be an empty vector of
        size m, to store the derivatives of n parameters, for m reflections.
        This is the sparse vector version."""

        new_results = [{key: sparse.matrix_column(m) for key in keys} for _ in range(n)]
        results.extend(new_results)

        return results


class XYPhiPredictionParameterisation(PredictionParameterisation):
    _grad_names = ("dX_dp", "dY_dp", "dphi_dp")

    def _local_setup(self, reflections):
        """Setup additional attributes used in gradients calculation. These are
        specific to scans-type prediction parameterisations"""

        # Spindle rotation matrices for every reflection
        # R = self._axis.axis_and_angle_as_r3_rotation_matrix(phi)
        # R = flex.mat3_double(len(reflections))
        # NB for now use flex.vec3_double.rotate_around_origin each time I need the
        # rotation matrix R.

        # r is the reciprocal lattice vector, in the lab frame
        self._phi_calc = reflections["xyzcal.mm"].parts()[2]
        q = self._fixed_rotation * (self._UB * self._h)
        self._r = self._setting_rotation * q.rotate_around_origin(
            self._axis, self._phi_calc
        )

        # All of the derivatives of phi have a common denominator, given by
        # (e X r).s0, where e is the rotation axis. Calculate this once, here.
        self._e_X_r = (self._setting_rotation * self._axis).cross(self._r)
        self._e_r_s0 = (self._e_X_r).dot(self._s0)

        # Note that e_r_s0 -> 0 when the rotation axis, beam vector and
        # relp are coplanar. This occurs when a reflection just touches
        # the Ewald sphere.
        #
        # There is a relationship between e_r_s0 and zeta_factor.
        # Uncommenting the code below shows that
        # s0.(e X r) = zeta * |s X s0|

        # from dials.algorithms.profile_model.gaussian_rs import zeta_factor
        # from libtbx.test_utils import approx_equal
        # s = matrix.col(reflections['s1'][0])
        # z = zeta_factor(axis[0], s0[0], s)
        # ss0 = (s.cross(matrix.col(s0[0]))).length()
        # assert approx_equal(e_r_s0[0], z * ss0)

        # catch small values of e_r_s0
        e_r_s0_mag = flex.abs(self._e_r_s0)
        try:
            assert flex.min(e_r_s0_mag) > 1.0e-6
        except AssertionError as e:
            imin = flex.min_index(e_r_s0_mag)
            print("(e X r).s0 too small:")
            print("for", (e_r_s0_mag <= 1.0e-6).count(True), "reflections")
            print("out of", len(e_r_s0_mag), "total")
            print("such as", reflections["miller_index"][imin])
            print("with scattering vector", reflections["s1"][imin])
            print("where r =", self._r[imin])
            print("e =", self._axis[imin])
            print("s0 =", self._s0[imin])
            print("this reflection forms angle with the equatorial plane " "normal:")
            vecn = (
                matrix.col(self._s0[imin])
                .cross(matrix.col(self._axis[imin]))
                .normalize()
            )
            print(matrix.col(reflections["s1"][imin]).accute_angle(vecn))
            raise e

    def _beam_derivatives(
        self, isel, parameterisation=None, ds0_dbeam_p=None, reflections=None
    ):
        """helper function to extend the derivatives lists by derivatives of the
        beam parameterisations."""

        # Get required data
        r = self._r.select(isel)
        e_X_r = self._e_X_r.select(isel)
        e_r_s0 = self._e_r_s0.select(isel)
        D = self._D.select(isel)

        if ds0_dbeam_p is None:
            # get the derivatives of the beam vector wrt the parameters
            ds0_dbeam_p = parameterisation.get_ds_dp(use_none_as_null=True)

            ds0_dbeam_p = [
                None if e is None else flex.vec3_double(len(r), e.elems)
                for e in ds0_dbeam_p
            ]

        dphi_dp = []
        dpv_dp = []

        # loop through the parameters
        for der in ds0_dbeam_p:
            if der is None:
                dphi_dp.append(None)
                dpv_dp.append(None)
                continue

            # calculate the derivative of phi for this parameter
            dphi = (der.dot(r) / e_r_s0) * -1.0
            dphi_dp.append(dphi)

            # calculate the derivative of pv for this parameter
            dpv_dp.append(D * (e_X_r * dphi + der))

        return dpv_dp, dphi_dp

    def _xl_derivatives(self, isel, derivatives, b_matrix, parameterisation=None):
        """helper function to extend the derivatives lists by derivatives of
        generic parameterisations."""

        # Get required data
        axis = self._axis.select(isel)
        fixed_rotation = self._fixed_rotation.select(isel)
        setting_rotation = self._setting_rotation.select(isel)
        phi_calc = self._phi_calc.select(isel)
        h = self._h.select(isel)
        s1 = self._s1.select(isel)
        e_X_r = self._e_X_r.select(isel)
        e_r_s0 = self._e_r_s0.select(isel)
        if b_matrix:
            B = self._B.select(isel)
        else:
            U = self._U.select(isel)
        D = self._D.select(isel)

        if derivatives is None:
            # get derivatives of the B/U matrix wrt the parameters
            derivatives = [
                None if der is None else flex.mat3_double(len(isel), der.elems)
                for der in parameterisation.get_ds_dp(use_none_as_null=True)
            ]

        dphi_dp = []
        dpv_dp = []

        # loop through the parameters
        for der in derivatives:
            if der is None:
                dphi_dp.append(None)
                dpv_dp.append(None)
                continue

            # calculate the derivative of r for this parameter
            if b_matrix:
                tmp = fixed_rotation * (der * B * h)
            else:
                tmp = fixed_rotation * (U * der * h)
            dr = setting_rotation * tmp.rotate_around_origin(axis, phi_calc)

            # calculate the derivative of phi for this parameter
            dphi = -1.0 * dr.dot(s1) / e_r_s0
            dphi_dp.append(dphi)

            # calculate the derivative of pv for this parameter
            dpv_dp.append(D * (dr + e_X_r * dphi))

        return dpv_dp, dphi_dp

    def _xl_orientation_derivatives(
        self, isel, parameterisation=None, dU_dxlo_p=None, reflections=None
    ):
        """helper function to extend the derivatives lists by derivatives of the
        crystal orientation parameterisations"""
        return self._xl_derivatives(
            isel, dU_dxlo_p, b_matrix=True, parameterisation=parameterisation
        )

    def _xl_unit_cell_derivatives(
        self, isel, parameterisation=None, dB_dxluc_p=None, reflections=None
    ):
        """helper function to extend the derivatives lists by
        derivatives of the crystal unit cell parameterisations"""
        return self._xl_derivatives(
            isel, dB_dxluc_p, b_matrix=False, parameterisation=parameterisation
        )

    def _goniometer_derivatives(
        self, isel, parameterisation=None, dS_dgon_p=None, reflections=None
    ):
        """helper function to extend the derivatives lists by
        derivatives of the goniometer parameterisations"""

        # Get required data
        axis = self._axis.select(isel)
        fixed_rotation = self._fixed_rotation.select(isel)
        phi_calc = self._phi_calc.select(isel)
        h = self._h.select(isel)
        s1 = self._s1.select(isel)
        e_X_r = self._e_X_r.select(isel)
        e_r_s0 = self._e_r_s0.select(isel)
        UB = self._UB.select(isel)
        D = self._D.select(isel)

        if dS_dgon_p is None:
            # get derivatives of the setting matrix S wrt the parameters
            dS_dgon_p = [
                None if der is None else flex.mat3_double(len(isel), der.elems)
                for der in parameterisation.get_ds_dp(use_none_as_null=True)
            ]

        dphi_dp = []
        dpv_dp = []

        # loop through the parameters
        for der in dS_dgon_p:
            if der is None:
                dphi_dp.append(None)
                dpv_dp.append(None)
                continue

            # calculate the derivative of r for this parameter
            tmp = fixed_rotation * (UB * h)
            dr = der * tmp.rotate_around_origin(axis, phi_calc)

            # calculate the derivative of phi for this parameter
            dphi = -1.0 * dr.dot(s1) / e_r_s0
            dphi_dp.append(dphi)

            # calculate the derivative of pv for this parameter
            dpv_dp.append(D * (dr + e_X_r * dphi))

        return dpv_dp, dphi_dp

    @staticmethod
    def _calc_dX_dp_and_dY_dp_from_dpv_dp(w_inv, u_w_inv, v_w_inv, dpv_dp):
        """helper function to calculate positional derivatives from
        dpv_dp using the quotient rule"""

        dX_dp = []
        dY_dp = []

        for der in dpv_dp:
            if der is None:
                dX_dp.append(None)
                dY_dp.append(None)
            else:
                du_dp, dv_dp, dw_dp = der.parts()

                dX_dp.append(w_inv * (du_dp - dw_dp * u_w_inv))
                dY_dp.append(w_inv * (dv_dp - dw_dp * v_w_inv))

        return dX_dp, dY_dp


class XYPhiPredictionParameterisationSparse(
    SparseGradientVectorMixin, XYPhiPredictionParameterisation
):
    """A version of XYPhiPredictionParameterisation that uses a sparse matrix
    data structure for memory efficiency when there are a large number of
    Experiments"""

    # Explicitly delegate to the method from SparseGradientVectorMixin
    # (https://lgtm.com/rules/7860084/)
    @staticmethod
    def _extend_gradient_vectors(results, m, n, keys=("dX_dp", "dY_dp", "dZ_dp")):
        return SparseGradientVectorMixin._extend_gradient_vectors(results, m, n, keys)


class LauePredictionParameterisation(PredictionParameterisation):
    """A basic extension to PredictionParameterisation for Laue data,
    where gradients for the wavelength of each reflection are also considered."""

    _grad_names = ("dX_dp", "dY_dp", "dwavelength_dp")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def _local_setup(self, reflections):
        self._wavelength = reflections["wavelength_cal"]
        self._r = self._setting_rotation * self._fixed_rotation * self._UB * self._h
        self._s0 = reflections["s0_cal"]
        self._e_X_r = (self._setting_rotation * self._axis).cross(self._r)
        self._e_r_s0 = (self._e_X_r).dot(self._s0)
        self._ds0_dbeam_p = None
        return

    def get_gradients(self, reflections, callback=None):
        """Calculate gradients of the prediction formula with respect to each
        of the parameters of the contained models, for all of the reflections.
        This method sets up required quantities relevant to the current step of
        refinement and then loops over the parameterisations of each type extending
        a results list each time."""

        # Set up arrays of quantities of interest for each reflection
        self._nref = len(reflections)
        self._D = flex.mat3_double(self._nref)
        self._s0 = flex.vec3_double(self._nref)
        self._U = flex.mat3_double(self._nref)
        self._B = flex.mat3_double(self._nref)
        self._axis = flex.vec3_double(self._nref, (0.0, 1.0, 0.0))
        self._fixed_rotation = flex.mat3_double(self._nref, (1, 0, 0, 0, 1, 0, 0, 0, 1))
        self._setting_rotation = flex.mat3_double(
            self._nref, (1, 0, 0, 0, 1, 0, 0, 0, 1)
        )

        # Set up experiment to index mapping
        self._experiment_to_idx = []

        # Populate values in these arrays
        for iexp, exp in enumerate(self._experiments):
            sel = reflections["id"] == iexp
            isel = sel.iselection()
            self._experiment_to_idx.append(isel)
            subref = reflections.select(sel)
            states = self._get_model_data_for_experiment(exp, subref)

            self._D.set_selected(sel, states["D"])
            self._s0.set_selected(sel, states["s0"])
            self._U.set_selected(sel, states["U"])
            self._B.set_selected(sel, states["B"])
            if exp.goniometer:
                self._setting_rotation.set_selected(sel, states["S"])
                self._axis.set_selected(sel, exp.goniometer.get_rotation_axis_datum())
                self._fixed_rotation.set_selected(
                    sel, exp.goniometer.get_fixed_rotation()
                )

        # Other derived values
        self._h = reflections["miller_index"].as_vec3_double()
        self._UB = self._U * self._B
        self._s1 = reflections["s1"]
        self._pv = self._D * self._s1  # 'projection vector' for the ray along s1.

        # Quantities derived from pv, precalculated for efficiency
        u, v, w = self._pv.parts()
        assert w.all_ne(0)
        self._w_inv = 1.0 / w
        self._u_w_inv = u * self._w_inv
        self._v_w_inv = v * self._w_inv

        # Reset a pointer to the parameter number
        self._iparam = 0

        # Do additional setup specified by derived classes
        self._local_setup(reflections)

        # Set up empty list in which to store gradients
        results = []

        # loop over detector parameterisations and extend results
        results = self._grads_detector_loop(reflections, results, callback=callback)

        # loop over the beam parameterisations and extend results
        results = self._grads_beam_loop(reflections, results, callback=callback)

        # loop over the crystal orientation parameterisations and extend results
        results = self._grads_xl_orientation_loop(
            reflections, results, callback=callback
        )

        # loop over the crystal unit cell parameterisations and extend results
        results = self._grads_xl_unit_cell_loop(reflections, results, callback=callback)

        # loop over the goniometer parameterisations and extend results
        results = self._grads_goniometer_loop(reflections, results, callback=callback)

        return results

    def _beam_derivatives(
        self, isel, parameterisation=None, ds0_dbeam_p=None, reflections=None
    ):
        """helper function to extend the derivatives lists by derivatives of the
        beam parameterisations."""

        # Get required data
        r = self._r.select(isel)
        e_X_r = self._e_X_r.select(isel)
        e_r_s0 = self._e_r_s0.select(isel)
        D = self._D.select(isel)

        if ds0_dbeam_p is None:
            # get the derivatives of the beam vector wrt the parameters
            ds0_dbeam_p = parameterisation.get_ds_dp(use_none_as_null=True)

            ds0_dbeam_p = [
                None if e is None else flex.vec3_double(len(r), e.elems)
                for e in ds0_dbeam_p
            ]
            self._ds0_dbeam_p = ds0_dbeam_p

        dphi_dp = []
        dpv_dp = []

        # loop through the parameters
        for der in ds0_dbeam_p:
            if der is None:
                dphi_dp.append(None)
                dpv_dp.append(None)
                continue

            # calculate the derivative of phi for this parameter
            dphi = (der.dot(r) / e_r_s0) * -1.0
            dphi_dp.append(dphi)

            # calculate the derivative of pv for this parameter
            dpv_dp.append(D * (e_X_r * dphi + der))

        return dpv_dp, dphi_dp

    def _xl_derivatives(self, isel, derivatives, b_matrix, parameterisation=None):
        """helper function to extend the derivatives lists by derivatives of
        generic parameterisations."""

        # Get required data
        h = self._h.select(isel)
        if b_matrix:
            B = self._B.select(isel)
        else:
            U = self._U.select(isel)
        D = self._D.select(isel)
        wavelength = self._wavelength.select(isel)
        fixed_rotation = self._fixed_rotation.select(isel)
        setting_rotation = self._setting_rotation.select(isel)
        r = self._r.select(isel)

        if derivatives is None:
            # get derivatives of the B/U matrix wrt the parameters
            derivatives = [
                None if der is None else flex.mat3_double(len(isel), der.elems)
                for der in parameterisation.get_ds_dp(use_none_as_null=True)
            ]

        dpv_dp = []
        dwavelength_dp = []

        # loop through the parameters
        for idx, der in enumerate(derivatives):
            if der is None:
                dpv_dp.append(None)
                dwavelength_dp.append(None)
                continue

            # calculate the derivative of r for this parameter
            if b_matrix:
                dr = setting_rotation * fixed_rotation * der * B * h
            else:
                dr = setting_rotation * fixed_rotation * U * der * h

            unit_s0 = flex.vec3_double(len(dr), self._experiments[0].beam.get_unit_s0())
            r_dot_r = r.dot(r)
            dwavelength = -2 * unit_s0.dot(
                ((dr * r_dot_r) - ((2 * r) * dr.dot(r))) / (r_dot_r**2)
            )
            dwavelength_dp.append(dwavelength)

            # calculate the derivative of pv for this parameter
            dpv_dp.append(D * (dr - (unit_s0 / wavelength**2) * dwavelength))

        return dpv_dp, dwavelength_dp

    def _xl_orientation_derivatives(
        self, isel, parameterisation=None, dU_dxlo_p=None, reflections=None
    ):
        """helper function to extend the derivatives lists by derivatives of the
        crystal orientation parameterisations"""
        return self._xl_derivatives(
            isel, dU_dxlo_p, b_matrix=True, parameterisation=parameterisation
        )

    def _xl_unit_cell_derivatives(
        self, isel, parameterisation=None, dB_dxluc_p=None, reflections=None
    ):
        """helper function to extend the derivatives lists by
        derivatives of the crystal unit cell parameterisations"""
        return self._xl_derivatives(
            isel, dB_dxluc_p, b_matrix=False, parameterisation=parameterisation
        )

    @staticmethod
    def _calc_dX_dp_and_dY_dp_from_dpv_dp(w_inv, u_w_inv, v_w_inv, dpv_dp):
        """helper function to calculate positional derivatives from
        dpv_dp using the quotient rule"""

        dX_dp = []
        dY_dp = []

        for der in dpv_dp:
            if der is None:
                dX_dp.append(None)
                dY_dp.append(None)
            else:
                du_dp, dv_dp, dw_dp = der.parts()

                dX_dp.append(w_inv * (du_dp - dw_dp * u_w_inv))
                dY_dp.append(w_inv * (dv_dp - dw_dp * v_w_inv))

        return dX_dp, dY_dp
