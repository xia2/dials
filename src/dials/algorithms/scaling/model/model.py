"""
Definitions of scaling models.

A scaling model is a collection of scaling model components with appropriate
methods to define how these are composed into one model.
"""

from __future__ import annotations

import importlib.metadata
import logging
import math

from libtbx import Auto, phil

from dials.algorithms.scaling.error_model.error_model import BasicErrorModel
from dials.algorithms.scaling.model.components.analytical_component import (
    AnalyticalComponent,
)
from dials.algorithms.scaling.model.components.scale_components import (
    LinearDoseDecay,
    QuadraticDoseDecay,
    SHScaleComponent,
    SingleBScaleFactor,
    SingleScaleFactor,
)
from dials.algorithms.scaling.model.components.smooth_scale_components import (
    SmoothBScaleComponent1D,
    SmoothScaleComponent1D,
    SmoothScaleComponent2D,
    SmoothScaleComponent3D,
)
from dials.algorithms.scaling.plots import (
    plot_absorption_parameters,
    plot_absorption_plots,
    plot_array_absorption_plot,
    plot_array_decay_plot,
    plot_array_modulation_plot,
    plot_dose_decay,
    plot_relative_Bs,
    plot_smooth_scales,
)
from dials.algorithms.scaling.scaling_utilities import sph_harm_table
from dials.array_family import flex
from dials_scaling_ext import (
    calc_lookup_index,
    calc_theta_phi,
    create_sph_harm_lookup_table,
)

logger = logging.getLogger("dials")

base_model_phil_str = """\
correction.fix = None
    .type = strings
    .help = "If specified, this correction will not be refined in this scaling run"
analytical_correction = False
    .type = bool
    .help = "If True, the "analytical_correction" column from the reflection table will"
            "be used as a fixed correction during scaling (in addition to any corrections"
            "specified as fixed with the option correction.fix=)"
"""

kb_model_phil_str = (
    """\
decay_correction = True
    .type = bool
    .help = "Option to turn off decay correction (for physical/array/KB
            default models)."
    .expert_level = 1
"""
    + base_model_phil_str
)

dose_decay_model_phil_str = (
    """\
scale_interval = 2.0
    .type = float(value_min=1.0)
    .help = "Rotation (phi) interval between model parameters for the scale"
            "component."
    .expert_level = 1
relative_B_correction = True
    .type = bool
    .help = "Option to turn off relative B correction."
    .expert_level = 1
decay_correction = True
    .type = bool
    .help = "Option to turn off decay correction."
    .expert_level = 1
share.decay = True
    .type = bool
    .help = "Share the decay model between sweeps."
    .expert_level = 1
resolution_dependence = *quadratic linear
    .type = choice
    .help = "Use a dose model that depends linearly or quadratically on 1/d"
    .expert_level = 1
absorption_correction = False
    .type = bool
    .help = "Option to turn on spherical harmonic absorption correction."
    .expert_level = 1
lmax = 4
    .type = int(value_min=2)
    .help = "Number of spherical harmonics to include for absorption"
            "correction, recommended to be no more than 6."
    .expert_level = 2
surface_weight = 1e6
    .type = float(value_min=0.0)
    .help = "Restraint weight applied to spherical harmonic terms in the"
            "absorption correction."
    .expert_level = 2
fix_initial = True
    .type = bool
    .help = "If performing full matrix minimisation, in the final cycle,"
            "constrain the initial parameter for more reliable parameter and"
            "scale factor error estimates."
    .expert_level = 2
"""
    + base_model_phil_str
)


physical_model_phil_str = (
    """\
scale_interval = auto
    .type = float(value_min=1.0)
    .help = "Rotation (phi) interval between model parameters for the scale"
            "component (auto scales interval depending on oscillation range)."
    .expert_level = 1
decay_correction = True
    .type = bool
    .help = "Option to turn off decay correction."
    .expert_level = 1
decay_interval = auto
    .type = float(value_min=1.0)
    .help = "Rotation (phi) interval between model parameters for the decay"
            "component (auto scales interval depending on oscillation range)."
    .expert_level = 1
decay_restraint = 1e-1
    .type = float(value_min=0.0)
    .help = "Weight to weakly restrain B-values to 0."
    .expert_level = 2
absorption_correction = auto
    .type = bool
    .help = "Option to turn off absorption correction (default True if oscillation > 60.0)."
    .expert_level = 1
absorption_level = low medium high
    .type = choice
    .help = "Expected degree of relative absorption for different scattering"
            "paths through the crystal(s). If an option is selected, the"
            "scaling model parameters lmax and surface_weight will be set to"
            "appropriate values."
            "Relative absorption increases as crystal size increases,"
            "increases as wavelength increases and is increased as the crystal"
            "dimensions become less equal (i.e. is higher for needle shaped"
            "crystals and zero for a spherical crystal)."
            "Definitions of the levels and approximate correction magnitude:"
            "low:    ~1%% relative absorption, expected for typical protein"
            "        crystals (containing no strongly absorbing atoms) on the"
            "        order of ~100um measured at ~1A wavelength."
            "medium: ~5%% relative absorption"
            "high:   >25%% relative absorption, e.g. for measurements at long"
            "        wavelength or crystals with high absorption from heavy atoms."
    .expert_level = 1
lmax = auto
    .type = int(value_min=2)
    .help = "Number of spherical harmonics to include for absorption"
            "correction, defaults to 4 if no absorption_level is chosen."
            "It is recommended that the value need be no more than 6."
    .expert_level = 1
surface_weight = auto
    .type = float(value_min=0.0)
    .help = "Restraint weight applied to spherical harmonic terms in the"
            "absorption correction. A lower restraint allows a higher amount"
            "of absorption correction. Defaults to 5e5 if no absorption_level"
            "is chosen."
    .expert_level = 1
share.absorption = False
    .type = bool
    .help = "If True, a common absorption correction is refined across all sweeps".
fix_initial = True
    .type = bool
    .help = "If performing full matrix minimisation, in the final cycle,"
            "constrain the initial parameter for more reliable parameter and"
            "scale factor error estimates."
    .expert_level = 2
"""
    + base_model_phil_str
)

array_model_phil_str = (
    """\
decay_correction = True
    .type = bool
    .help = "Option to turn off decay correction (a 2D grid of parameters as"
            "a function of rotation and resolution (d-value))."
    .expert_level = 1
decay_interval = 20.0
    .type = float(value_min=1.0)
    .help = "Rotation (phi) interval between model parameters for the decay"
            "and absorption corrections."
    .expert_level = 1
n_resolution_bins = 10
    .type = int(value_min=1)
    .help = "Number of resolution bins to use for the decay term."
    .expert_level = 1
absorption_correction = True
    .type = bool
    .help = "Option to turn off absorption correction (a 3D grid of"
            "parameters as a function of rotation angle, detector-x and"
            "detector-y position)."
    .expert_level = 1
n_absorption_bins = 3
    .type = int(value_min=1)
    .help = "Number of bins in each dimension (applied to both x and y) for"
            "binning the detector position for the absorption term of the"
            "array model."
    .expert_level = 1
modulation_correction = False
    .type = bool
    .help = "Option to turn on a detector correction for the array default"
            "model."
    .expert_level = 2
n_modulation_bins = 20
    .type = int(value_min=1)
    .help = "Number of bins in each dimension (applied to both x and y) for"
            "binning the detector position for the modulation correction."
    .expert_level = 2
"""
    + base_model_phil_str
)

autos = [Auto, "auto", "Auto"]


class ScalingModelBase:
    """Abstract base class for scaling models."""

    id_ = None

    def __init__(self, configdict, is_scaled=False):
        """Initialise the model with no components and a :obj:`configdict`."""
        if not configdict["corrections"]:
            raise ValueError("No model components created.")
        self._components = {}
        self._configdict = configdict
        self._is_scaled = is_scaled
        self._error_model = None
        self._fixed_components = []

    @property
    def is_scaled(self):
        """:obj:`bool`: Indicate whether this model has previously been refined."""
        return self._is_scaled

    def fix_initial_parameter(self, params):
        """Fix a parameter of the scaling model."""
        return False

    @property
    def fixed_components(self):
        return self._fixed_components

    @fixed_components.setter
    def fixed_components(self, components):
        self._fixed_components = components

    def limit_image_range(self, new_image_range):
        """Modify the model if necessary due to reducing the image range.

        Args:
            new_image_range (tuple): The (start, end) of the new image range.
        """
        pass

    def set_scaling_model_as_scaled(self):
        """Set the boolean 'is_scaled' flag as True."""
        self._is_scaled = True

    def set_scaling_model_as_unscaled(self):
        """Set the boolean 'is_scaled' flag as False."""
        self._is_scaled = False

    def configure_components(self, reflection_table, experiment, params):
        """Add the required reflection table data to the model components."""
        raise NotImplementedError()

    def plot_model_components(self, reflection_table=None):
        """Return a dict of plots for plotting model components with plotly."""
        return {}

    @classmethod
    def from_data(cls, params, experiment, reflection_table):
        """Create the model from input data."""
        raise NotImplementedError()

    def set_valid_image_range(self, image_range):
        """Set the valid image range for the model in the :obj:`configdict`."""
        self._configdict["valid_image_range"] = image_range

    @property
    def error_model(self):
        """:obj:`error_model`: The error model associated with the scaling model."""
        return self._error_model

    @property
    def configdict(self):
        """:obj:`dict`: a dictionary of the model configuration parameters."""
        return self._configdict

    @property
    def components(self):
        """:obj:`dict`: a dictionary of the model components."""
        return self._components

    @property
    def n_params(self):
        """:obj:`dict`: a dictionary of the model components."""
        return sum(c.parameters.size() for c in self._components.values())

    def consecutive_refinement_order(self):
        """:obj:`list`: a nested list of component names.

        This list indicates to the scaler the order to perform scaling in
        consecutive scaling mode.
        e.g. [['scale', 'decay'], ['absorption']] would cause the first cycle to
        refine scale and decay, and then absorption in a subsequent cycle.
        """
        raise NotImplementedError()

    def to_dict(self):
        """Serialize the model to a dictionary.

        Returns:
            dict: A dictionary representation of the model.
        """
        dictionary = {"__id__": self.id_}
        for key in self.components:
            dictionary[key] = {
                "n_parameters": self._components[key].n_params,
                "parameters": list(self._components[key].parameters),
                "null_parameter_value": self._components[key].null_parameter_value,
            }
            if self._components[key].parameter_esds:
                dictionary[key]["est_standard_devs"] = list(
                    self._components[key].parameter_esds
                )
        dictionary["configuration_parameters"] = self._configdict
        return dictionary

    @classmethod
    def from_dict(cls, obj):
        """Create a scaling model from a dictionary."""
        raise NotImplementedError()

    def update(self, model_params):
        pass

    def load_error_model(self, error_params):
        # load existing model if there, but use user-specified values if given
        new_model = None
        if (
            "error_model_type" in self._configdict
            and not error_params.reset_error_model
        ):
            if self._configdict["error_model_type"] == "BasicErrorModel":
                p = self._configdict["error_model_parameters"]
                a = None
                b = None
                if not error_params.basic.a:
                    a = p[0]
                if not error_params.basic.b:
                    b = p[1]
                new_model = BasicErrorModel(a, b, error_params.basic)
        if not new_model:
            new_model = BasicErrorModel(basic_params=error_params.basic)
        logger.info(f"Loaded error model: {new_model}")
        self.set_error_model(new_model)

    def set_error_model(self, error_model):
        """Associate an error model with the dataset."""
        self._error_model = error_model
        self._configdict.update(
            {
                "error_model_type": self.error_model.__class__.__name__,
                "error_model_parameters": list(error_model.parameters),
            }
        )

    def record_intensity_combination_Imid(self, Imid):
        """Record the intensity combination Imid value."""
        self._configdict["Imid"] = Imid

    def get_shared_components(self):
        return None

    def __str__(self):
        """:obj:`str`: Return a string representation of a scaling model."""
        msg = ["Scaling model:"]
        msg.append("  type : " + str(self.id_))
        for name, component in self.components.items():
            msg.append("  " + str(name).capitalize() + " component:")
            if component.parameter_esds:
                msg.append("    parameters (sigma)")
                for p, e in zip(component.parameters, component.parameter_esds):
                    if p < 0.0:
                        msg.append(f"    {p:.4f}   ({e:.4f})")
                    else:
                        msg.append(f"     {p:.4f}   ({e:.4f})")
            else:
                msg.append("    parameters")
                for p in component.parameters:
                    if p < 0.0:
                        msg.append(f"    {p:.4f}")
                    else:
                        msg.append(f"     {p:.4f}")
        msg.append("")
        return "\n".join(msg)


def _add_absorption_component_to_physically_derived_model(model, reflection_table):
    lmax = model.configdict["lmax"]
    if reflection_table.size() > 100000:
        assert "s0c" in reflection_table
        assert "s1c" in reflection_table
        theta_phi_0 = calc_theta_phi(
            reflection_table["s0c"]
        )  # array of tuples in radians
        theta_phi_1 = calc_theta_phi(reflection_table["s1c"])
        s0_lookup_index = calc_lookup_index(theta_phi_0, points_per_degree=2)
        s1_lookup_index = calc_lookup_index(theta_phi_1, points_per_degree=2)
        if SHScaleComponent.coefficients_list is None:
            SHScaleComponent.coefficients_list = create_sph_harm_lookup_table(
                lmax, points_per_degree=2
            )  # set the class variable and share
        elif len(SHScaleComponent.coefficients_list) < (lmax * (2.0 + lmax)):
            # this (rare) case can happen if adding a new dataset with a larger lmax!
            SHScaleComponent.coefficients_list = create_sph_harm_lookup_table(
                lmax, points_per_degree=2
            )  # set the class variable and share
        model.components["absorption"].data = {
            "s0_lookup": s0_lookup_index,
            "s1_lookup": s1_lookup_index,
        }
    # here just pass in good reflections
    else:
        model.components["absorption"].data = {
            "sph_harm_table": sph_harm_table(reflection_table, lmax)
        }
    surface_weight = model.configdict["abs_surface_weight"]
    parameter_restraints = flex.double([])
    for i in range(1, lmax + 1):
        parameter_restraints.extend(flex.double([1.0] * ((2 * i) + 1)))
    parameter_restraints *= surface_weight
    model.components["absorption"].parameter_restraints = parameter_restraints


class DoseDecay(ScalingModelBase):
    """A model similar to the physical model, where an exponential decay
    component is used plus a relative B-factor per sweep, with no absorption
    surface by default. Most suitable for multi-crystal datasets."""

    id_ = "dose_decay"

    phil_scope = phil.parse(dose_decay_model_phil_str)

    def __init__(self, parameters_dict, configdict, is_scaled=False):
        """Create the physical scaling model components."""
        super().__init__(configdict, is_scaled)
        if "scale" in configdict["corrections"]:
            scale_setup = parameters_dict["scale"]
            self._components["scale"] = SmoothScaleComponent1D(
                scale_setup["parameters"], scale_setup["parameter_esds"]
            )
        if "decay" in configdict["corrections"]:
            decay_setup = parameters_dict["decay"]
            if configdict["resolution_dependence"] == "linear":
                self._components["decay"] = LinearDoseDecay(
                    decay_setup["parameters"], decay_setup["parameter_esds"]
                )
            else:
                self._components["decay"] = QuadraticDoseDecay(
                    decay_setup["parameters"], decay_setup["parameter_esds"]
                )
        if "relative_B" in configdict["corrections"]:
            B_setup = parameters_dict["relative_B"]
            self._components["relative_B"] = SingleBScaleFactor(
                B_setup["parameters"], B_setup["parameter_esds"]
            )
        if "absorption" in configdict["corrections"]:
            absorption_setup = parameters_dict["absorption"]
            self._components["absorption"] = SHScaleComponent(
                absorption_setup["parameters"], absorption_setup["parameter_esds"]
            )
        if "analytical" in configdict["corrections"]:
            self._components["analytical"] = AnalyticalComponent(flex.double([]))

    @property
    def consecutive_refinement_order(self):
        """:obj:`list`: a nested list of component names to indicate scaling order."""
        return [["scale", "relative_B", "decay"], ["absorption"]]

    def fix_initial_parameter(self, params):
        if "scale" in self.components and params.dose_decay.fix_initial:
            self.components["scale"].fix_initial_parameter()
        return True

    def get_shared_components(self):
        if "shared" in self.configdict:
            if (
                "decay" in self.configdict["shared"]
                and "decay" not in self.fixed_components
            ):
                return "decay"
        return None

    def update(self, params):
        if params.dose_decay.correction.fix:
            self.fixed_components = params.dose_decay.correction.fix

    def configure_components(self, reflection_table, experiment, params):
        """Add the required reflection table data to the model components."""
        phi = reflection_table["xyzobs.px.value"].parts()[2]
        if "scale" in self.components:
            norm = phi * self._configdict["s_norm_fac"]
            self.components["scale"].data = {"x": norm}
        if "decay" in self.components:
            self.components["decay"].data = {"x": phi, "d": reflection_table["d"]}
        if "relative_B" in self.components:
            self.components["relative_B"].data = {"d": reflection_table["d"]}
        if "absorption" in self.components:
            _add_absorption_component_to_physically_derived_model(
                self, reflection_table
            )
        if "analytical" in self.components:
            if "analytical_correction" not in reflection_table:
                raise ValueError(
                    "The reflection table must contain the column 'analytical_correction'"
                    + "\nwhen using the option analytical_correction=True."
                )
            self.components["analytical"].data = {
                "correction": reflection_table["analytical_correction"]
            }

    def limit_image_range(self, new_image_range):
        """Modify the model to be suitable for a reduced image range.

        For this model, this involves determining whether the number of parameters
        should be reduced and may reduce the number of parameters in the scale and
        decay components.

        Args:
            new_image_range (tuple): The (start, end) of the new image range.
        """
        conf = self.configdict
        current_image_range = conf["valid_image_range"]
        current_osc_range = conf["valid_osc_range"]
        # calculate one osc as don't have access to scan object here
        one_osc = (conf["valid_osc_range"][1] - conf["valid_osc_range"][0]) / (
            conf["valid_image_range"][1] - (conf["valid_image_range"][0] - 1)
        )
        new_osc_range = (
            (new_image_range[0] - current_image_range[0]) * one_osc,
            (new_image_range[1] - current_image_range[0] + 1) * one_osc,
        )
        if "scale" in self.components:
            n_param, s_norm_fac, scale_rot_int = initialise_smooth_input(
                new_osc_range, one_osc, conf["scale_rot_interval"]
            )
            n_old_params = len(self.components["scale"].parameters)
            conf["scale_rot_interval"] = scale_rot_int
            conf["s_norm_fac"] = s_norm_fac
            offset = calculate_new_offset(
                current_image_range[0],
                new_image_range[0],
                s_norm_fac,
                n_old_params,
                n_param,
            )
            new_params = self.components["scale"].parameters[offset : offset + n_param]
            self.components["scale"].set_new_parameters(new_params)

        new_osc_range_0 = current_osc_range[0] + (
            (new_image_range[0] - current_image_range[0]) * one_osc
        )
        new_osc_range_1 = current_osc_range[1] + (
            (new_image_range[1] - current_image_range[1]) * one_osc
        )
        self._configdict["valid_osc_range"] = (new_osc_range_0, new_osc_range_1)
        self.set_valid_image_range(new_image_range)

    @classmethod
    def from_data(cls, params, experiment, reflection_table):
        """Create the scaling model defined by the params."""

        params = params.dose_decay
        configdict = {"corrections": []}
        parameters_dict = {}

        osc_range = experiment.scan.get_oscillation_range()
        one_osc_width = experiment.scan.get_oscillation()[1]
        configdict.update({"valid_osc_range": osc_range})
        if params.share.decay:
            configdict.update({"shared": ["decay"]})

        configdict["corrections"].append("scale")
        n_scale_param, s_norm_fac, scale_rot_int = initialise_smooth_input(
            osc_range, one_osc_width, params.scale_interval
        )
        configdict.update(
            {"s_norm_fac": s_norm_fac, "scale_rot_interval": scale_rot_int}
        )
        parameters_dict["scale"] = {
            "parameters": flex.double(n_scale_param, 1.0),
            "parameter_esds": None,
        }

        if params.relative_B_correction:
            configdict["corrections"].append("relative_B")
            parameters_dict["relative_B"] = {
                "parameters": flex.double(1, 0.0),
                "parameter_esds": None,
            }

        if params.decay_correction:
            configdict["corrections"].append("decay")
            configdict.update({"resolution_dependence": params.resolution_dependence})
            parameters_dict["decay"] = {
                "parameters": flex.double(1, 0.0),
                "parameter_esds": None,
            }

        if params.absorption_correction:
            configdict["corrections"].append("absorption")
            n_abs_param = params.lmax * (
                2 + params.lmax
            )  # arithmetic sum formula (a1=3, d=2)
            configdict.update({"lmax": params.lmax})
            configdict.update({"abs_surface_weight": params.surface_weight})
            parameters_dict["absorption"] = {
                "parameters": flex.double(n_abs_param, 0.0),
                "parameter_esds": None,
            }
        if params.analytical_correction:
            if "analytical_correction" not in reflection_table:
                raise ValueError(
                    "The reflection table must contain the column 'analytical_correction'"
                    + "\nwhen using the option analytical_correction=True."
                )
            configdict["corrections"].append("analytical")

        model = cls(parameters_dict, configdict)
        if params.correction.fix:
            model.fixed_components = params.correction.fix
        return model

    @classmethod
    def from_dict(cls, obj):
        """Create a :obj:`DoseDecayScalingModel` from a dictionary."""
        if obj["__id__"] != cls.id_:
            raise RuntimeError(f"expected __id__ {cls.id_}, got {obj['__id__']}")
        (s_params, d_params, abs_params, B) = (None, None, None, None)
        (s_params_sds, d_params_sds, a_params_sds, B_sd) = (None, None, None, None)
        configdict = obj["configuration_parameters"]
        if "scale" in configdict["corrections"]:
            s_params = flex.double(obj["scale"]["parameters"])
            if "est_standard_devs" in obj["scale"]:
                s_params_sds = flex.double(obj["scale"]["est_standard_devs"])
        if "relative_B" in configdict["corrections"]:
            B = flex.double(obj["relative_B"]["parameters"])
            if "est_standard_devs" in obj["relative_B"]:
                B_sd = flex.double(obj["relative_B"]["est_standard_devs"])
        if "decay" in configdict["corrections"]:
            d_params = flex.double(obj["decay"]["parameters"])
            if "est_standard_devs" in obj["decay"]:
                d_params_sds = flex.double(obj["decay"]["est_standard_devs"])
        if "absorption" in configdict["corrections"]:
            abs_params = flex.double(obj["absorption"]["parameters"])
            if "est_standard_devs" in obj["absorption"]:
                a_params_sds = flex.double(obj["absorption"]["est_standard_devs"])

        parameters_dict = {
            "scale": {"parameters": s_params, "parameter_esds": s_params_sds},
            "decay": {"parameters": d_params, "parameter_esds": d_params_sds},
            "relative_B": {"parameters": B, "parameter_esds": B_sd},
            "absorption": {"parameters": abs_params, "parameter_esds": a_params_sds},
        }

        return cls(parameters_dict, configdict, is_scaled=True)

    def plot_model_components(self, reflection_table=None):
        d = plot_dose_decay(self)
        if "absorption" in self.components:
            d.update(plot_absorption_parameters(self))
            d.update(plot_absorption_plots(self, reflection_table))
        return d


def determine_auto_absorption_params(absorption):
    if absorption == "high":
        lmax = 6
        surface_weight = 5e3
    elif absorption == "medium":
        lmax = 6
        surface_weight = 5e4
    else:  # low
        lmax = 4
        surface_weight = 5e5
    return lmax, surface_weight


class PhysicalScalingModel(ScalingModelBase):
    """A scaling model for a physical parameterisation."""

    id_ = "physical"

    phil_scope = phil.parse(physical_model_phil_str)

    def __init__(self, parameters_dict, configdict, is_scaled=False):
        """Create the physical scaling model components."""
        super().__init__(configdict, is_scaled)
        if "scale" in configdict["corrections"]:
            scale_setup = parameters_dict["scale"]
            self._components["scale"] = SmoothScaleComponent1D(
                scale_setup["parameters"], scale_setup["parameter_esds"]
            )
        if "decay" in configdict["corrections"]:
            decay_setup = parameters_dict["decay"]
            self._components["decay"] = SmoothBScaleComponent1D(
                decay_setup["parameters"], decay_setup["parameter_esds"]
            )
        if "absorption" in configdict["corrections"]:
            absorption_setup = parameters_dict["absorption"]
            self._components["absorption"] = SHScaleComponent(
                absorption_setup["parameters"], absorption_setup["parameter_esds"]
            )
        if "analytical" in configdict["corrections"]:
            self._components["analytical"] = AnalyticalComponent(flex.double([]))

    @property
    def consecutive_refinement_order(self):
        """:obj:`list`: a nested list of component names to indicate scaling order."""
        return [["scale", "decay"], ["absorption"]]

    def fix_initial_parameter(self, params):
        if "scale" in self.components and params.physical.fix_initial:
            self.components["scale"].fix_initial_parameter()
        return True

    def configure_components(self, reflection_table, experiment, params):
        """Add the required reflection table data to the model components."""
        phi = reflection_table["xyzobs.px.value"].parts()[2]
        if "scale" in self.components:
            norm = phi * self._configdict["s_norm_fac"]
            self.components["scale"].data = {"x": norm}
        if "decay" in self.components:
            norm = phi * self._configdict["d_norm_fac"]
            self.components["decay"].parameter_restraints = flex.double(
                self.components["decay"].parameters.size(),
                params.physical.decay_restraint,
            )
            self.components["decay"].data = {"x": norm, "d": reflection_table["d"]}
        if "absorption" in self.components:
            _add_absorption_component_to_physically_derived_model(
                self, reflection_table
            )
        if "analytical" in self.components:
            if "analytical_correction" not in reflection_table:
                raise ValueError(
                    "The reflection table must contain the column 'analytical_correction'"
                    + "\nwhen using the option analytical_correction=True."
                )
            self.components["analytical"].data = {
                "correction": reflection_table["analytical_correction"]
            }

    def limit_image_range(self, new_image_range):
        """Modify the model to be suitable for a reduced image range.

        For this model, this involves determining whether the number of parameters
        should be reduced and may reduce the number of parameters in the scale and
        decay components.

        Args:
            new_image_range (tuple): The (start, end) of the new image range.
        """
        conf = self.configdict
        current_image_range = conf["valid_image_range"]
        current_osc_range = conf["valid_osc_range"]
        # calculate one osc as don't have access to scan object here
        one_osc = (conf["valid_osc_range"][1] - conf["valid_osc_range"][0]) / (
            conf["valid_image_range"][1] - (conf["valid_image_range"][0] - 1)
        )
        new_osc_range = (
            (new_image_range[0] - current_image_range[0]) * one_osc,
            (new_image_range[1] - current_image_range[0] + 1) * one_osc,
        )
        if "scale" in self.components:
            n_param, s_norm_fac, scale_rot_int = initialise_smooth_input(
                new_osc_range, one_osc, conf["scale_rot_interval"]
            )
            n_old_params = len(self.components["scale"].parameters)
            conf["scale_rot_interval"] = scale_rot_int
            conf["s_norm_fac"] = s_norm_fac
            offset = calculate_new_offset(
                current_image_range[0],
                new_image_range[0],
                s_norm_fac,
                n_old_params,
                n_param,
            )
            new_params = self.components["scale"].parameters[offset : offset + n_param]
            self.components["scale"].set_new_parameters(new_params)
        if "decay" in self.components:
            n_param, d_norm_fac, decay_rot_int = initialise_smooth_input(
                new_osc_range, one_osc, conf["decay_rot_interval"]
            )
            n_old_params = len(self.components["decay"].parameters)
            conf["decay_rot_interval"] = decay_rot_int
            conf["d_norm_fac"] = d_norm_fac
            offset = calculate_new_offset(
                current_image_range[0],
                new_image_range[0],
                d_norm_fac,
                n_old_params,
                n_param,
            )
            new_params = self.components["decay"].parameters[offset : offset + n_param]
            self.components["decay"].set_new_parameters(new_params)

        new_osc_range_0 = current_osc_range[0] + (
            (new_image_range[0] - current_image_range[0]) * one_osc
        )
        new_osc_range_1 = current_osc_range[1] + (
            (new_image_range[1] - current_image_range[1]) * one_osc
        )
        self._configdict["valid_osc_range"] = (new_osc_range_0, new_osc_range_1)
        self.set_valid_image_range(new_image_range)

    @classmethod
    def from_data(cls, params, experiment, reflection_table):
        """Create the scaling model defined by the params."""

        params = params.physical
        configdict = {"corrections": []}
        parameters_dict = {}

        osc_range = experiment.scan.get_oscillation_range()
        one_osc_width = experiment.scan.get_oscillation()[1]
        configdict.update({"valid_osc_range": osc_range})

        abs_osc_range = abs(osc_range[1] - osc_range[0])

        if params.scale_interval in autos or params.decay_interval in autos:
            if abs_osc_range < 5.0:
                scale_interval, decay_interval = (1.0, 1.5)
            elif abs_osc_range < 10.0:
                scale_interval, decay_interval = (2.0, 3.0)
            elif abs_osc_range < 25.0:
                scale_interval, decay_interval = (4.0, 5.0)
            elif abs_osc_range < 90.0:
                scale_interval, decay_interval = (8.0, 10.0)
            else:
                scale_interval, decay_interval = (15.0, 20.0)

        if params.scale_interval not in autos:
            scale_interval = params.scale_interval
        if params.decay_interval not in autos:
            decay_interval = params.decay_interval

        configdict["corrections"].append("scale")
        n_scale_param, s_norm_fac, scale_rot_int = initialise_smooth_input(
            osc_range, one_osc_width, scale_interval
        )
        configdict.update(
            {"s_norm_fac": s_norm_fac, "scale_rot_interval": scale_rot_int}
        )
        parameters_dict["scale"] = {
            "parameters": flex.double(n_scale_param, 1.0),
            "parameter_esds": None,
        }

        if params.decay_correction:
            configdict["corrections"].append("decay")
            n_decay_param, d_norm_fac, decay_rot_int = initialise_smooth_input(
                osc_range, one_osc_width, decay_interval
            )
            configdict.update(
                {"d_norm_fac": d_norm_fac, "decay_rot_interval": decay_rot_int}
            )
            parameters_dict["decay"] = {
                "parameters": flex.double(n_decay_param, 0.0),
                "parameter_esds": None,
            }
        if params.absorption_correction in autos:
            if abs_osc_range > 60.0:
                absorption_correction = True
            else:
                absorption_correction = False
            if max(reflection_table["s1"].angle(experiment.beam.get_s0())) < math.acos(
                0.999
            ):
                # https://github.com/dials/dials/issues/2316
                absorption_correction = False
        else:
            absorption_correction = params.absorption_correction
        if absorption_correction or params.absorption_level:
            configdict["corrections"].append("absorption")
            if params.share.absorption:
                configdict.update({"shared": ["absorption"]})
            if params.absorption_level:
                lmax, surface_weight = determine_auto_absorption_params(
                    params.absorption_level
                )
                if (params.lmax not in autos) or (params.surface_weight not in autos):
                    logger.info(
                        """Using lmax, surface_weight parameters set by the absorption_level option,
                        rather than user specified options"""
                    )
            else:
                lmax, surface_weight = (params.lmax, params.surface_weight)
                if lmax in autos:
                    lmax = 4
                if params.surface_weight in autos:
                    surface_weight = 5e5
            n_abs_param = (2 * lmax) + (lmax**2)  # arithmetic sum formula (a1=3, d=2)
            configdict.update({"lmax": lmax})
            configdict.update({"abs_surface_weight": surface_weight})
            parameters_dict["absorption"] = {
                "parameters": flex.double(n_abs_param, 0.0),
                "parameter_esds": None,
            }
        if params.analytical_correction:
            if "analytical_correction" not in reflection_table:
                raise ValueError(
                    "The reflection table must contain the column 'analytical_correction'"
                    + "\nwhen using the option analytical_correction=True."
                )
            configdict["corrections"].append("analytical")

        model = cls(parameters_dict, configdict)
        if params.correction.fix:
            model.fixed_components = params.correction.fix
        return model

    @classmethod
    def from_dict(cls, obj):
        """Create a :obj:`PhysicalScalingModel` from a dictionary."""
        if obj["__id__"] != cls.id_:
            raise RuntimeError(f"expected __id__ {cls.id_}, got {obj['__id__']}")
        (s_params, d_params, abs_params) = (None, None, None)
        (s_params_sds, d_params_sds, a_params_sds) = (None, None, None)
        configdict = obj["configuration_parameters"]
        if "scale" in configdict["corrections"]:
            s_params = flex.double(obj["scale"]["parameters"])
            if "est_standard_devs" in obj["scale"]:
                s_params_sds = flex.double(obj["scale"]["est_standard_devs"])
        if "decay" in configdict["corrections"]:
            d_params = flex.double(obj["decay"]["parameters"])
            if "est_standard_devs" in obj["decay"]:
                d_params_sds = flex.double(obj["decay"]["est_standard_devs"])
        if "absorption" in configdict["corrections"]:
            abs_params = flex.double(obj["absorption"]["parameters"])
            if "est_standard_devs" in obj["absorption"]:
                a_params_sds = flex.double(obj["absorption"]["est_standard_devs"])

        parameters_dict = {
            "scale": {"parameters": s_params, "parameter_esds": s_params_sds},
            "decay": {"parameters": d_params, "parameter_esds": d_params_sds},
            "absorption": {"parameters": abs_params, "parameter_esds": a_params_sds},
        }
        return cls(parameters_dict, configdict, is_scaled=True)

    def update(self, params):
        """Update the model if new options chosen in the phil scope."""
        if params.physical.correction.fix:
            self.fixed_components = params.physical.correction.fix
        if "absorption" in self.components:
            new_lmax = None
            if params.physical.absorption_level:
                lmax, surface_weight = determine_auto_absorption_params(
                    params.physical.absorption_level
                )
                self._configdict.update({"abs_surface_weight": surface_weight})
                if lmax != self._configdict["lmax"]:
                    new_lmax = lmax
            else:  # check manually specified parameters.
                if params.physical.surface_weight not in autos:
                    self._configdict.update(
                        {"abs_surface_weight": params.physical.surface_weight}
                    )
                if (params.physical.lmax not in autos) and (
                    params.physical.lmax != self._configdict["lmax"]
                ):
                    new_lmax = params.physical.lmax
            if new_lmax:
                # need to change the parameters for this component
                current_parameters = self.components["absorption"].parameters
                n_abs_param = new_lmax * (2 + new_lmax)
                self._configdict.update({"lmax": new_lmax})
                new_parameters = flex.double(n_abs_param, 0.0)
                # copy across matching parameters:
                for i in range(0, min(n_abs_param, current_parameters.size())):
                    new_parameters[i] = current_parameters[i]
                self._components["absorption"] = SHScaleComponent(
                    new_parameters, flex.double(n_abs_param, 0.0)
                )
            if params.physical.share.absorption:
                self._configdict.update({"shared": ["absorption"]})

    def plot_model_components(self, reflection_table=None):
        d = plot_smooth_scales(self)
        if "absorption" in self.components:
            d.update(plot_absorption_parameters(self))
            d.update(plot_absorption_plots(self, reflection_table))
        return d

    def get_shared_components(self):
        if "shared" in self.configdict:
            if (
                "absorption" in self.configdict["shared"]
                and "absorption" not in self.fixed_components
            ):
                return "absorption"
        return None


class ArrayScalingModel(ScalingModelBase):
    """A scaling model for an array-based parameterisation."""

    id_ = "array"

    phil_scope = phil.parse(array_model_phil_str)

    def __init__(self, parameters_dict, configdict, is_scaled=False):
        """Create the array scaling model components."""
        super().__init__(configdict, is_scaled)
        if not any(i in configdict["corrections"] for i in ["decay", "absorption"]):
            raise ValueError(
                "Array model must have at least one of decay or absorption corrections"
            )
        if "decay" in configdict["corrections"]:
            decay_setup = parameters_dict["decay"]
            self._components["decay"] = SmoothScaleComponent2D(
                decay_setup["parameters"],
                shape=(configdict["n_res_param"], configdict["n_time_param"]),
                parameter_esds=decay_setup["parameter_esds"],
            )
        if "absorption" in configdict["corrections"]:
            abs_setup = parameters_dict["absorption"]
            self._components["absorption"] = SmoothScaleComponent3D(
                abs_setup["parameters"],
                shape=(
                    configdict["n_x_param"],
                    configdict["n_y_param"],
                    configdict["n_time_param"],
                ),
                parameter_esds=abs_setup["parameter_esds"],
            )
        if "modulation" in configdict["corrections"]:
            mod_setup = parameters_dict["modulation"]
            self._components["modulation"] = SmoothScaleComponent2D(
                mod_setup["parameters"],
                shape=(configdict["n_x_mod_param"], configdict["n_y_mod_param"]),
                parameter_esds=mod_setup["parameter_esds"],
            )
        if "analytical" in configdict["corrections"]:
            self._components["analytical"] = AnalyticalComponent(flex.double([]))

    @property
    def consecutive_refinement_order(self):
        """:obj:`list`: a nested list of component names to indicate scaling order."""
        return [["decay"], ["absorption"], ["modulation"]]

    def update(self, params):
        if params.array.correction.fix:
            self.fixed_components = params.array.correction.fix

    def configure_components(self, reflection_table, experiment, params):
        """Add the required reflection table data to the model components."""
        xyz = reflection_table["xyzobs.px.value"].parts()
        norm_time = xyz[2] * self.configdict["time_norm_fac"]
        if "decay" in self.components:
            d = reflection_table["d"]
            norm_res = (
                (1.0 / flex.pow2(d)) - self.configdict["resmin"]
            ) / self.configdict["res_bin_width"]
            self.components["decay"].data = {"x": norm_res, "y": norm_time}
        if "absorption" in self.components:
            norm_x_abs = (xyz[0] - self.configdict["xmin"]) / self.configdict[
                "x_bin_width"
            ]
            norm_y_abs = (xyz[1] - self.configdict["ymin"]) / self.configdict[
                "y_bin_width"
            ]
            self.components["absorption"].data = {
                "x": norm_x_abs,
                "y": norm_y_abs,
                "z": norm_time,
            }
        if "modulation" in self.components:
            norm_x_det = (xyz[0] - self.configdict["xmin"]) / self.configdict[
                "x_det_bin_width"
            ]
            norm_y_det = (xyz[1] - self.configdict["ymin"]) / self.configdict[
                "y_det_bin_width"
            ]
            self.components["modulation"].data = {"x": norm_x_det, "y": norm_y_det}
        if "analytical" in self.components:
            if "analytical_correction" not in reflection_table:
                raise ValueError(
                    "The reflection table must contain the column 'analytical_correction'"
                    + "\nwhen using the option analytical_correction=True."
                )
            self.components["analytical"].data = {
                "correction": reflection_table["analytical_correction"]
            }

    def limit_image_range(self, new_image_range):
        """Modify the model to be suitable for a reduced image range.

        For this model, this involves determining whether the number of parameters
        should be reduced and may reduce the number of parameters in the absorption
        and decay components.

        Args:
            new_image_range (tuple): The (start, end) of the new image range.
        """
        conf = self.configdict
        current_image_range = conf["valid_image_range"]
        current_osc_range = conf["valid_osc_range"]
        # calculate one osc as don't have access to scan object here
        one_osc = (conf["valid_osc_range"][1] - conf["valid_osc_range"][0]) / (
            conf["valid_image_range"][1] - (conf["valid_image_range"][0] - 1)
        )
        new_osc_range = (
            (new_image_range[0] - current_image_range[0]) * one_osc,
            (new_image_range[1] - current_image_range[0] + 1) * one_osc,
        )

        n_param, time_norm_fac, time_rot_int = initialise_smooth_input(
            new_osc_range, one_osc, conf["time_rot_interval"]
        )
        if "decay" in self.components:
            n_old_time_params = int(
                len(self.components["decay"].parameters)
                / self.components["decay"].n_x_params
            )
        else:
            n_old_time_params = int(
                len(self.components["absorption"].parameters)
                / (
                    self.components["absorption"].n_x_params
                    * self.components["absorption"].n_y_params
                )
            )
        offset = calculate_new_offset(
            current_image_range[0],
            new_image_range[0],
            time_norm_fac,
            n_old_time_params,
            n_param,
        )

        if "absorption" in self.components:
            params = self.components["absorption"].parameters
            n_x_params = self.components["absorption"].n_x_params
            n_y_params = self.components["absorption"].n_y_params
            # can't do simple slice as 3-dim array
            time_offset = offset * n_x_params * n_y_params
            new_params = params[
                time_offset : time_offset + (n_param * n_x_params * n_y_params)
            ]
            self.components["absorption"].set_new_parameters(
                new_params, shape=(n_x_params, n_y_params, n_param)
            )
        if "decay" in self.components:
            params = self.components["decay"].parameters
            n_decay_params = self.components["decay"].n_x_params
            # can't do simple slice as 2-dim array
            decay_offset = offset * n_decay_params
            new_params = params[
                decay_offset : decay_offset + (n_param * n_decay_params)
            ]
            self.components["decay"].set_new_parameters(
                new_params, shape=(n_decay_params, n_param)
            )
        self._configdict["n_time_param"] = n_param
        self._configdict["time_norm_fac"] = time_norm_fac
        self._configdict["time_rot_interval"] = time_rot_int
        new_osc_range_0 = current_osc_range[0] + (
            (new_image_range[0] - current_image_range[0]) * one_osc
        )
        new_osc_range_1 = current_osc_range[1] + (
            (new_image_range[1] - current_image_range[1]) * one_osc
        )
        self._configdict["valid_osc_range"] = (new_osc_range_0, new_osc_range_1)
        self.set_valid_image_range(new_image_range)

    @classmethod
    def from_data(cls, params, experiment, reflection_table):
        """create an array-based scaling model."""
        params = params.array
        reflections = reflection_table.select(reflection_table["d"] > 0.0)
        configdict = {"corrections": []}
        # First initialise things common to more than one correction.
        one_osc_width = experiment.scan.get_oscillation()[1]
        osc_range = experiment.scan.get_oscillation_range()
        configdict.update({"valid_osc_range": osc_range})
        n_time_param, time_norm_fac, time_rot_int = initialise_smooth_input(
            osc_range, one_osc_width, params.decay_interval
        )
        (xvalues, yvalues, _) = reflections["xyzobs.px.value"].parts()
        (xmax, xmin) = (flex.max(xvalues) + 0.001, flex.min(xvalues) - 0.001)
        (ymax, ymin) = (flex.max(yvalues) + 0.001, flex.min(yvalues) - 0.001)

        parameters_dict = {}

        if params.decay_correction:
            configdict["corrections"].append("decay")
            resmax = (1.0 / (flex.min(reflections["d"]) ** 2)) + 1e-6
            resmin = (1.0 / (flex.max(reflections["d"]) ** 2)) - 1e-6
            n_res_bins = params.n_resolution_bins
            n_res_param, res_bin_width = calc_n_param_from_bins(
                resmin, resmax, n_res_bins
            )
            configdict.update(
                {
                    "n_res_param": n_res_param,
                    "n_time_param": n_time_param,
                    "resmin": resmin,
                    "res_bin_width": res_bin_width,
                    "time_norm_fac": time_norm_fac,
                    "time_rot_interval": time_rot_int,
                }
            )
            parameters_dict["decay"] = {
                "parameters": flex.double((n_time_param * n_res_param), 1.0),
                "parameter_esds": None,
            }

        if params.absorption_correction:
            configdict["corrections"].append("absorption")
            nxbins = nybins = params.n_absorption_bins
            n_x_param, x_bin_width = calc_n_param_from_bins(xmin, xmax, nxbins)
            n_y_param, y_bin_width = calc_n_param_from_bins(ymin, ymax, nybins)
            configdict.update(
                {
                    "n_x_param": n_x_param,
                    "n_y_param": n_y_param,
                    "xmin": xmin,
                    "ymin": ymin,
                    "x_bin_width": x_bin_width,
                    "y_bin_width": y_bin_width,
                    "n_time_param": n_time_param,
                    "time_norm_fac": time_norm_fac,
                    "time_rot_interval": time_rot_int,
                }
            )
            parameters_dict["absorption"] = {
                "parameters": flex.double((n_x_param * n_y_param * n_time_param), 1.0),
                "parameter_esds": None,
            }

        if params.modulation_correction:
            configdict["corrections"].append("modulation")
            nx_det_bins = ny_det_bins = params.n_modulation_bins
            n_x_mod_param, x_det_bw = calc_n_param_from_bins(xmin, xmax, nx_det_bins)
            n_y_mod_param, y_det_bw = calc_n_param_from_bins(ymin, ymax, ny_det_bins)
            configdict.update(
                {
                    "n_x_mod_param": n_x_mod_param,
                    "n_y_mod_param": n_y_mod_param,
                    "xmin": xmin,
                    "ymin": ymin,
                    "x_det_bin_width": x_det_bw,
                    "y_det_bin_width": y_det_bw,
                }
            )
            parameters_dict["modulation"] = {
                "parameters": flex.double((n_x_mod_param * n_y_mod_param), 1.0),
                "parameter_esds": None,
            }
        if params.analytical_correction:
            if "analytical_correction" not in reflection_table:
                raise ValueError(
                    "The reflection table must contain the column 'analytical_correction'"
                    + "\nwhen using the option analytical_correction=True."
                )
            configdict["corrections"].append("analytical")

        model = cls(parameters_dict, configdict)
        if params.correction.fix:
            model.fixed_components = params.correction.fix
        return model

    @classmethod
    def from_dict(cls, obj):
        """Create an :obj:`ArrayScalingModel` from a dictionary."""
        if obj["__id__"] != cls.id_:
            raise RuntimeError(f"expected __id__ {cls.id_}, got {obj['__id__']}")
        configdict = obj["configuration_parameters"]
        (dec_params, abs_params, mod_params) = (None, None, None)
        (d_params_sds, a_params_sds, m_params_sds) = (None, None, None)
        if "decay" in configdict["corrections"]:
            dec_params = flex.double(obj["decay"]["parameters"])
            if "est_standard_devs" in obj["decay"]:
                d_params_sds = flex.double(obj["decay"]["est_standard_devs"])
        if "absorption" in configdict["corrections"]:
            abs_params = flex.double(obj["absorption"]["parameters"])
            if "est_standard_devs" in obj["absorption"]:
                a_params_sds = flex.double(obj["absorption"]["est_standard_devs"])
        if "modulation" in configdict["corrections"]:
            mod_params = flex.double(obj["modulation"]["parameters"])
            if "est_standard_devs" in obj["modulation"]:
                m_params_sds = flex.double(obj["modulation"]["est_standard_devs"])

        parameters_dict = {
            "decay": {"parameters": dec_params, "parameter_esds": d_params_sds},
            "absorption": {"parameters": abs_params, "parameter_esds": a_params_sds},
            "modulation": {"parameters": mod_params, "parameter_esds": m_params_sds},
        }

        return cls(parameters_dict, configdict, is_scaled=True)

    def plot_model_components(self, reflection_table=None):
        d = {}
        if "absorption" in self.components:
            d.update(plot_array_absorption_plot(self))
        if "decay" in self.components:
            d.update(plot_array_decay_plot(self))
        if "modulation" in self.components:
            d.update(plot_array_modulation_plot(self))
        return d


class KBScalingModel(ScalingModelBase):
    """A scaling model for a KB parameterisation."""

    id_ = "KB"

    phil_scope = phil.parse(kb_model_phil_str)

    def __init__(self, parameters_dict, configdict, is_scaled=False):
        """Create the KB scaling model components."""
        super().__init__(configdict, is_scaled)
        if "scale" in configdict["corrections"]:
            self._components["scale"] = SingleScaleFactor(
                parameters_dict["scale"]["parameters"],
                parameters_dict["scale"]["parameter_esds"],
            )
        if "decay" in configdict["corrections"]:
            self._components["decay"] = SingleBScaleFactor(
                parameters_dict["decay"]["parameters"],
                parameters_dict["decay"]["parameter_esds"],
            )
        if "analytical" in configdict["corrections"]:
            self._components["analytical"] = AnalyticalComponent(flex.double([]))

    def configure_components(self, reflection_table, experiment, params):
        """Add the required reflection table data to the model components."""
        if "scale" in self.components:
            self.components["scale"].data = {"id": reflection_table["id"]}
        if "decay" in self.components:
            self.components["decay"].data = {
                "d": reflection_table["d"],
            }
        if "analytical" in self.components:
            if "analytical_correction" not in reflection_table:
                raise ValueError(
                    "The reflection table must contain the column 'analytical_correction'"
                    + "\nwhen using the option analytical_correction=True."
                )
            self.components["analytical"].data = {
                "correction": reflection_table["analytical_correction"]
            }

    @property
    def consecutive_refinement_order(self):
        """:obj:`list`: a nested list of component names to indicate scaling order."""
        return [["scale", "decay"]]

    def update(self, params):
        if params.KB.correction.fix:
            self.fixed_components = params.KB.correction.fix

    @classmethod
    def from_dict(cls, obj):
        """Create an :obj:`KBScalingModel` from a dictionary."""
        if obj["__id__"] != cls.id_:
            raise RuntimeError(f"expected __id__ {cls.id_}, got {obj['__id__']}")
        configdict = obj["configuration_parameters"]
        (s_params, d_params) = (None, None)
        (s_params_sds, d_params_sds) = (None, None)
        if "scale" in configdict["corrections"]:
            s_params = flex.double(obj["scale"]["parameters"])
            if "est_standard_devs" in obj["scale"]:
                s_params_sds = flex.double(obj["scale"]["est_standard_devs"])
        if "decay" in configdict["corrections"]:
            d_params = flex.double(obj["decay"]["parameters"])
            if "est_standard_devs" in obj["decay"]:
                d_params_sds = flex.double(obj["decay"]["est_standard_devs"])

        parameters_dict = {
            "scale": {"parameters": s_params, "parameter_esds": s_params_sds},
            "decay": {"parameters": d_params, "parameter_esds": d_params_sds},
        }

        return cls(parameters_dict, configdict, is_scaled=True)

    @classmethod
    def from_data(cls, params, experiment, reflection_table):
        """Create the :obj:`KBScalingModel` from data."""
        configdict = {"corrections": []}
        parameters_dict = {}

        if params.KB.decay_correction:
            configdict["corrections"].append("decay")
            parameters_dict["decay"] = {
                "parameters": flex.double([0.0]),
                "parameter_esds": None,
            }
        configdict["corrections"].append("scale")
        parameters_dict["scale"] = {
            "parameters": flex.double([1.0]),
            "parameter_esds": None,
        }
        if params.KB.analytical_correction:
            if "analytical_correction" not in reflection_table:
                raise ValueError(
                    "The reflection table must contain the column 'analytical_correction'"
                    + "\nwhen using the option analytical_correction=True."
                )
            configdict["corrections"].append("analytical")

        model = cls(parameters_dict, configdict)
        if params.KB.correction.fix:
            model.fixed_components = params.KB.correction.fix
        return model


def calculate_new_offset(
    current_image_0, new_image_0, new_norm_fac, n_old_param, n_new_param
):
    """Calculate the parameter offset for the new image range.

    Returns:
        int: An offset to apply when selecting the new parameters from the
          existing parameters.
    """
    if n_old_param == 2:
        return 0  # can't have less than two params
    batch_difference = (new_image_0 - current_image_0) * new_norm_fac
    n_to_shift = int(batch_difference // 1)
    if batch_difference % 1 > 0.5:
        n_to_shift += 1
    return min(n_old_param - n_new_param, n_to_shift)  # can't shift by more
    # than difference between old and new


def initialise_smooth_input(osc_range, one_osc_width, interval):
    """Calculate the required smoother parameters.

    Using information about the sequence and the chosen parameterisation
    interval, the required parameters for the smoother are determined.

    Args:
        osc_range (tuple): The (start, stop) of an oscillation in degrees.
        one_osc_width (float): The oscillation width of a single image in degrees.
        interval (float): The required maximum separation between parameters
            in degrees.

    Returns:
        tuple: 3-element tuple containing;
            n_params (:obj:`int`): The number of parameters to use.
            norm_fac (:obj:`float`): The degrees to parameters space normalisation factor.
            interval (:obj:`float`): The actual interval in degrees between the parameters.
    """
    interval += 0.00001
    if (osc_range[1] - osc_range[0]) < (2.0 * interval):
        if (osc_range[1] - osc_range[0]) <= interval:
            rot_int = osc_range[1] - osc_range[0]
            n_param = 2
        else:
            rot_int = (osc_range[1] - osc_range[0]) / 2.0
            n_param = 3
    else:
        n_bins = max(int((osc_range[1] - osc_range[0]) / interval) + 1, 3)
        rot_int = (osc_range[1] - osc_range[0]) / float(n_bins)
        n_param = n_bins + 2
    norm_fac = 0.999 * one_osc_width / rot_int  # to make sure normalise values
    # fall within range of smoother.
    return n_param, norm_fac, rot_int


def calc_n_param_from_bins(value_min, value_max, n_bins):
    """Return the correct number of bins for initialising the gaussian
    smoothers."""
    assert n_bins > 0
    assert isinstance(n_bins, int)
    bin_width = (value_max - value_min) / n_bins
    if n_bins == 1:
        n_param = 2
    elif n_bins == 2:
        n_param = 3
    else:
        n_param = n_bins + 2
    return n_param, bin_width


model_phil_scope = phil.parse("")
_dxtbx_scaling_models = {
    ep.name: ep
    for ep in importlib.metadata.entry_points(group="dxtbx.scaling_model_ext")
}
assert (
    _dxtbx_scaling_models
), "No models registered with dxtbx.scaling_model_ext entry point"
model_phil_scope.adopt_scope(
    phil.parse(
        "model ="
        + " ".join(_dxtbx_scaling_models)
        + "\n    .type = choice"
        + "\n    .help = Set scaling model to be applied to input datasets"
        + "\n    .expert_level = 0"
    )
)
for entry_point_name, entry_point in _dxtbx_scaling_models.items():
    ext_master_scope = phil.parse("%s .expert_level=1 {}" % entry_point_name)
    ext_phil_scope = ext_master_scope.get_without_substitution(entry_point_name)
    assert len(ext_phil_scope) == 1
    ext_phil_scope = ext_phil_scope[0]
    ext_phil_scope.adopt_scope(entry_point.load().phil_scope)
    model_phil_scope.adopt_scope(ext_master_scope)


def plot_scaling_models(model, reflection_table=None):
    """Return a dict of component plots for the model for plotting with plotly."""
    return model.plot_model_components(reflection_table=reflection_table)


def make_combined_plots(data):
    """Make any plots that require evaluation of all models."""
    if all(d.id_ == "dose_decay" for d in data.values()):
        relative_Bs = [
            d.to_dict()["relative_B"]["parameters"][0] for d in data.values()
        ]
        return plot_relative_Bs(relative_Bs)
    return {}
