"""
This module defines an abstract CrossValidator and an implementation of a
cross validator for dials.scale
"""

from __future__ import annotations

import importlib.metadata
import itertools
from copy import deepcopy

from libtbx import phil
from libtbx.table_utils import simple_table
from scitbx.array_family import flex


class CrossValidator:
    """Abstract class defining common methods for cross validation and methods
    that must be implemented for concrete implementations"""

    # metadata needed when constructing the results table
    results_metadata = {
        "names": [],  # the names of the 'results' that should be printed in the
        # results table upon completion
        "indices_to_monitor": [],  # these indices of the above list will be
        # monitored to see which config gives the best values
        "best_criterion": [],  # define 'best' for the above indices ('min' or 'max')
    }

    def __init__(self, experiments, reflections):
        self.experiments = experiments
        self.reflections = reflections
        self.results_dict = {}

    def run_script(self, params, config_no):
        """Run the appropriate command line script with the params, get the
        free/work set results and add to the results dict. Indicate the
        configuration number being run."""
        raise NotImplementedError()

    def get_results_from_script(self, script):
        """Return the work/free results list from the command line script object"""
        raise NotImplementedError()

    def get_parameter_type(self, name):
        """Find the parameter type for a discrete phil option - bool or choice."""
        raise NotImplementedError()

    def set_parameter(self, params, name, val):
        """Find the name in the params scope extract and set it to the val"""
        raise NotImplementedError()

    def set_free_set_offset(self, params, n):
        """Set the free set offset in the correct place in the scope"""
        raise NotImplementedError()

    def get_free_set_percentage(self, params):
        """Inspect the free set percentage in the correct place in the scope"""
        raise NotImplementedError()

    @staticmethod
    def _avg_sd_from_list(lst):
        """simple function to get average and standard deviation"""
        arr = flex.double(lst)
        avg = round(flex.mean(arr), 5)
        std = round(arr.standard_deviation_of_the_sample(), 5)
        return avg, std

    def create_results_dict(self, n_options):
        """Create a results dict of the correct size for filling later"""
        if n_options == 1:
            self.results_dict[0] = {"configuration": ["user"]}
            for name in self.results_metadata["names"]:
                self.results_dict[0][name] = []
        else:
            self.results_dict = {}
            for i in range(n_options):
                self.results_dict[i] = {"configuration": []}
                for name in self.results_metadata["names"]:
                    self.results_dict[i][name] = []

    def set_results_dict_configuration(self, keys, values):
        """Add configuration information to the results dict"""
        assert len(keys) == len(values)
        for i, v in enumerate(itertools.product(*values)):
            e = dict(zip(keys, v))
            for k, val in e.items():
                self.results_dict[i]["configuration"].append(str(k) + "=" + str(val))

    def add_results_to_results_dict(self, config_no, results):
        """Add the results to the correct place in the dict"""
        assert len(results) == len(self.results_metadata["names"])
        for name, result in zip(self.results_metadata["names"], results):
            self.results_dict[config_no][name].append(result)

    def interpret_results(self):
        """Inspect the data in results_dict, make a nice table with the mean and
        average over many attempts and indicate the 'best' option"""
        rows = []
        headers = ["option", ""] + self.results_metadata["names"]
        monitored_values = []

        # Construct the rows, using the metadata from the results dict
        for v in self.results_dict.values():
            config_str = " ".join(v["configuration"])
            vals, stds = [], []
            for i, name in enumerate(self.results_metadata["names"]):
                val, std = self._avg_sd_from_list(v[name])
                vals.append(val)
                stds.append(std)
                if i in self.results_metadata["indices_to_monitor"]:
                    monitored_values.append(val)
            rows.append([config_str, "mean"] + [str(i) for i in vals])
            rows.append(["", "std dev"] + [str(i) for i in stds])

        # Now go through monitored values, finding the best and adding a '*'
        n_monitored = len(self.results_metadata["indices_to_monitor"])
        for i in range(n_monitored):
            vals = monitored_values[i::n_monitored]
            if self.results_metadata["best_criterion"][i] == "max":
                best_idx = vals.index(max(vals)) * 2  # *2 to skip std rows
            elif self.results_metadata["best_criterion"][i] == "min":
                best_idx = vals.index(min(vals)) * 2  # *2 to skip std rows
            rows[best_idx][self.results_metadata["indices_to_monitor"][i] + 2] += "*"
            # line above, 2 is to offset first two columns in table

        return simple_table(rows, headers)


class DialsScaleCrossValidator(CrossValidator):
    """An implementation of the CrossValidator for running dials.scale"""

    results_metadata = {  # metadata used when constructing the results table
        "names": [
            "work Rmeas",
            "free Rmeas",
            "Rmeas gap",
            "work CC1/2",
            "free CC1/2",
            "CC1/2 gap",
        ],
        "indices_to_monitor": [1, 2, 4, 5],  # these indices of the above list
        # will be monitored to see which config gives the best values
        "best_criterion": ["min", "min", "max", "min"],
    }

    def get_results_from_script(self, script):
        """Return the work/free results list from the command line script object"""
        result = script.scaler.work_free_stats
        return result

    def get_parameter_type(self, name):
        """Find the parameter type for a scaling phil option."""
        phil_scope = phil.parse(
            """include scope dials.command_line.scale.phil_scope""",
            process_includes=True,
        )
        obj = phil.find_scope(phil_scope, name)
        if not obj:
            raise ValueError(
                """Unable to resolve %s in the phil scope, make sure full phil path
is provided. For example, physical.decay_correction rather than decay_correction"""
                % name
            )
        return obj.type.phil_type  # a str: "int", "bool" etc

    def set_parameter(self, params, name, val):
        """Find the name in the params scope extract and set it to the val"""
        if name == "model":
            params.model = val
            return params
        available_models = [
            entry_point.name
            for entry_point in importlib.metadata.entry_points(
                group="dxtbx.scaling_model_ext"
            )
        ]
        phil_branches = [
            params.weighting.error_model,
            params.cut_data,
            params.scaling_options,
            params.reflection_selection,
            params.reflection_selection.random,
            params.reflection_selection.random.multi_dataset,
        ]
        if params.model:
            phil_branches.append(params.__getattribute__(str(params.model)))
        elif ("." in name) and (name.split(".")[0] in available_models):
            # if the user hasn't specified the model, but have done
            # e.g physical.parameter = *, then set model=physical
            params.model = name.split(".")[0]
            phil_branches.append(params.__getattribute__(str(params.model)))
        if "." in name:  # handle e.g physical.absorption_correction
            name = name.split(".")[-1]
        for branch in phil_branches:
            try:
                branch.__setattr__(name, val)
                return params
            except AttributeError:
                pass
        # if get here, haven't found what we're trying to set
        raise ValueError("Unable to set chosen attribute " + str(name) + "=" + str(val))

    def set_free_set_offset(self, params, n):
        """Set the free set offset in the correct place in the scope"""
        params.scaling_options.free_set_offset = n
        return params

    def get_free_set_percentage(self, params):
        """Inspect the free set percentage in the correct place in the scope"""
        return params.scaling_options.free_set_percentage

    def run_script(self, params, config_no):
        """Run the scaling script with the params, get the free/work set results
        and add to the results dict"""
        from dials.algorithms.scaling.algorithm import ScalingAlgorithm

        params.scaling_options.__setattr__("use_free_set", True)
        algorithm = ScalingAlgorithm(
            params,
            experiments=deepcopy(self.experiments),
            reflections=deepcopy(self.reflections),
        )
        algorithm.run()
        results = self.get_results_from_script(algorithm)
        self.add_results_to_results_dict(config_no, results)
