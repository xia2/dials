from __future__ import annotations

import importlib.metadata
import sys

import dials.util

BOLD = "\033[1m"
RED = "\033[1;31m"
GREEN = "\033[32m"
NC = "\033[0m"


def read_entry_point(entry_point):
    """
    Finds all registered plugins for a given entry point.

    :return: A dictionary of entry point plugins
    """
    return {e.name: e for e in importlib.metadata.entry_points(group=entry_point)}


def installation_is_valid():
    """
    Verifies that all required plugins are present.

    :return: True if so, False otherwise
    """
    for ep, ep_dict in known_entry_points.items():
        required_plugins = set(ep_dict.get("required", []))
        if not required_plugins:
            continue
        plugins = read_entry_point(ep)
        missing_plugins = required_plugins - set(plugins)
        if missing_plugins:
            return False
    return True


known_entry_points = {
    "dxtbx.profile_model": {
        "description": "profile models",
        "required": ["gaussian_rs"],
    },
    "dxtbx.scaling_model_ext": {"description": "scaling models"},
    "dials.index.basis_vector_search": {
        "description": "Basis vector search strategies",
        "required": ["fft1d", "fft3d", "real_space_grid_search"],
    },
    "dials.index.lattice_search": {
        "description": "Lattice search strategies",
        "required": ["low_res_spot_match"],
    },
    "dials.spotfinder.threshold": {
        "description": "Spotfinding threshold algorithms",
        "required": ["dispersion", "dispersion_extended", "radial_profile"],
    },
}


@dials.util.show_mail_handle_errors()
def run(_=None):
    for ep, ep_dict in known_entry_points.items():
        print(f"{BOLD}{ep}{NC}  {ep_dict['description']}")
        plugins = read_entry_point(ep)
        for p in sorted(plugins):
            print(
                f" {GREEN}{p} {NC}({plugins[p].module_name} via {BOLD}{plugins[p].dist.project_name}{NC} {plugins[p].dist.version})"
            )
        required_plugins = set(ep_dict.get("required", []))
        missing_plugins = required_plugins - set(plugins)
        if missing_plugins:
            print(f" {RED}  --- missing required plugins:{NC}")
            for p in missing_plugins:
                print(f" {RED}{p}{NC}")
        print()
    sys.exit(not installation_is_valid())


if __name__ == "__main__":
    run()
