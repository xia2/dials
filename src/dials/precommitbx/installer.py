from __future__ import annotations

import functools
import importlib.metadata
import json
import operator
import os
import pathlib
import stat
import sys
from distutils.version import LooseVersion as parse_version

import libtbx.load_env

try:
    import conda.cli.python_api
except ModuleNotFoundError:
    conda = None

try:
    import pre_commit.constants
except ModuleNotFoundError:
    pre_commit = None

BOLD = "\033[1m"
GREEN = "\033[32m"
MAGENTA = "\033[1;35m"
NC = "\033[0m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"

minimum_conda_version = "4.8.0"
minimum_precommit_version = "2.16"
minimum_python_version = "3.9.0"

repo_prefix = "  {:.<15}:"
repo_no_precommit = "(no pre-commit hooks)"
repo_precommit_installed = GREEN + "pre-commit installed" + NC
repo_precommit_conflict = (
    RED + "pre-commit available but a different pre-commit hook is installed" + NC
)
repo_precommit_legacy = YELLOW + "pre-commit out of date" + NC
repo_precommit_available = MAGENTA + "pre-commit available but not installed" + NC
repo_is_worktree = "(is a git worktree checkout)"


def precommitbx_template():
    prefix_path = pathlib.Path(_conda_info()["default_prefix"])
    assert prefix_path.is_dir(), f"Detected conda prefix path {prefix_path} is invalid"
    activate_script = prefix_path / "etc" / "profile.d" / "conda.sh"
    assert (
        activate_script.is_file()
    ), f"Expected conda activation script {activate_script} not found"
    file_content = (
        "#!/bin/bash",
        "# File generated by precommitbx for conda environments",
        "export LD_LIBRARY_PATH=",
        "export PYTHONPATH=",
        "export SETUPTOOLS_USE_DISTUTILS=stdlib",
        f"export PRE_COMMIT_HOME={prefix_path}/.precommit",
        "if [ ! -f .pre-commit-config.yaml ]; then",
        "  echo No pre-commit configuration. Skipping pre-commit checks.",
        "  exit 0",
        "fi",
        f"source {activate_script}",
        f"conda activate {prefix_path}",
        "pre-commit run --config .pre-commit-config.yaml --hook-stage commit",
    )
    return "\n".join(file_content) + "\n"


def install_precommitbx_hook(path):
    with path.joinpath(".git", "hooks", "pre-commit").open("w") as fh:
        fh.write(precommitbx_template())
        if os.name != "nt":
            mode = os.fstat(fh.fileno()).st_mode
            mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            os.fchmod(fh.fileno(), stat.S_IMODE(mode))


def check_precommitbx_hook(path):
    hookfile = path / ".git" / "hooks" / "pre-commit"
    if not hookfile.is_file():
        return False
    if not os.access(hookfile, os.X_OK):
        return False
    hook = hookfile.read_text()
    if hook == precommitbx_template():
        return repo_precommit_installed
    if "generated by precommitbx" in hook:
        return repo_precommit_legacy
    if "DIALS_WITHOUT_PRECOMMITS" in hook:
        return False
    if hook:
        return repo_precommit_conflict
    return False


@functools.lru_cache
def _conda_info():
    conda_info, error, return_code = conda.cli.python_api.run_command(
        conda.cli.python_api.Commands.INFO,
        "--json",
        use_exception_handler=True,
    )
    if return_code:
        if conda_info and conda_info.get("message"):
            print(conda_info["message"])
        raise RuntimeError(error)
    return json.loads(conda_info)


def _install_precommit():
    conda_info, error, return_code = conda.cli.python_api.run_command(
        conda.cli.python_api.Commands.INSTALL,
        "--json",
        "--quiet",
        "--yes",
        "pre-commit",
        use_exception_handler=True,
    )
    if return_code:
        raise RuntimeError(error)
    information = json.loads(conda_info)
    if not information.get("success"):
        print(information)
        raise RuntimeError("Could not install precommit into conda environment")
    installed_packages = information.get("actions", {}).get("LINK", [])
    precommit_version = None
    for package in sorted(installed_packages, key=operator.itemgetter("name")):
        print(f"  installed {package['name']} {package['version']}")
        if package["name"] == "pre-commit":
            precommit_version = package["version"]
    return precommit_version


def list_all_repository_candidates():
    repositories = {}
    for module in sorted(libtbx.env.module_dict):
        module_paths = [
            pathlib.Path(abs(path))
            for path in libtbx.env.module_dict[module].dist_paths
            if path and (path / ".git").exists()
        ]
        if not module_paths:
            continue
        if len(module_paths) == 1:
            repositories[module] = module_paths[0]
        else:
            for path in module_paths:
                repositories[f"{module}:{path}"] = path
    for ep in importlib.metadata.entry_points(group="libtbx.precommit"):
        path = pathlib.Path(ep.load().__path__[0])
        if path.joinpath(".git").is_dir():
            repositories[ep.name] = path
        elif path.parent.joinpath(".git").is_dir():
            repositories[ep.name] = path.parent
    return repositories


def _version_check(package, version, minimum):
    if parse_version(version) < parse_version(minimum):
        colour = YELLOW
        expected = f" (expected: {minimum} or higher)"
    else:
        colour = GREEN
        expected = ""
    print(f"{package:10s}: {colour}{version}{NC}{expected}")
    return colour == GREEN


def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        print(
            """Usage: libtbx.precommit [-h|--help] [install] [TARGET [TARGET ...]]

Installs pre-commit into any development repositories known by libtbx.
Pass "install" to actually do the installation; otherwise, will only
list what would be done.

Arguments:

    TARGET      Add extra candidates, beyond those known by cctbx.
"""
        )
        sys.exit(0)

    install_things = "install" in sys.argv

    install_precommit = False
    if _conda_info():
        _version_check("Conda", _conda_info()["conda_version"], minimum_conda_version)
    else:
        print(f"Conda:      {RED}could not read information{NC}")
    if pre_commit:
        if not _version_check(
            "Pre-commit", pre_commit.constants.VERSION, minimum_precommit_version
        ):
            install_precommit = True
    else:
        print(f"Pre-commit: {RED}not installed{NC}")
        install_precommit = True
    _version_check(
        "Python",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        minimum_python_version,
    )

    # Need to obtain list of repository candidates before potentially installing precommit
    # as this may jumble up installed packages while the python process is running.
    repositories = list_all_repository_candidates()

    if install_things and install_precommit:
        if not _conda_info():
            exit(f"{RED}Can not install precommit without conda present{NC}")
        print("\nInstalling precommit...")
        precommit_version = _install_precommit()
        if not precommit_version:
            exit(f"{RED}Could not install precommit{NC}")
        try:
            import pre_commit.constants as pcc
        except ModuleNotFoundError:
            exit(
                f"Installation of precommit {GREEN}succeeded{NC}, {RED}but could not confirm version number{NC}"
            )
        if not _version_check("Pre-commit", pcc.VERSION, minimum_precommit_version):
            exit(
                f"Installation of precommit {GREEN}succeeded{NC}, {RED}but version is still insufficient{NC}"
            )

    print()
    print("Repositories:")
    changes_required = False
    for path in sys.argv[1:]:
        if path == "install":
            continue
        path = pathlib.Path(".").joinpath(path).absolute()
        if path.name in repositories:
            base = os.fspath(path)
        else:
            base = path.name
        repositories[base] = path
    for module in sorted(repositories):
        if not repositories[module].joinpath(".pre-commit-config.yaml").is_file():
            print(repo_prefix.format(module), repo_no_precommit)
            continue
        if repositories[module].joinpath(".git").is_file():
            print(repo_prefix.format(module), repo_is_worktree)
            continue
        message = (
            check_precommitbx_hook(repositories[module]) or repo_precommit_available
        )
        if message != repo_precommit_installed and install_things:
            install_precommitbx_hook(repositories[module])
            message = (
                check_precommitbx_hook(repositories[module]) or repo_precommit_available
            )
        print(repo_prefix.format(module), message)
        if message != repo_precommit_installed:
            changes_required = True

    if changes_required:
        print()
        exit(f"To install pre-commit hooks run {BOLD}libtbx.precommit install{NC}")
