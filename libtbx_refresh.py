from __future__ import annotations

import contextlib
import inspect
import io
import os
import subprocess
import sys
from pathlib import Path

import libtbx
import libtbx.pkg_utils

# So that we can import DIALS within this script, work out where the
# sources are and make them importable
_src_path_root = str(Path(libtbx.env.dist_path("dials")).joinpath("src"))
if _src_path_root not in sys.path:
    sys.path.insert(0, _src_path_root)

# Other packages, configured first, might have attempted to import dials, which
# would only get a namespace package, if the modules/ folder is configured as a
# repository. This means just setting the path wouldn't work. So, check to see
# if we have a namespace package imported, remove it (and any sub-packages),
# set the __path__, then import the real copy of DIALS.
#
# If this is *only* dxtbx, then we can probably get away without this by
# removing this part from dxtbx.
_dials = sys.modules.get("dials")
if _dials and _dials.__file__ is None:
    # Someone tried to import us and got a namespace package
    del sys.modules["dials"]
    # Remove any sub-modules that we might have tried and failed to import
    for _module in [x for x in sys.modules if x.startswith("dials.")]:
        del sys.modules[_module]

# Now, check to see if we configured XFEL first. If so, this is an error and we
# have a mis-configured environment.
if xfel_module := libtbx.env.module_dict.get("xfel"):
    dials_module = libtbx.env.module_dict.get("dials")
    if libtbx.env.module_list.index(xfel_module) < libtbx.env.module_list.index(
        dials_module
    ):
        sys.exit(
            """\033[31;1m
Error:
    The \033[34mxfel\033[31m module is loaded before the \033[34mdials\033[31m module in the libtbx
    module list. dials has changed to \033[0;1msrc/\033[1;31m layout, so in order for xfel
    to continue to find the headers it needs, it must be reconfigured.

    To fix this, please run:

        \033[0;1mlibtbx.configure xfel\033[0m

    \033[31;1mApologies for the inconvenience!\033[0m
"""
        )

import dials.precommitbx.nagger  # noqa: E402

try:
    from dials.util.version import dials_version

    print(dials_version())
except Exception:
    pass

dials.precommitbx.nagger.nag()

# During src/ transition, this ensures that the old entry points are cleared
libtbx.pkg_utils.define_entry_points({})


def _install_setup_readonly_fallback(package_name: str):
    """
    Partially install package in the libtbx build folder.
    This is a less complete installation - base python console_scripts
    entrypoints will not be installed, but the basic package metadata
    and other entrypoints will be enumerable through dispatcher black magic
    """
    root_path = libtbx.env.dist_path(package_name)
    import_path = os.path.join(root_path, "src")

    # Install this into a build/dxtbx subfolder
    build_path = abs(libtbx.env.build_path / package_name)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--prefix",
            build_path,
            "--no-build-isolation",
            "--no-deps",
            "-e",
            root_path,
        ],
        check=True,
    )

    # Get the actual environment being configured (NOT libtbx.env)
    env = _get_real_env_hack_hack_hack()

    # Update the libtbx environment pythonpaths to point to the source
    # location which now has an .egg-info folder; this will mean that
    # the PYTHONPATH is written into the libtbx dispatchers
    rel_path = libtbx.env.as_relocatable_path(import_path)
    if rel_path not in env.pythonpath:
        env.pythonpath.insert(0, rel_path)

    # Update the sys.path so that we can find the .egg-info in this process
    # if we do a full reconstruction of the working set
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

    # Add the src/ folder as an extra command_line_locations for dispatchers
    module = env.module_dict[package_name]
    if f"src/{package_name}" not in module.extra_command_line_locations:
        module.extra_command_line_locations.append(f"src/{package_name}")

    # Regenerate dispatchers for all modules with this new sys.path
    with contextlib.redirect_stdout(io.StringIO()):
        for module in env.module_list:
            module.process_command_line_directories()


def _get_real_env_hack_hack_hack():
    """
    Get the real, currently-being-configured libtbx.env environment.
    This is not libtbx.env, because although libtbx.env_config.environment.cold_start
    does:
        self.pickle()
        libtbx.env = self
    the first time there is an "import libtbx.load_env" this environment
    gets replaced by unpickling the freshly-written libtbx_env file onto
    libtbx.env, thereby making the environment accessed via libtbx.env
    *not* the actual one that is currently being constructed.
    So, the only way to get this environment being configured in order
    to - like - configure it, is to walk the stack trace and extract the
    self object from environment.refresh directly.
    """
    for frame in inspect.stack():
        if (
            frame.filename.endswith("env_config.py")
            and frame.function == "refresh"
            and "self" in frame.frame.f_locals
        ):
            return frame.frame.f_locals["self"]

    raise RuntimeError("Could not determine real libtbx.env_config.environment object")


def _create_dials_env_script():
    """
    write dials environment setup script and clobber cctbx setup scripts
    does nothing unless a file named 'dials'/'dials.bat' exists above
    the build/ directory
    """
    import os

    import libtbx.load_env

    if os.name == "nt":
        filename = abs(libtbx.env.build_path.dirname() / "dials.bat")
    else:
        filename = abs(libtbx.env.build_path.dirname() / "dials")

    if not os.path.exists(filename):
        return

    if os.name == "nt":
        script = """
rem enable conda environment
call %~dp0conda_base\\condabin\\activate.bat
rem prepend cctbx /build/bin directory to PATH
set PATH=%~dp0build\\bin;%PATH%
"""
    else:
        script = """
#!/bin/bash

if [ ! -z "${LIBTBX_BUILD_RELOCATION_HINT:-}" ]; then
  # possibly used for some logic in the installer
  LIBTBX_BUILD="${LIBTBX_BUILD_RELOCATION_HINT}"
  LIBTBX_BUILD_RELOCATION_HINT=
  export LIBTBX_BUILD_RELOCATION_HINT
elif [ -n "$BASH_SOURCE" ]; then
  LIBTBX_BUILD="$(dirname -- "${BASH_SOURCE[0]}")/build"
else
  LIBTBX_BUILD="%s"
fi

# make path absolute and resolve symlinks
LIBTBX_BUILD=$(cd -P -- "${LIBTBX_BUILD}" && pwd -P)

# enable conda environment
source ${LIBTBX_BUILD}/../conda_base/etc/profile.d/conda.sh
conda activate $(dirname -- "${LIBTBX_BUILD}")/conda_base

# prepend cctbx /build/bin directory to PATH
PATH="${LIBTBX_BUILD}/bin:${PATH}"
export PATH

# enable DIALS command line completion
[ -n "$BASH_VERSION" ] && {
  source $(libtbx.find_in_repositories dials/util/autocomplete.sh) && \
  source ${LIBTBX_BUILD}/dials/autocomplete/bash.sh || \
    echo dials command line completion not available
}

unset LIBTBX_BUILD
""" % abs(libtbx.env.build_path)
    with open(filename, "w") as fh:
        fh.write(script.lstrip())

    if os.name != "nt":
        mode = os.stat(filename).st_mode
        mode |= (mode & 0o444) >> 2  # copy R bits to X
        os.chmod(filename, mode)

    if os.name == "nt":
        clobber = """
echo {stars}
echo The script to set up the DIALS environment has changed
echo Please source or run {newscript} instead
echo {stars}
"""
        clobber_extensions = (".sh", ".csh", ".bat")
    else:
        clobber = """
echo '{stars}'
echo The script to set up the DIALS environment has changed
echo Please source or run '{newscript}' instead
echo '{stars}'
"""
        clobber_extensions = (".sh", ".csh", ".bat")

    for clobberfile_name in (
        "setpaths",
        "setpaths_all",
        "setpaths_debug",
    ):
        for clobber_ext in clobber_extensions:
            with open(
                abs(libtbx.env.build_path / (clobberfile_name + clobber_ext)), "w"
            ) as fh:
                fh.write(clobber.format(newscript=filename, stars="*" * 74))


def _install_dials_autocompletion():
    """generate bash.sh and SConscript file in /build/dials/autocomplete"""
    import os  # required due to cctbx weirdness

    import libtbx.load_env

    # Find the dials source directory
    dist_path = libtbx.env.dist_path("dials")

    # Set the location of the output directory
    output_directory = libtbx.env.under_build(os.path.join("dials", "autocomplete"))
    try:
        os.makedirs(output_directory)
    except OSError:
        pass

    # Build a list of autocompleteable commands
    commands_dir = os.path.join(dist_path, "src", "dials", "command_line")
    command_list = []
    for filename in sorted(os.listdir(commands_dir)):
        if not filename.startswith("_") and filename.endswith(".py"):
            # Check if this file marks itself as completable
            with open(os.path.join(commands_dir, filename), "rb") as f:
                if b"DIALS_ENABLE_COMMAND_LINE_COMPLETION" in f.read():
                    command_name = f"dials.{filename[:-3]}"
                    command_list.append(command_name)
    print("Identified autocompletable commands: " + " ".join(command_list))

    # Generate the autocompletion SConscript.
    with open(os.path.join(output_directory, "SConscript"), "w") as builder:
        builder.write(
            """import os.path
import libtbx.load_env\n
Import("env")\n\n
def dispatcher_outer(name):
    return os.path.join(libtbx.env.under_build("bin"), name)\n\n
def dispatcher_inner(name):
    return os.path.join(
        libtbx.env.dist_path("dials"), "src", "dials", "command_line", "%s.py" % name.partition(".")[2]
    )\n\n
env.Append(
    BUILDERS={{
        "AutoComplete": Builder(action="-$SOURCE --export-autocomplete-hints > $TARGET")
    }}
)
env["ENV"]["DIALS_NOBANNER"] = "1"
for cmd in [
{}
]:
    ac = env.AutoComplete(cmd, [dispatcher_outer(cmd), dispatcher_inner(cmd)])
    Requires(ac, Dir(libtbx.env.under_build("lib")))
    Depends(ac, os.path.join(libtbx.env.dist_path("dials"), "src", "dials", "util", "options.py"))
    Depends(ac, os.path.join(libtbx.env.dist_path("dials"), "util", "autocomplete.sh"))
""".format("\n".join([f'    "{cmd}",' for cmd in command_list]))
        )

    # Generate a bash script activating command line completion for each relevant command
    with open(os.path.join(output_directory, "bash.sh"), "w") as script:
        script.write("type compopt &>/dev/null && {\n")
        for cmd in command_list:
            script.write(f" complete -F _dials_autocomplete {cmd}\n")
        script.write("}\n")
        script.write("type compopt &>/dev/null || {\n")
        for cmd in command_list:
            script.write(f" complete -o nospace -F _dials_autocomplete {cmd}\n")
        script.write("}\n")


# _install_package_setup("dials")
_install_setup_readonly_fallback("dials")
_create_dials_env_script()
_install_dials_autocompletion()
