import libtbx.load_env
import os
import platform
from libtbx.env_config import get_boost_library_with_python_version

Import("env_etc")

env_etc.dials_dist = os.path.join(libtbx.env.dist_path("dials"), "src", "dials")
env_etc.dials_include = os.path.dirname(env_etc.dials_dist)
if not env_etc.no_boost_python and hasattr(env_etc, "boost_adaptbx_include"):
    Import("env_no_includes_boost_python_ext")
    env = env_no_includes_boost_python_ext.Clone()
    env_etc.enable_more_warnings(env=env)
    include_paths = [
        env_etc.libtbx_include,
        env_etc.scitbx_include,
        env_etc.cctbx_include,
        env_etc.ccp4io_include,
        env_etc.rstbx_include,
        env_etc.boost_include,
        env_etc.boost_adaptbx_include,
        env_etc.python_include,
        env_etc.dxtbx_include,
        env_etc.dials_include,
    ]

    if libtbx.env.build_options.use_conda:
        boost_python = get_boost_library_with_python_version(
            "boost_python", env_etc.conda_libpath
        )
        env.Append(LIBPATH=env_etc.conda_libpath)
        include_paths.extend(env_etc.conda_cpppath)
    else:
        boost_python = "boost_python"
    env_etc.include_registry.append(env=env, paths=include_paths)
    env.Append(
        LIBS=env_etc.libm
        + ["scitbx_boost_python", boost_python, "boost_thread", "cctbx"],
    )

    # Fix the build environment so that it doesn't break on modern C++
    for path in list(env["CPPPATH"]):
        if "msvc9.0_include" in path:
            env["CPPPATH"].remove(path)

    # Fix compilation errors on windows, caused by function redefinition
    # See: https://github.com/boostorg/system/issues/32#issuecomment-462912013
    if env_etc.compiler == "win32_cl":
        env.Append(CPPDEFINES="HAVE_SNPRINTF")

    env.SConscript("src/dials/model/SConscript", exports={"env": env})
    env.SConscript("src/dials/array_family/SConscript", exports={"env": env})
    env.SConscript("src/dials/algorithms/SConscript", exports={"env": env})
    env.SConscript("src/dials/pychef/SConscript", exports={"env": env})
    env.SConscript("src/dials/viewer/SConscript", exports={"env": env})
    # env.SConscript('src/dials/nexus/SConscript', exports={ 'env' : env })
    env.SConscript("src/dials/util/SConscript", exports={"env": env})

    autocomplete_scons = os.path.join(
        libtbx.env.under_build(os.path.join("dials", "autocomplete")), "SConscript"
    )
    if not any(platform.win32_ver()) and os.path.isfile(autocomplete_scons):
        env.SConscript(autocomplete_scons, exports={"env": env})

    #
    # NOTE: This must go at the bottom. The LIBS are replaced with an empty
    # list. This is done because errors occur when building the tests if it
    # isn't done. Replacing the libs afterwards still results in those errors.
    #
    env.SConscript("tests/SConscript", exports={"env": env})
