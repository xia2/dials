from __future__ import annotations

from pathlib import Path

import setuptools

from build import build

__version_tag__ = "3.18.dev"

setup_kwargs = {
    "name": "dials",
    "version": __version_tag__,
    "long_description": Path(__file__).parent.joinpath("README.md").read_text(),
    "description": "Diffraction Integration for Advanced Light Sources",
    "author": "Diamond Light Source",
    "license": "BSD-3-Clause",
    "author_email": "dials-support@lists.sourceforge.net",
    "project_urls": {
        "homepage": "https://dials.github.io",
        "repository": "https://github.com/dials/dials",
    },
    "packages": setuptools.find_packages(where="src"),
    "package_dir": {"": "src"},
    "package_data": {
        "": ["*", "boost_python/*"],
        "dials": [
            "static/katex/*",
            "static/katex/fonts/*",
            "static/katex/contrib/*",
            "static/js/*",
            "static/css/*",
            "templates/*",
        ],
        "dials.algorithms": ["spatial_indexing/*"],
    },
    "classifiers": [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    "entry_points": {
        "libtbx.precommit": ["dials=dials"],
        "libtbx.dispatcher.script": ["pytest=pytest"],
        "dxtbx.profile_model": [
            "gaussian_rs = dials.extensions.gaussian_rs_profile_model_ext:GaussianRSProfileModelExt",
            "ellipsoid = dials.extensions.ellipsoid_profile_model_ext:EllipsoidProfileModelExt",
        ],
        "dxtbx.scaling_model_ext": [
            "physical = dials.algorithms.scaling.model.model:PhysicalScalingModel",
            "KB = dials.algorithms.scaling.model.model:KBScalingModel",
            "array = dials.algorithms.scaling.model.model:ArrayScalingModel",
            "dose_decay = dials.algorithms.scaling.model.model:DoseDecay",
        ],
        "dials.index.basis_vector_search": [
            "fft1d = dials.algorithms.indexing.basis_vector_search:FFT1D",
            "fft3d = dials.algorithms.indexing.basis_vector_search:FFT3D",
            "real_space_grid_search = dials.algorithms.indexing.basis_vector_search:RealSpaceGridSearch",
        ],
        "dials.index.lattice_search": [
            "low_res_spot_match = dials.algorithms.indexing.lattice_search:LowResSpotMatch",
            "small_cell = dials.algorithms.indexing.lattice_search:SmallCell",
        ],
        "dials.integration.background": [
            "Auto = dials.extensions.auto_background_ext:AutoBackgroundExt",
            "glm = dials.extensions.glm_background_ext:GLMBackgroundExt",
            "gmodel = dials.extensions.gmodel_background_ext:GModelBackgroundExt",
            "simple = dials.extensions.simple_background_ext:SimpleBackgroundExt",
            "null = dials.extensions.null_background_ext:NullBackgroundExt",
        ],
        "dials.integration.centroid": [
            "simple = dials.extensions.simple_centroid_ext:SimpleCentroidExt",
        ],
        "dials.spotfinder.threshold": [
            "dispersion = dials.extensions.dispersion_spotfinder_threshold_ext:DispersionSpotFinderThresholdExt",
            "dispersion_extended = dials.extensions.dispersion_extended_spotfinder_threshold_ext:DispersionExtendedSpotFinderThresholdExt",
            "radial_profile = dials.extensions.radial_profile_spotfinder_threshold_ext:RadialProfileSpotFinderThresholdExt",
        ],
    },
}

build(setup_kwargs)
setuptools.setup(**setup_kwargs)
