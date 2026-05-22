""" """

import importlib.metadata

__version__ = importlib.metadata.version("spikeinterface")

from .core import *


def __getattr__(name):
    if name == "generate_ground_truth_recording":
        import warnings

        warnings.warn(
            "Importing generate_ground_truth_recording from spikeinterface is deprecated. "
            "Import from spikeinterface.generation instead: "
            "`from spikeinterface.generation import generate_ground_truth_recording`. "
            "This will be removed in version 0.106.0.",
            FutureWarning,
            stacklevel=2,
        )
        from .generation.ground_truth_generator import generate_ground_truth_recording

        return generate_ground_truth_recording
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


import warnings

warnings.filterwarnings("ignore", message="distutils Version classes are deprecated")
warnings.filterwarnings("ignore", message="the imp module is deprecated")

"""
submodules are imported only if needed
so there is no heavy dependencies:

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw

or alternativley you can do
import spikeinterface.full as si

"""

# This flag must be set to False for release
# This avoids using versioning that contains ".dev0" (and this is a better choice)
# This is mainly useful when using run_sorter in a container and spikeinterface install
DEV_MODE = True
# DEV_MODE = False
