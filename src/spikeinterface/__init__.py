"""


"""
import importlib.metadata

__version__ = importlib.metadata.version("spikeinterface")

from .core import *

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

# This flag must set to False for release
# This avoid to use versionning that contains ".dev0" (and this is a better choice)
# This mainly usefull when using run_sorter in container and spikeinterface install
DEV_MODE = True
# DEV_MODE = False