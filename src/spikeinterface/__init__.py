"""


"""
import importlib.metadata
__version__ = importlib.metadata.version("spikeinterface")

from .core import *

import warnings
warnings.simplefilter('always', DeprecationWarning)

"""
submodules are imported only if needed
so there is no heavy dependencies:

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw

or alternativley you can do 
import spikeinterface.full as si

"""
