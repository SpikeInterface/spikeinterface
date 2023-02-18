"""
This is a module to import the entire spikeinterface in a flat way.
With `import spikeinterface.full as si`
Note that import is convenient, but quite heavy.

# this imports the core only
import spikeinterface as si

# this imports everything in a flat module
import spieinterface.full as si
"""
import importlib.metadata

__version__ = importlib.metadata.version("spikeinterface")

from .comparison import *
from .core import *
from .curation import *
from .exporters import *
from .extractors import *
from .postprocessing import *
from .preprocessing import *
from .qualitymetrics import *
from .sorters import *
from .widgets import *
