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

from .core import *
from .extractors import *
from .sorters import *
from .preprocessing import *
from .postprocessing import *
from .qualitymetrics import *
from .curation import *
from .comparison import *
from .widgets import *
from .exporters import *
from .generation import *
from .benchmark import *
