"""
This is a module to import the entire spikeinterface in a flat way.
With `import spieinterface.full as si`
Note that import is convinient but wuite heavy.

# this import the core only
import spieinterface as si

# this import module one by one
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw


# this import everything in a flat module
import spieinterface.full as si
"""

from .core import *
from .extractors import *
from .sorters import *
from .toolkit import *
from .comparison import *
from .widgets import *
from .exporters import *

from .version import version as __version__
