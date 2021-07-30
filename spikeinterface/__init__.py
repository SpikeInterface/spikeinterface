"""

"""

from .version import version as __version__

from .core import *

"""
submodules are imported only if needed
so there is no heavy dependencies:

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw

or alternativley you can do 
import spikeinterface.full as si

"""
