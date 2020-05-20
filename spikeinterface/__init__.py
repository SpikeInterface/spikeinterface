"""

.. automodule:: spikeinterface.extractors

.. automodule:: spikeinterface.toolkit

.. automodule:: spikeinterface.sorters

.. automodule:: spikeinterface.widgets

.. automodule:: spikeinterface.comparison

"""


from .version import version as __version__

from . import extractors
from . import toolkit
from . import sorters
from . import comparison
from . import widgets

def print_spikeinterface_version():
    txt = 'spikeinterface: {}\n'.format(__version__)
    txt += '  * spikeextractor: {}\n'.format(extractors.__version__)
    txt += '  * spiketoolkit: {}\n'.format(toolkit.__version__)
    txt += '  * spikesorters: {}\n'.format(sorters.__version__)
    txt += '  * spikecomparison: {}\n'.format(comparison.__version__)
    txt += '  * spikewidgets: {}\n'.format(widgets.__version__)
    
    print(txt)
    