# check if backend are available
try:
    import matplotlib
    HAVE_MPL = True
except:
    HAVE_MPL = False

try:
    import sortingview
    HAVE_SV = True
except:
    HAVE_SV = False

try:
    import ipywidgets
    HAVE_IPYW = True
except:
    HAVE_IPYW = False


# theses import make the Widget.resgister() at import time
if HAVE_MPL:
    import spikeinterface.widgets.matplotlib

if HAVE_SV:
    import spikeinterface.widgets.sortingview

if HAVE_IPYW:
    import spikeinterface.widgets.ipywidgets

# when importing widget list backend are already registered
from .widget_list import *

# general functions
from .utils import get_some_colors, get_unit_colors, array_to_image
from .base import set_default_plotter_backend, get_default_plotter_backend


# we keep this to keep compatibility so we have all previous widgets
# except the one that have been ported that are imported
# with "from .widget_list import *" in the first line
from ._legacy_mpl_widgets import *



