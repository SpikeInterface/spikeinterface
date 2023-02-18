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
# we keep this to keep compatibility so we have all previous widgets
# except the one that have been ported that are imported
# with "from .widget_list import *" in the first line
from ._legacy_mpl_widgets import *
from .base import get_default_plotter_backend, set_default_plotter_backend

# general functions
from .utils import array_to_image, get_some_colors, get_unit_colors
from .widget_list import *
