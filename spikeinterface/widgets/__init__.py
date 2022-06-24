from .widget_list import *
from .utils import get_unit_colors, array_to_image
from .base import set_default_plotter_backend, get_default_plotter_backend

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

# for debegging I mock sortingview
HAVE_SV = True


if HAVE_MPL:
    from .matplotlib import *

if HAVE_SV:
    from .sortingview import *


# we keep this to keep compatibility having all previous widgets
# TODO : remove when the refactoring will be done
# from .legacy_mpl_widgets import *