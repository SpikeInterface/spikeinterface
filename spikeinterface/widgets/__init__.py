from .widget_list import *
from .utils import get_unit_colors, array_to_image
from .base import set_default_plotter_backend, get_default_plotter_backend

# check if backend are available
try:
    import matplotlib
    HAVE_MPL = True
except:
    HAVE_MPL = False
    
try:
    import figurl
    HAVE_FIGURL = True
except:
    HAVE_FIGURL = False

# theses import make the Widget.resgister() at import time
if HAVE_MPL:
    from .matplotlib import *

if HAVE_FIGURL:
    from .figurl import *


# we keep this to keep compatibility so we have all previous widgets
# except the one that have been ported that are imported
# with "from .widget_list import *" in the first line
from .legacy_mpl_widgets import *
