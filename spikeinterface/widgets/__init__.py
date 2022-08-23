from .widget_list import *
from .utils import get_some_colors, get_unit_colors, array_to_image
from .base import set_default_plotter_backend, get_default_plotter_backend

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

# we keep this to keep compatibility so we have all previous widgets
# except the one that have been ported that are imported
# with "from .widget_list import *" in the first line
from ._legacy_mpl_widgets import *


# add backends and kwargs to doc
for wcls in widget_list:
    wcls_doc = wcls.__doc__
    
    wcls_doc += """
    Backends
    --------
    
    backends: str
    {backends}
    backend_kwargs: kwargs
    {backend_kwargs}
    """
    backend_str = f"    {list(wcls.possible_backends.keys())}"
    backend_kwargs_str = ""
    for backend, backend_plotter in wcls.possible_backends.items():
        backend_kwargs_desc = backend_plotter.backend_kwargs_desc
        if len(backend_kwargs_desc) > 0:
            backend_kwargs_str += f"\n        {backend}:"
            for bk, bk_dsc in backend_kwargs_desc.items():
                backend_kwargs_str += f"\n        - {bk}: {bk_dsc}"
    wcls.__doc__ = wcls_doc.format(backends=backend_str, backend_kwargs=backend_kwargs_str)
