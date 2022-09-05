from spikeinterface.widgets.base import BackendPlotter

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


class IpywidgetsPlotter(BackendPlotter):
    backend = 'ipywidgets'
    backend_kwargs_desc = {
        "width_cm": "Width of the figure in cm (default 10)",
        "height_cm": "Height of the figure in cm (default 6)",
        "display": "If True, widgets are immediately displayed"
    }
    default_backend_kwargs = {
        "width_cm": 25,
        "height_cm": 10,
        "display": True
    }
    
    def __init__(self) -> None:
        super().__init__()
        mpl_backend = mpl.get_backend()
        assert "ipympl" in mpl_backend, ("To use the 'ipywidgets' backend, you have to set %matplotlib widget")
