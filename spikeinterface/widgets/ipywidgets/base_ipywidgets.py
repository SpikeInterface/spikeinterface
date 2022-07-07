from spikeinterface.widgets.base import BackendPlotter

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


class IpywidgetsPlotter(BackendPlotter):
    backend = 'ipywidgets'
    backend_kwargs = {
        "width_cm": "Width of the figure in cm (default 10)",
        "height_cm": "Height of the figure in cm (default 6)"
    }
    default_backend_kwargs = {
        "width_cm": 10,
        "height_cm": 6
    }
    
    