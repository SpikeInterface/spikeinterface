from spikeinterface.widgets.base import BackendPlotter

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


class IpywidgetsPlotter(BackendPlotter):
    backend = 'ipywidgets'
    
    