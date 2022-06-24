import numpy as np

from .base_ipywidgets import IpywidgetsPlotter

from ..timeseries import TimeseriesWidget


class TimeseriesPlotter(IpywidgetsPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        print('ici')

TimeseriesPlotter.register(TimeseriesWidget)
