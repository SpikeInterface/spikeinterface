import numpy as np

from ..amplitudes import AmplitudeTimeseriesWidget
from .base_mpl import MplPlotter, to_attr


class AmplitudeTimeseriesPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        d = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        

AmplitudeTimeseriesPlotter.register(AmplitudeTimeseriesWidget)
