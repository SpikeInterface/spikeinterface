
from .base_mpl import MplPlotter


class AmplitudeTimeseriesPlotter(MplPlotter):
    def plot(self, data):
        print('plot')

from ..amplitudes import AmplitudeTimeseriesWidget
AmplitudeTimeseriesPlotter.register(AmplitudeTimeseriesWidget)
