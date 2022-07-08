import numpy as np

from ..base import to_attr
from ..amplitudes import AmplitudeTimeseriesWidget
from ..utils import get_some_colors
from .base_mpl import MplPlotter


class AmplitudeTimeseriesPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        backend_kwargs["num_axes"] = 1
        self.make_mpl_figure(**backend_kwargs)
        
        unit_colors = get_some_colors(dp.unit_ids)
        
        for unit_id in dp.unit_ids:
            self.ax.scatter(dp.spiketrains[unit_id], dp.amplitudes[unit_id],
                            color=unit_colors[unit_id], s=3, alpha=1,
                            label=unit_id)
        self.ax.legend()
        self.ax.set_xlim(0, dp.total_duration)
        self.ax.set_xlabel('Times [s]')
        self.ax.set_ylabel(f'Amplitude')
        

AmplitudeTimeseriesPlotter.register(AmplitudeTimeseriesWidget)
