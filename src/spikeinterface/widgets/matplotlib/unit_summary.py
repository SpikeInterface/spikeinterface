import numpy as np

from ..base import to_attr
from ..unit_summary import UnitSummaryWidget
from .base_mpl import MplPlotter


from .unit_locations import UnitLocationsPlotter
from .amplitudes import AmplitudesPlotter
from .unit_waveforms import UnitWaveformPlotter
from .unit_waveforms_density_map import UnitWaveformDensityMapPlotter

from .autocorrelograms import AutoCorrelogramsPlotter


class UnitSummaryPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):

        dp = to_attr(data_plot)
        
        # force the figure without axes
        if 'figsize' not in backend_kwargs:
            backend_kwargs['figsize'] = (18, 7)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        backend_kwargs['num_axes'] = 0
        backend_kwargs['ax'] = None
        backend_kwargs['axes'] = None
        
        self.make_mpl_figure(**backend_kwargs)
        
        # and use custum grid spec
        fig = self.figure
        nrows = 2
        ncols = 3
        if dp.plot_data_acc is not None or dp.plot_data_amplitudes is not None:
            ncols += 1
        if dp.plot_data_amplitudes is not None:
            nrows += 1
        gs = fig.add_gridspec(nrows, ncols)

        if dp.plot_data_unit_locations is not None:
            ax1 = fig.add_subplot(gs[:2, 0])
            UnitLocationsPlotter().do_plot(dp.plot_data_unit_locations, ax=ax1)
            x, y = dp.unit_location[0], dp.unit_location[1]
            ax1.set_xlim(x - 80, x + 80)
            ax1.set_ylim(y - 250, y + 250)
            ax1.set_xticks([])
            ax1.set_xlabel(None)
            ax1.set_ylabel(None)
 
        ax2 = fig.add_subplot(gs[:2, 1])
        UnitWaveformPlotter().do_plot(dp.plot_data_waveforms, ax=ax2)
        ax2.set_title(None)
        
        ax3 = fig.add_subplot(gs[:2, 2])
        UnitWaveformDensityMapPlotter().do_plot(dp.plot_data_waveform_density, ax=ax3)
        ax3.set_ylabel(None)
        
        if dp.plot_data_acc is not None:
            ax4 = fig.add_subplot(gs[:2, 3])
            AutoCorrelogramsPlotter().do_plot(dp.plot_data_acc, ax=ax4)
            ax4.set_title(None)
            ax4.set_yticks([])
 
        if dp.plot_data_amplitudes is not None:
            ax5 = fig.add_subplot(gs[2, :3])
            ax6 = fig.add_subplot(gs[2, 3])
            axes = np.array([ax5, ax6])
            AmplitudesPlotter().do_plot(dp.plot_data_amplitudes, axes=axes)
        
        fig.suptitle(f'unit_id: {dp.unit_id}')

UnitSummaryPlotter.register(UnitSummaryWidget)
