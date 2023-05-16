from ..base import to_attr
from ..unit_depths import UnitDepthsWidget
from .base_mpl import MplPlotter


class UnitDepthsPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        self.make_mpl_figure(**backend_kwargs)

        ax = self.ax
        size = dp.num_spikes / max(dp.num_spikes) * 120
        ax.scatter(dp.unit_amplitudes, dp.unit_depths, color=dp.colors, s=size)

        ax.set_aspect(3)
        ax.set_xlabel('amplitude')
        ax.set_ylabel('depth [um]')
        ax.set_xlim(0, max(dp.unit_amplitudes) * 1.2)


UnitDepthsPlotter.register(UnitDepthsWidget)
