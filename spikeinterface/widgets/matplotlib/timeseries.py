import numpy as np

from ..base import to_attr
from ..timeseries import TimeseriesWidget
from .base_mpl import MplPlotter
from matplotlib.ticker import MaxNLocator


class TimeseriesPlotter(MplPlotter):
    
    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        self.make_mpl_figure(**backend_kwargs)
        ax = self.ax
        n = len(dp.channel_ids)
        y_locs = dp.channel_locations[:, 1]
        min_y = np.min(y_locs)
        max_y = np.max(y_locs)

        if dp.mode == 'line':
            offset = dp.vspacing * (n - 1)

            for layer_key, traces in zip(dp.layer_keys, dp.list_traces):
                for i, chan_id in enumerate(dp.channel_ids):
                    offset = dp.vspacing * (n - 1 - i)
                    color = dp.colors[layer_key][chan_id]
                    ax.plot(dp.times, offset + traces[:, i], color=color)
                ax.get_lines()[-1].set_label(layer_key)

            if dp.show_channel_ids:
                ax.set_yticks(np.arange(n) * dp.vspacing)
                channel_labels = np.array([str(chan_id) for chan_id in dp.channel_ids])[::-1]
                ax.set_yticklabels(channel_labels)
            ax.set_xlim(*dp.time_range)
            ax.set_ylim(-dp.vspacing, dp.vspacing * n)
            ax.get_xaxis().set_major_locator(MaxNLocator(prune='both'))
            ax.set_xlabel('time (s)')
            if dp.add_legend:
                ax.legend(loc='upper right')

        elif dp.mode == 'map':
            assert len(dp.list_traces) == 1, 'plot_timeseries with mode="map" do not support multi recording'
            assert len(dp.clims) == 1
            clim = list(dp.clims.values())[0]
            extent = (dp.time_range[0], dp.time_range[1], min_y, max_y)
            im = ax.imshow(dp.list_traces[0].T, interpolation='nearest',
                           origin='upper', aspect='auto', extent=extent, cmap=dp.cmap)

            im.set_clim(*clim)

            if dp.with_colorbar:
                self.figure.colorbar(im, ax=ax)

            if dp.show_channel_ids:
                ax.set_yticks(np.linspace(min_y, max_y, n) + (max_y - min_y) / n * 0.5)
                channel_labels = np.array([str(chan_id) for chan_id in dp.channel_ids])[::-1]
                ax.set_yticklabels(channel_labels)

TimeseriesPlotter.register(TimeseriesWidget)
