import numpy as np

from ..timeseries import TimeseriesWidget
from .base_mpl import MplPlotter, to_attr
from matplotlib.ticker import MaxNLocator


class TimeseriesPlotter(MplPlotter):
    
    def do_plot(self, data_plot, **backend_kwargs):
        d = to_attr(data_plot)
        
        self.make_mpl_figure(**backend_kwargs)
        
        ax = self.ax

        n = len(d.channel_ids)

        if d.mode == 'line':
            offset = d.vspacing * (n - 1)

            for layer_key, traces in zip(d.layer_keys, d.list_traces):
                for i, chan_id in enumerate(d.channel_ids):
                    offset = d.vspacing * (n - 1 - i)
                    color = d.colors[layer_key][chan_id]
                    ax.plot(d.times, offset + traces[:, i], color=color)
                ax.get_lines()[-1].set_label(layer_key)

            if d.show_channel_ids:
                ax.set_yticks(np.arange(n) * d.vspacing)
                ax.set_yticklabels([str(chan_id) for chan_id in d.channel_ids[::-1]])

            ax.set_xlim(*d.time_range)
            ax.set_ylim(-d.vspacing, d.vspacing * n)
            ax.get_xaxis().set_major_locator(MaxNLocator(prune='both'))
            ax.get_yaxis().set_ticks([])
            ax.set_xlabel('time (s)')
            ax.legend(loc='upper right')

        elif d.mode == 'map':
            assert len(d.list_traces) == 1, 'plot_timeseries with mode="map" do not support multi recording'
            extent = (d.time_range[0], d.time_range[1], 0, len(d.channel_ids))
            im = ax.imshow(d.list_traces[0].T, interpolation='nearest',
                           origin='upper', aspect='auto', extent=extent, cmap=d.cmap)

            if d.clim is None:
                im.set_clim(-d.max_channel_amp, d.max_channel_amp)
            else:
                im.set_clim(*d.clim)

            if d.with_colorbar:
                self.figure.colorbar(im, ax=ax)

            if d.show_channel_ids:
                ax.set_yticks(np.arange(n) + 0.5)
                ax.set_yticklabels([str(chan_id) for chan_id in d.channel_ids[::-1]])

TimeseriesPlotter.register(TimeseriesWidget)
