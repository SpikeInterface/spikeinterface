import numpy as np

from ..unit_waveforms_density_map import UnitWaveformDensityMapWidget
from .base_mpl import MplPlotter, to_attr


class UnitWaveformDensityMapPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        d = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        if 'axes' in backend_kwargs or 'ax' in backend_kwargs:
            self.make_mpl_figure(**backend_kwargs)
        else:
            if d.same_axis:
                num_axes = 1
            else:
                num_axes = len(d.unit_ids)
            backend_kwargs["ncols"] = 1
            backend_kwargs["num_axes"] = num_axes
            self.make_mpl_figure(**backend_kwargs)
        
        if d.same_axis:
            ax = self.ax
            hist2d = d.all_hist2d
            im = ax.imshow(hist2d.T, interpolation='nearest',
                           origin='lower', aspect='auto', extent=(0, hist2d.shape[0], d.bin_min, d.bin_max), cmap='hot')
        else:
            for unit_index, unit_id in enumerate(d.unit_ids):
                hist2d = d.all_hist2d[unit_id]
                ax = self.axes[unit_index]
                im = ax.imshow(hist2d.T, interpolation='nearest',
                    origin='lower', aspect='auto',
                    extent=(0, hist2d.shape[0], d.bin_min, d.bin_max), cmap='hot')

        for unit_index, unit_id in enumerate(d.unit_ids):
            if d.same_axis:
                ax = self.ax
            else:
                ax = self.axes[unit_index]
            color = d.unit_colors[unit_id]
            ax.plot(d.templates_flat[unit_id], color=color, lw=1)

        # final cosmetics
        for unit_index, unit_id in enumerate(d.unit_ids):
            if d.same_axis:
                ax = self.ax
                if unit_index != 0:
                    continue
            else:
                ax = self.axes[unit_index]
            chan_inds = d.channel_inds[unit_id]
            for i, chan_ind in enumerate(chan_inds):
                if i != 0:
                    ax.axvline(i * d.template_width, color='w', lw=3)
                channel_id = d.channel_ids[chan_ind]
                x = i * d.template_width + d.template_width // 2
                y = (d.bin_max + d.bin_min) / 2.
                ax.text(x, y, f'chan_id {channel_id}', color='w', ha='center', va='center')

            ax.set_xticks([])
            ax.set_ylabel(f'unit_id {unit_id}')



UnitWaveformDensityMapPlotter.register(UnitWaveformDensityMapWidget)
