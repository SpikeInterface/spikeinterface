import numpy as np

from ..base import to_attr
from ..unit_waveforms_density_map import UnitWaveformDensityMapWidget
from .base_mpl import MplPlotter


class UnitWaveformDensityMapPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        if backend_kwargs["axes"] is not None or backend_kwargs["ax"] is not None:
            self.make_mpl_figure(**backend_kwargs)
        else:
            if dp.same_axis:
                num_axes = 1
            else:
                num_axes = len(dp.unit_ids)
            backend_kwargs["ncols"] = 1
            backend_kwargs["num_axes"] = num_axes
            self.make_mpl_figure(**backend_kwargs)
        
        if dp.same_axis:
            ax = self.ax
            hist2d = dp.all_hist2d
            im = ax.imshow(hist2d.T, interpolation='nearest',
                           origin='lower', aspect='auto',
                           extent=(0, hist2d.shape[0], dp.bin_min, dp.bin_max), cmap='hot')
        else:
            for unit_index, unit_id in enumerate(dp.unit_ids):
                hist2d = dp.all_hist2d[unit_id]
                ax = self.axes.flatten()[unit_index]
                im = ax.imshow(hist2d.T, interpolation='nearest',
                    origin='lower', aspect='auto',
                    extent=(0, hist2d.shape[0], dp.bin_min, dp.bin_max), cmap='hot')

        for unit_index, unit_id in enumerate(dp.unit_ids):
            if dp.same_axis:
                ax = self.ax
            else:
                ax = self.axes.flatten()[unit_index]
            color = dp.unit_colors[unit_id]
            ax.plot(dp.templates_flat[unit_id], color=color, lw=1)

        # final cosmetics
        for unit_index, unit_id in enumerate(dp.unit_ids):
            if dp.same_axis:
                ax = self.ax
                if unit_index != 0:
                    continue
            else:
                ax = self.axes.flatten()[unit_index]
            chan_inds = dp.channel_inds[unit_id]
            for i, chan_ind in enumerate(chan_inds):
                if i != 0:
                    ax.axvline(i * dp.template_width, color='w', lw=3)
                channel_id = dp.channel_ids[chan_ind]
                x = i * dp.template_width + dp.template_width // 2
                y = (dp.bin_max + dp.bin_min) / 2.
                ax.text(x, y, f'chan_id {channel_id}', color='w', ha='center', va='center')

            ax.set_xticks([])
            ax.set_ylabel(f'unit_id {unit_id}')



UnitWaveformDensityMapPlotter.register(UnitWaveformDensityMapWidget)
