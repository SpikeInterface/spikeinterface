import numpy as np

from ..base import to_attr
from ..unit_waveforms import UnitWaveformsWidget
from .base_mpl import MplPlotter


class UnitWaveformPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        backend_kwargs["ncols"] = min(dp.ncols, len(dp.unit_ids))
        backend_kwargs["num_axes"] = dp.num_axes

        self.make_mpl_figure(**backend_kwargs)
        
        for i, unit_id in enumerate(dp.unit_ids):

            ax = self.axes.flatten()[i]
            color = dp.unit_colors[unit_id]

            chan_inds = dp.channel_inds[unit_id]
            xvectors_flat = dp.xvectors[:, chan_inds].T.flatten()

            # plot waveforms
            if dp.plot_waveforms:
                wfs = dp.wfs_by_ids[unit_id]
                if dp.unit_selected_waveforms is not None:
                    wfs = wfs[dp.unit_selected_waveforms[unit_id], :, chan_inds]
                else:
                    wfs = wfs[:, :, chan_inds]
                wfs = wfs * dp.y_scale + dp.y_offset[None, :, chan_inds]
                wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1).T
                ax.plot(xvectors_flat, wfs_flat, lw=1, alpha=0.3, color=color)

            # plot template
            if dp.plot_templates:
                template = dp.all_templates[i, :, :][:, chan_inds] * dp.y_scale + dp.y_offset[:, chan_inds]
                if dp.plot_waveforms and dp.plot_templates:
                    color = 'k'
                ax.plot(xvectors_flat, template.T.flatten(), lw=1, color=color)
                template_label = dp.unit_ids[i]
                ax.set_title(f'template {template_label}')

            # plot channels
            if dp.plot_channels:
                # TODO enhance this
                ax.scatter(dp.channel_locations[:, 0], dp.channel_locations[:, 1], color='k')



UnitWaveformPlotter.register(UnitWaveformsWidget)
