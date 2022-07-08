import numpy as np

from ..unit_waveforms import UnitWaveformsWidget
from .base_mpl import MplPlotter, to_attr


class UnitWaveformPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        d = to_attr(data_plot)

        ncols = min(d.ncols, len(d.unit_ids))
        nrows = int(np.ceil(len(d.unit_ids) / ncols))

        self.make_mpl_figure(ncols=ncols, num_axes=d.num_axes, **backend_kwargs)
        
        for i, unit_id in enumerate(d.unit_ids):

            ax = self.axes.flatten()[i]
            color = d.unit_colors[unit_id]

            chan_inds = d.channel_inds[unit_id]
            xvectors_flat = d.xvectors[:, chan_inds].T.flatten()

            # plot waveforms
            if d.plot_waveforms:
                wfs = d.wfs_by_ids[unit_id]
                if d.unit_selected_waveforms is not None:
                    wfs = wfs[d.unit_selected_waveforms[unit_id], :, chan_inds]
                else:
                    wfs = wfs[:, :, chan_inds]
                wfs = wfs * d.y_scale + d.y_offset[None, :, chan_inds]
                wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1).T
                ax.plot(xvectors_flat, wfs_flat, lw=1, alpha=0.3, color=color)

            # plot template
            if d.plot_templates:
                template = d.all_templates[i, :, :][:, chan_inds] * d.y_scale + d.y_offset[:, chan_inds]
                if d.plot_waveforms and d.plot_templates:
                    color = 'k'
                ax.plot(xvectors_flat, template.T.flatten(), lw=1, color=color)
                template_label = d.unit_ids[i]
                ax.set_title(f'template {template_label}')

            # plot channels
            if d.plot_channels:
                # TODO enhance this
                ax.scatter(d.channel_locations[:, 0], d.channel_locations[:, 1], color='k')



UnitWaveformPlotter.register(UnitWaveformsWidget)
