import numpy as np

from ..base import to_attr
from ..unit_waveforms import UnitWaveformsWidget
from .base_mpl import MplPlotter


class UnitWaveformPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        if backend_kwargs["axes"] is not None:
            assert len(backend_kwargs) >= len(dp.units)
        elif backend_kwargs["ax"] is not None:
            assert dp.same_axis, "If 'same_axis' is not used, provide as many 'axes' as neurons"
        else:
            if dp.same_axis:
                backend_kwargs["num_axes"] = 1
                backend_kwargs["ncols"] = None
            else:
                backend_kwargs["num_axes"] = len(dp.unit_ids)
                backend_kwargs["ncols"] = min(dp.ncols, len(dp.unit_ids))

        self.make_mpl_figure(**backend_kwargs)

        for i, unit_id in enumerate(dp.unit_ids):
            if dp.same_axis:
                ax = self.ax
            else:
                ax = self.axes.flatten()[i]
            color = dp.unit_colors[unit_id]

            chan_inds = dp.sparsity.unit_id_to_channel_indices[unit_id]
            xvectors_flat = dp.xvectors[:, chan_inds].T.flatten()

            # plot waveforms
            if dp.plot_waveforms:
                wfs = dp.wfs_by_ids[unit_id]
                if dp.unit_selected_waveforms is not None:
                    wfs = wfs[dp.unit_selected_waveforms[unit_id]]
                elif dp.max_spikes_per_unit is not None:
                    if len(wfs) > dp.max_spikes_per_unit:
                        random_idxs = np.random.permutation(len(wfs))[:dp.max_spikes_per_unit]
                        wfs = wfs[random_idxs]
                wfs = wfs * dp.y_scale + dp.y_offset[None, :, chan_inds]
                wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1).T

                if dp.x_offset_units:
                    # 0.7 is to match spacing in xvect
                    xvec = xvectors_flat + i * 0.7 * dp.delta_x
                else:
                    xvec = xvectors_flat

                ax.plot(xvec, wfs_flat, lw=dp.lw_waveforms, alpha=dp.alpha_waveforms, color=color)

                if not dp.plot_templates:
                    ax.get_lines()[-1].set_label(f"{unit_id}")

            # plot template
            if dp.plot_templates:
                template = dp.templates[i, :, :][:, chan_inds] * dp.y_scale + dp.y_offset[:, chan_inds]

                if dp.x_offset_units:
                    # 0.7 is to match spacing in xvect
                    xvec = xvectors_flat + i * 0.7 * dp.delta_x
                else:
                    xvec = xvectors_flat

                ax.plot(xvec, template.T.flatten(), lw=dp.lw_templates, alpha=dp.alpha_templates,
                        color=color, label=unit_id)

                template_label = dp.unit_ids[i]
                if dp.set_title:
                    ax.set_title(f'template {template_label}')

            # plot channels
            if dp.plot_channels:
                # TODO enhance this
                ax.scatter(dp.channel_locations[:, 0], dp.channel_locations[:, 1], color='k')
            
            if dp.same_axis and dp.plot_legend:
                self.figure.legend(loc='upper center', bbox_to_anchor=(0.5, 1.),
                                   ncol=5, fancybox=True, shadow=True)


UnitWaveformPlotter.register(UnitWaveformsWidget)
