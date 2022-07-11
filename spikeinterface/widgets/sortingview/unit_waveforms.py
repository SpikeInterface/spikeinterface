from ..base import to_attr
from .base_sortingview import SortingviewPlotter


class UnitWaveformPlotter(SortingviewPlotter):
    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv

        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        
        # ensure serializable for sortingview
        unit_ids, channel_ids, sparsity = self.make_serializable(dp.unit_ids, dp.channel_ids, dp.sparsity)

        templates_dict = {}
        for u_i, unit in enumerate(unit_ids):
            templates_dict[unit] = {}
            templates_dict[unit]["mean"] = dp.templates[u_i].T
            templates_dict[unit]["std"] = dp.template_stds[u_i].T

        aw_items = [
            vv.AverageWaveformItem(
                unit_id=u,
                channel_ids=list(dp.sparsity[u]),
                waveform=t['mean'].astype('float32'),
                waveform_std_dev=t['std'].astype('float32')
            )
            for u, t in templates_dict.items()
        ]

        locations = {ch: dp.channel_locations[i_ch].astype("float32")
                     for i_ch, ch in enumerate(channel_ids)}
        v_average_waveforms = vv.AverageWaveforms(
            average_waveforms=aw_items,
            channel_locations=locations
        )
        if backend_kwargs["generate_url"]:
            if backend_kwargs.get("figlabel") is None:
                label = "SpikeInterface - AverageWaveforms"
            url = v_average_waveforms.url(label=label)
            print(url)
        return v_average_waveforms


from ..unit_waveforms import UnitWaveformsWidget
UnitWaveformPlotter.register(UnitWaveformsWidget)
