from ..base import to_attr
from ..unit_templates import UnitTemplatesWidget
from .base_sortingview import SortingviewPlotter, generate_unit_table_view


class UnitTemplatesPlotter(SortingviewPlotter):
    default_label = "SpikeInterface - Unit Templates"

    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv

        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        
        # ensure serializable for sortingview
        unit_ids, channel_ids, sparsity = self.make_serializable(dp.unit_ids, dp.channel_ids, dp.sparsity)
        channel_inds = dp.channel_inds

        templates_dict = {}
        for u_i, unit in enumerate(unit_ids):
            templates_dict[unit] = {}
            templates_dict[unit]["mean"] = dp.templates[u_i].T.astype("float32")[channel_inds[unit]]
            templates_dict[unit]["std"] = dp.template_stds[u_i].T.astype("float32")[channel_inds[unit]]

        aw_items = [
            vv.AverageWaveformItem(
                unit_id=u,
                channel_ids=list(sparsity[u]),
                waveform=t['mean'].astype('float32'),
                waveform_std_dev=t['std'].astype('float32')
            )
            for u, t in templates_dict.items()
        ]

        locations = {str(ch): dp.channel_locations[i_ch].astype("float32")
                     for i_ch, ch in enumerate(channel_ids)}
        v_average_waveforms = vv.AverageWaveforms(
            average_waveforms=aw_items,
            channel_locations=locations
        )

        if not dp.hide_unit_selector:
            v_units_table = generate_unit_table_view(dp.waveform_extractor.sorting)

            view = vv.Box(direction='horizontal',
                        items=[
                            vv.LayoutItem(v_units_table, max_size=150),
                            vv.LayoutItem(v_average_waveforms)
                        ]
                    )
        else:
            view = v_average_waveforms

        self.handle_display_and_url(view, **backend_kwargs)
        return view


UnitTemplatesPlotter.register(UnitTemplatesWidget)
