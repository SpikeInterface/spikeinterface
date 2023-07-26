from .unit_waveforms import UnitWaveformsWidget
from .base import to_attr


class UnitTemplatesWidget(UnitWaveformsWidget):
    # doc is copied from UnitWaveformsWidget

    def __init__(self, *args, **kargs):
        kargs["plot_waveforms"] = False
        UnitWaveformsWidget.__init__(self, *args, **kargs)

    def plot_sortingview(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        from .utils_sortingview import generate_unit_table_view, make_serializable, handle_display_and_url

        dp = to_attr(data_plot)

        # ensure serializable for sortingview
        unit_id_to_channel_ids = dp.sparsity.unit_id_to_channel_ids
        unit_id_to_channel_indices = dp.sparsity.unit_id_to_channel_indices

        unit_ids, channel_ids = make_serializable(dp.unit_ids, dp.channel_ids)

        templates_dict = {}
        for u_i, unit in enumerate(unit_ids):
            templates_dict[unit] = {}
            templates_dict[unit]["mean"] = dp.templates[u_i].T.astype("float32")[unit_id_to_channel_indices[unit]]
            templates_dict[unit]["std"] = dp.template_stds[u_i].T.astype("float32")[unit_id_to_channel_indices[unit]]

        aw_items = [
            vv.AverageWaveformItem(
                unit_id=u,
                channel_ids=list(unit_id_to_channel_ids[u]),
                waveform=t["mean"].astype("float32"),
                waveform_std_dev=t["std"].astype("float32"),
            )
            for u, t in templates_dict.items()
        ]

        locations = {str(ch): dp.channel_locations[i_ch].astype("float32") for i_ch, ch in enumerate(channel_ids)}
        v_average_waveforms = vv.AverageWaveforms(average_waveforms=aw_items, channel_locations=locations)

        if not dp.hide_unit_selector:
            v_units_table = generate_unit_table_view(dp.waveform_extractor.sorting)

            self.view = vv.Box(
                direction="horizontal",
                items=[vv.LayoutItem(v_units_table, max_size=150), vv.LayoutItem(v_average_waveforms)],
            )
        else:
            self.view = v_average_waveforms

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)


UnitTemplatesWidget.__doc__ = UnitWaveformsWidget.__doc__
