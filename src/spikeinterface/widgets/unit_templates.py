from __future__ import annotations

from spikeinterface.core import SortingAnalyzer
from .unit_waveforms import UnitWaveformsWidget
from .base import to_attr


class UnitTemplatesWidget(UnitWaveformsWidget):
    # doc is copied from UnitWaveformsWidget

    def __init__(self, *args, **kargs):
        kargs["plot_waveforms"] = False
        UnitWaveformsWidget.__init__(self, *args, **kargs)

    def plot_sortingview(self, data_plot, **backend_kwargs):
        self.plot_figpack(data_plot, use_sortingview=True, **backend_kwargs)

    def plot_figpack(self, data_plot, **backend_kwargs):
        from .utils_figpack import (
            make_serializable,
            handle_display_and_url,
            import_figpack_or_sortingview,
            generate_unit_table_view,
        )

        use_sortingview = backend_kwargs.get("use_sortingview", False)
        vv_base, vv_views = import_figpack_or_sortingview(use_sortingview)

        dp = to_attr(data_plot)

        sorting_analyzer = dp.sorting_analyzer_or_templates
        assert isinstance(sorting_analyzer, SortingAnalyzer), "This widget requires a SortingAnalyzer as input"

        assert len(dp.templates_shading) in (2, 4), "Only 2 ans 4 templates shading are supported in sortingview"

        # ensure serializable for sortingview
        unit_id_to_channel_ids = dp.final_sparsity.unit_id_to_channel_ids
        unit_id_to_channel_indices = dp.final_sparsity.unit_id_to_channel_indices

        unit_ids, channel_ids = make_serializable(dp.unit_ids, sorting_analyzer.channel_ids)

        templates_dict = {}
        for u_i, unit in enumerate(unit_ids):
            templates_dict[unit] = {}
            template_data = dp.templates[u_i]
            templates_dict[unit]["mean"] = template_data[:, unit_id_to_channel_indices[unit]]
            if dp.do_shading:
                templates_dict[unit]["shading"] = []
                for shading_data in dp.templates_shading:
                    templates_dict[unit]["shading"].append(shading_data[u_i][:, unit_id_to_channel_indices[unit]])
            else:
                templates_dict[unit]["shading"] = None

        aw_items = [
            vv_views.AverageWaveformItem(
                unit_id=u,
                channel_ids=list(unit_id_to_channel_ids[u]),
                waveform=t["mean"],
                waveform_percentiles=t["shading"],
            )
            for u, t in templates_dict.items()
        ]

        channel_locations = {str(ch): dp.channel_locations[i_ch].astype(float) for i_ch, ch in enumerate(channel_ids)}
        v_average_waveforms = vv_views.AverageWaveforms(average_waveforms=aw_items, channel_locations=channel_locations)

        if not dp.hide_unit_selector:
            v_units_table = generate_unit_table_view(sorting_analyzer.sorting, use_sortingview=use_sortingview)

            self.view = vv_base.Box(
                direction="horizontal",
                items=[vv_base.LayoutItem(v_units_table, max_size=150), vv_base.LayoutItem(v_average_waveforms)],
            )
        else:
            self.view = v_average_waveforms

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)


UnitTemplatesWidget.__doc__ = UnitWaveformsWidget.__doc__
