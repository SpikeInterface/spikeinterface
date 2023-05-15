from ..base import to_attr
from ..spike_locations import SpikeLocationsWidget, estimate_axis_lims
from .base_sortingview import SortingviewPlotter, generate_unit_table_view


class SpikeLocationsPlotter(SortingviewPlotter):
    default_label = "SpikeInterface - Spike Locations"

    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        dp = to_attr(data_plot)
        spike_locations = dp.spike_locations

        # ensure serializable for sortingview
        unit_ids, channel_ids = self.make_serializable(dp.unit_ids, dp.channel_ids)

        locations = {
            str(ch): dp.channel_locations[i_ch].astype("float32")
            for i_ch, ch in enumerate(channel_ids)
        }
        xlims, ylims = estimate_axis_lims(spike_locations)

        unit_items = []
        for unit in unit_ids:
            spike_times_sec = dp.sorting.get_unit_spike_train(
                unit_id=unit, segment_index=dp.segment_index, return_times=True
            )
            unit_items.append(
                vv.SpikeLocationsItem(
                    unit_id=unit,
                    spike_times_sec=spike_times_sec.astype("float32"),
                    x_locations=spike_locations[unit]["x"].astype("float32"),
                    y_locations=spike_locations[unit]["y"].astype("float32"),
                )
            )

        v_spike_locations = vv.SpikeLocations(
            units=unit_items,
            hide_unit_selector=dp.hide_unit_selector,
            x_range=xlims.astype("float32"),
            y_range=ylims.astype("float32"),
            channel_locations=locations,
            disable_auto_rotate=True,
        )

        if not dp.hide_unit_selector:
            v_units_table = generate_unit_table_view(dp.sorting)

            view = vv.Box(
                direction="horizontal",
                items=[
                    vv.LayoutItem(v_units_table, max_size=150),
                    vv.LayoutItem(v_spike_locations),
                ],
            )
        else:
            view = v_spike_locations

        self.set_view(view)

        self.handle_display_and_url(view, **backend_kwargs)
        return view


SpikeLocationsPlotter.register(SpikeLocationsWidget)
