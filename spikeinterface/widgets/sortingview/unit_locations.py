from ..base import to_attr
from ..unit_locations import UnitLocationsWidget
from .base_sortingview import SortingviewPlotter, generate_unit_table_view


class UnitLocationsPlotter(SortingviewPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        dp = to_attr(data_plot)

        # ensure serializable for sortingview
        unit_ids, channel_ids = self.make_serializable(dp.unit_ids, dp.channel_ids)

        locations = {str(ch): dp.channel_locations[i_ch].astype("float32")
                     for i_ch, ch in enumerate(channel_ids)}

        unit_items = []
        for ii, unit_id in enumerate(unit_ids):
            unit_items.append(vv.UnitLocationsItem(
                unit_id=unit_id,
                x=float(dp.unit_locations[ii][0]),
                y=float(dp.unit_locations[ii][1])
            ))

        v_unit_locations = vv.UnitLocations(
            units=unit_items,
            channel_locations=locations,
            disable_auto_rotate=True
        )

        v_units_table = generate_unit_table_view(unit_ids)

        v_unit_locations_w_table = vv.Box(direction='horizontal',
                      items=[
                          vv.LayoutItem(v_units_table, max_size=150),
                          vv.LayoutItem(v_unit_locations)
                      ]
                )

        if backend_kwargs["generate_url"]:
            if backend_kwargs.get("figlabel") is None:
                label = "SpikeInterface - UnitLocations"
            url = v_unit_locations_w_table.url(label=label)
            print(url)
        return v_unit_locations_w_table


UnitLocationsPlotter.register(UnitLocationsWidget)
