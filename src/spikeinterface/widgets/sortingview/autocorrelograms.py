from ..base import to_attr
from ..autocorrelograms import AutoCorrelogramsWidget
from .base_sortingview import SortingviewPlotter


class AutoCorrelogramsPlotter(SortingviewPlotter):
    default_label = "SpikeInterface - Auto Correlograms"

    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        dp = to_attr(data_plot)
        unit_ids = self.make_serializable(dp.unit_ids)

        ac_items = []
        for i in range(len(unit_ids)):
            for j in range(i, len(unit_ids)):
                if i == j:
                    ac_items.append(
                        vv.AutocorrelogramItem(
                            unit_id=unit_ids[i],
                            bin_edges_sec=(dp.bins/1000.).astype("float32"),
                            bin_counts=dp.correlograms[i, j].astype("int32")
                        )
                    )

        v_autocorrelograms = vv.Autocorrelograms(
            autocorrelograms=ac_items
        )

        self.handle_display_and_url(v_autocorrelograms, **backend_kwargs)
        return v_autocorrelograms


AutoCorrelogramsPlotter.register(AutoCorrelogramsWidget)
