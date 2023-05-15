from ..base import to_attr
from ..crosscorrelograms import CrossCorrelogramsWidget
from .base_sortingview import SortingviewPlotter


class CrossCorrelogramsPlotter(SortingviewPlotter):
    default_label = "SpikeInterface - Cross Correlograms"

    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        dp = to_attr(data_plot)
        
        unit_ids = self.make_serializable(dp.unit_ids)

        cc_items = []
        for i in range(len(unit_ids)):
            for j in range(i, len(unit_ids)):
                cc_items.append(
                    vv.CrossCorrelogramItem(
                        unit_id1=unit_ids[i],
                        unit_id2=unit_ids[j],
                        bin_edges_sec=(dp.bins/1000.).astype("float32"),
                        bin_counts=dp.correlograms[i, j].astype("int32")
                    )
                )

        v_cross_correlograms = vv.CrossCorrelograms(
            cross_correlograms=cc_items,
            hide_unit_selector=dp.hide_unit_selector
        )

        self.handle_display_and_url(v_cross_correlograms, **backend_kwargs)
        return v_cross_correlograms


CrossCorrelogramsPlotter.register(CrossCorrelogramsWidget)
