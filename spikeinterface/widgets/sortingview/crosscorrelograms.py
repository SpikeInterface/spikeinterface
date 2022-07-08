from ..crosscorrelograms import CrossCorrelogramsWidget
from .base_sortingview import SortingviewPlotter


class CrossCorrelogramsPlotter(SortingviewPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        
        ccgs = data_plot["correlograms"]
        bins = data_plot["bins"]
        unit_ids = data_plot["unit_ids"]
        
        unit_ids = self.make_serializable(unit_ids)

        cc_items = []
        for i in range(ccgs.shape[0]):
            for j in range(i, ccgs.shape[0]):
                cc_items.append(
                    vv.CrossCorrelogramItem(
                        unit_id1=unit_ids[i],
                        unit_id2=unit_ids[j],
                        bin_edges_sec=(bins/1000.).astype("float32"),
                        bin_counts=ccgs[i, j].astype("int32")
                    )
                )

        v_cross_correlograms = vv.CrossCorrelograms(
            cross_correlograms=cc_items
        )

        if backend_kwargs["generate_url"]:
            label = backend_kwargs.get("figlabel", "SpikeInterface - CrossCorrelograms")
            url = v_cross_correlograms.url(label=label)
            print(url)
        return v_cross_correlograms


CrossCorrelogramsPlotter.register(CrossCorrelogramsWidget)
