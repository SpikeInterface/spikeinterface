from ..autocorrelograms import AutoCorrelogramsWidget
from .base_sortingview import SortingviewPlotter


class AutoCorrelogramsPlotter(SortingviewPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        vv = self.get_sortingviews()
        
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        
        ccgs = data_plot["correlograms"]
        bins = data_plot["bins"]
        unit_ids = data_plot["unit_ids"]
        
        unit_ids = self.make_serializable(unit_ids)

        ac_items = []
        for i in range(ccgs.shape[0]):
            for j in range(i, ccgs.shape[0]):
                if i == j:
                    ac_items.append(
                        vv.AutocorrelogramItem(
                            unit_id=unit_ids[i],
                            bin_edges_sec=(bins/1000.).astype("float32"),
                            bin_counts=ccgs[i, j].astype("int32")
                        )
                    )

        v_autocorrelograms = vv.Autocorrelograms(
            autocorrelograms=ac_items
        )

        if backend_kwargs["generate_url"]:
            url = v_autocorrelograms.url(label='SpikeInterface - AutoCorrelograms')
            print(url)
        return v_autocorrelograms


AutoCorrelogramsPlotter.register(AutoCorrelogramsWidget)
