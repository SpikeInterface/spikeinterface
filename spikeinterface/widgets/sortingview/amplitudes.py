import numpy as np

from ..base import to_attr
from ..amplitudes import AmplitudesWidget
from .base_sortingview import SortingviewPlotter


class AmplitudesPlotter(SortingviewPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        dp = to_attr(data_plot)
        
        unit_ids = self.make_serializable(dp.unit_ids)

        sa_items = [
            vv.SpikeAmplitudesItem(
                unit_id=u,
                spike_times_sec=dp.spiketrains[u].astype("float32"),
                spike_amplitudes=dp.amplitudes[u].astype("float32")
            )
            for u in unit_ids
        ]

        v_spike_amplitudes = vv.SpikeAmplitudes(
            start_time_sec=0,
            end_time_sec=dp.total_duration,
            plots=sa_items,
            hide_unit_selector=dp.hide_unit_selector
        )
        self.set_view(v_spike_amplitudes)

        if backend_kwargs["generate_url"]:
            if backend_kwargs.get("figlabel") is None:
                label = "SpikeInterface - SpikeAmplitudes"
            url = v_spike_amplitudes.url(label=label)
            print(url)
        return v_spike_amplitudes
        

AmplitudesPlotter.register(AmplitudesWidget)
