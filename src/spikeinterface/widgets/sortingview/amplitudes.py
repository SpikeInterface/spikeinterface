import numpy as np

from ..base import to_attr
from ..amplitudes import AmplitudesWidget
from .base_sortingview import SortingviewPlotter


class AmplitudesPlotter(SortingviewPlotter):
    default_label = "SpikeInterface - Amplitudes"

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

        self.handle_display_and_url(v_spike_amplitudes, **backend_kwargs)
        return v_spike_amplitudes
        

AmplitudesPlotter.register(AmplitudesWidget)
