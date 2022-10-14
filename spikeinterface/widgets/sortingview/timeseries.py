import numpy as np
import warnings

from ..base import to_attr
from ..timeseries import TimeseriesWidget
from ..utils import array_to_image
from .base_sortingview import SortingviewPlotter


class TimeseriesPlotter(SortingviewPlotter):
    default_label = "SpikeInterface - Timeseries"

    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        try:
            import pyvips
        except ImportError:
            raise ImportError("To use the timeseries in sorting view you need the pyvips package.")

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        dp = to_attr(data_plot)

        assert dp.mode == "map", 'sortingview plot_timeseries is only mode="map"'

        if not dp.order_channel_by_depth:
            warnings.warn(
                "It is recommended to set 'order_channel_by_depth' to True "
                "when using the sortingview backend"
            )

        tiled_layers = []
        for layer_key, traces in zip(dp.layer_keys, dp.list_traces):
            img = array_to_image(
                traces,
                clim=dp.clims[layer_key],
                num_timepoints_per_row=dp.num_timepoints_per_row,
                colormap=dp.cmap,
                scalebar=True,
                sampling_frequency=dp.recordings[layer_key].get_sampling_frequency(),
            )

            tiled_layers.append(vv.TiledImageLayer(layer_key, img))

        view_ts = vv.TiledImage(tile_size=dp.tile_size, layers=tiled_layers)

        self.set_view(view_ts)

        # timeseries currently doesn't display on the jupyter backend
        backend_kwargs["display"] = False
        self.handle_display_and_url(view_ts, **backend_kwargs)
        return view_ts


TimeseriesPlotter.register(TimeseriesWidget)
