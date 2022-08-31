import numpy as np
import warnings

from ..base import to_attr
from ..timeseries import TimeseriesWidget
from ..utils import array_to_image
from .base_sortingview import SortingviewPlotter


class TimeseriesPlotter(SortingviewPlotter):
    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv

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

        if backend_kwargs["generate_url"]:
            if backend_kwargs.get("figlabel") is None:
                label = "SpikeInterface - Timeseries"
            url = view_ts.url(label=label)
            print(url)
        return view_ts


TimeseriesPlotter.register(TimeseriesWidget)
