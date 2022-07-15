import numpy as np
import warnings

from ..base import to_attr
from ..timeseries import TimeseriesWidget
from ..utils import array_to_image
from .base_sortingview import SortingviewPlotter

try:
    from figurl_tiled_image import TiledImage
    HAVE_TILED_IMAGE = True
except ImportError:
    HAVE_TILED_IMAGE = False


class TimeseriesPlotter(SortingviewPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        assert HAVE_TILED_IMAGE, ("To use the sortingview backend for timeseries, you neeed the 'figurl_tiled_image'. "
                                  "Install it with >>> pip install figurl_tiled_image")
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        dp = to_attr(data_plot)
        
        assert dp.mode == 'map', 'sortingview plot_timeseries is only mode="map"'

        tiled_image = TiledImage(tile_size=dp.tile_size)
        
        if not dp.order_channel_by_depth:
            warnings.warn("It is recommended to set 'order_channel_by_depth' to True "
                          "when using the sortingview backend")
        
        for layer_key, traces in zip(dp.layer_keys, dp.list_traces):     
            img = array_to_image(traces, 
                                 clim=dp.clims[layer_key],
                                 num_timepoints_per_row=dp.num_timepoints_per_row,
                                 colormap=dp.cmap,
                                 scalebar=True,
                                 sampling_frequency=dp.recordings[layer_key].get_sampling_frequency())
            
            tiled_image.add_layer(layer_key, img)
        
        if backend_kwargs["generate_url"]: 
            if backend_kwargs.get("figlabel") is None:
                label = "SpikeInterface - Timeseries"
            url = tiled_image.url(label=label, verbose=False)
            print(url)
        return tiled_image


TimeseriesPlotter.register(TimeseriesWidget)
