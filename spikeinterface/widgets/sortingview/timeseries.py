import numpy as np
import warnings

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
        tile_size = data_plot["tile_size"]
        num_timepoints_per_row = data_plot["num_timepoints_per_row"]
        
        tiled_image = TiledImage(tile_size=tile_size)
        
        if not data_plot["order_channel_by_depth"]:
            warnings.warn("It is recommended to set 'order_channel_by_depth' to True "
                          "when using the sortingview backend")
        
        for layer_key, traces in zip(data_plot["layer_keys"], data_plot["list_traces"]):     
            img = array_to_image(traces, 
                                 clim=data_plot["clims"][layer_key],
                                 num_timepoints_per_row=num_timepoints_per_row,
                                 colormap=data_plot["cmap"])
            
            tiled_image.add_layer(layer_key, img)
        
        if backend_kwargs["generate_url"]: 
            label = backend_kwargs.get("figlabel", "SpikeInterface - Timeseries")
            url = tiled_image.url(label=label, verbose=False)
            print(url)
        return tiled_image


TimeseriesPlotter.register(TimeseriesWidget)
