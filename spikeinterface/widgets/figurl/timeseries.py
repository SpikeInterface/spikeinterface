import numpy as np
import warnings

from ..timeseries import TimeseriesWidget
from ..utils import array_to_image
from .base_figurl import FigurlPlotter

try:
    from figurl_tiled_image import TiledImage
    HAVE_TILED_IMAGE = True
except ImportError:
    HAVE_TILED_IMAGE = False


class TimeseriesPlotter(FigurlPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        assert HAVE_TILED_IMAGE, ("To use the figurl backend for timeseries, you neeed the 'figurl_tiled_image'. "
                                  "Install it with >>> pip install figurl_tiled_image")
        tile_size = data_plot.get("tile_size", 512)
        num_timepoints_per_row = data_plot.get("num_timepoints_per_row", 6000)
        
        tiled_image = TiledImage(tile_size=tile_size)
        
        if not data_plot["order_channel_by_depth"]:
            warnings.warn("It is recommended to set 'order_channel_by_depth' to True when using the figurl backend")
        
        for layer_key, traces in zip(data_plot["layer_keys"], data_plot["list_traces"]):     
            img = array_to_image(traces, 
                                 clim=data_plot["clims"][layer_key],
                                 num_timepoints_per_row=num_timepoints_per_row,
                                 colormap=data_plot["cmap"])
            
            tiled_image.add_layer(layer_key, img)
            
        url = tiled_image.url(label=f'SpikeInterface - {data_plot["layer_keys"]}', verbose=False)
        print(url)
        return url


TimeseriesPlotter.register(TimeseriesWidget)
