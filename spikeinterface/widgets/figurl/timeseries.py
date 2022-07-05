from tabnanny import verbose
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
        if "color_range" in backend_kwargs:
            color_ranges = backend_kwargs["color_range"]
        else:
            color_ranges = 200
        if not isinstance(color_ranges, list):
            color_ranges = [color_ranges] * len(data_plot["layer_keys"])
            
        if "tile_size" in backend_kwargs:
            tile_size = backend_kwargs["tile_size"]
        else:
            tile_size = 512
            
        if "num_timepoints_per_row" in backend_kwargs:
            num_timepoints_per_row = backend_kwargs["num_timepoints_per_row"]
        else:
            num_timepoints_per_row = 6000
        
        tiled_image = TiledImage(tile_size=tile_size)
        
        if not data_plot["order_channel_by_depth"]:
            warnings.warn("It is recommended to set 'order_channel_by_depth' to True when using the figurl backend")
        
        for layer_key, traces, cr in zip(data_plot["layer_keys"], data_plot["list_traces"], color_ranges):
            color_range = cr
            
            img = array_to_image(traces, 
                                 color_range=color_range,
                                 num_timepoints_per_row=num_timepoints_per_row,
                                 colormap=data_plot["cmap"])
            
            tiled_image.add_layer(layer_key, img)
            
        url = tiled_image.url(label=f'SpikeInterface - {data_plot["layer_keys"]}', verbose=False)
        print(url)


TimeseriesPlotter.register(TimeseriesWidget)
