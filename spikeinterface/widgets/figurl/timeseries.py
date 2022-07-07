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
        if "clim" in data_plot:
            clim = data_plot["clim"]
            if np.array(clim).ndim == 1:
                clims = [clim] * len(data_plot["layer_keys"])
            else:
                assert len(clim) == len(data_plot["layer_keys"])
                clims = clim
        else:
            clims = [-200, 200] * len(data_plot["layer_keys"])

        if "tile_size" in data_plot:
            tile_size = data_plot["tile_size"]
        else:
            tile_size = 512
            
        if "num_timepoints_per_row" in data_plot:
            num_timepoints_per_row = data_plot["num_timepoints_per_row"]
        else:
            num_timepoints_per_row = 6000
        
        tiled_image = TiledImage(tile_size=tile_size)
        
        if not data_plot["order_channel_by_depth"]:
            warnings.warn("It is recommended to set 'order_channel_by_depth' to True when using the figurl backend")
        
        for layer_key, traces, clim in zip(data_plot["layer_keys"], data_plot["list_traces"], clims):     
            img = array_to_image(traces, 
                                 clim=clim,
                                 num_timepoints_per_row=num_timepoints_per_row,
                                 colormap=data_plot["cmap"])
            
            tiled_image.add_layer(layer_key, img)
            
        url = tiled_image.url(label=f'SpikeInterface - {data_plot["layer_keys"]}', verbose=False)
        print(url)
        return url


TimeseriesPlotter.register(TimeseriesWidget)
