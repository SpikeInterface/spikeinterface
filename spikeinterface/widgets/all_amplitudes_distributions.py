import numpy as np
from warnings import warn

from .base import BaseWidget
from .utils import get_some_colors

from ..core.waveform_extractor import WaveformExtractor


class AllAmplitudesDistributionsWidget(BaseWidget):
    """
    Plots distributions of amplitudes as violon plot for all or some units.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The input waveform extractor
    unit_ids: list
        List of unit ids.
    unit_colors: None or dict
        Dict of colors
    """
    possible_backends = {}

    
    def __init__(self, waveform_extractor: WaveformExtractor, unit_ids=None, unit_colors=None,
                 backend=None, **backend_kwargs):
        
        we = waveform_extractor

        self.check_extensions(we, "spike_amplitudes")
        amplitudes = we.load_extension('spike_amplitudes').get_data(outputs='by_unit')
        
        num_segments = we.get_num_segments()
        
        if unit_ids is None:
            unit_ids = we.unit_ids
        
        if unit_colors is None:
            unit_colors = get_some_colors(we.unit_ids)

        
        plot_data = dict(
            unit_ids=unit_ids,
            unit_colors=unit_colors,
            num_segments=num_segments,
            amplitudes=amplitudes,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)