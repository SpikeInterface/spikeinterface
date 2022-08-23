import numpy as np
from typing import Union

from .base import BaseWidget, define_widget_function_from_class
from .utils import get_unit_colors
from ..core.waveform_extractor import WaveformExtractor
from ..core.basesorting import BaseSorting
from ..postprocessing import compute_spike_locations


class SpikeLocationsWidget(BaseWidget):
    """
    Plots spike locations.

    Parameters
    ----------
    waveform_or_sorting_extractor : WaveformExtractor or BaseSorting
        The object to compute/get crosscorrelograms from
    unit_ids: list
        List of unit ids.
    with_channel_ids: bool False default
        add channel ids text on the probe
    compute_kwargs : dict or None
        If given, dictionary with keyword arguments for `compute_unit_locations` function
    unit_colors :  dict or None
        If given, a dictionary with unit ids as keys and colors as values
    hide_unit_selector : bool
        For sortingview backend, if True the unit selector is not displayed
    """
    possible_backends = {}

    def __init__(self, waveform_extractor: Union[WaveformExtractor, BaseSorting], 
                 unit_ids=None, segment_index=None,
                 with_channel_ids=False, compute_kwargs=None, unit_colors=None, 
                 hide_unit_selector=False, units_in_legend=None,
                 backend=None, **backend_kwargs):
        if waveform_extractor.is_extension("spike_locations"):
            slc = waveform_extractor.load_extension("spike_locations")
            spike_locations = slc.get_data(outputs="by_unit")
        else:
            compute_kwargs = compute_kwargs if compute_kwargs is not None else {}
            spike_locations = compute_spike_locations(waveform_extractor, 
                                                      outputs="by_unit",
                                                      **compute_kwargs)
        
        recording = waveform_extractor.recording
        sorting = waveform_extractor.sorting
        
        channel_ids = recording.channel_ids
        channel_locations = recording.get_channel_locations()
        probegroup = recording.get_probegroup()
        
        if sorting.get_num_segments() > 1:
            assert segment_index is not None, "Specify segment index for multi-segment object"
        else:
            segment_index = 0
        
        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)
        
        if unit_ids is None:
            unit_ids = sorting.unit_ids
        else:
            unit_indices = sorting.ids_to_indices(unit_ids)
            unit_locations = unit_locations[unit_indices]

        plot_data = dict(
            spike_locations=spike_locations[segment_index],
            unit_ids=unit_ids,
            channel_ids=channel_ids,
            unit_colors=unit_colors,
            channel_locations=channel_locations,
            probegroup_dict=probegroup.to_dict(),
            with_channel_ids=with_channel_ids,
            hide_unit_selector=hide_unit_selector,
            units_in_legend=units_in_legend
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)


plot_spike_locations = define_widget_function_from_class(SpikeLocationsWidget, 'plot_spike_locations')
