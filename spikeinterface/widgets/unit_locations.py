import numpy as np
from typing import Union

from .base import BaseWidget
from .utils import get_unit_colors
from ..core.waveform_extractor import WaveformExtractor



class UnitLocationsWidget(BaseWidget):
    """
    Plots unit locations.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The object to compute/get unit locations from
    unit_ids: list
        List of unit ids.
    with_channel_ids: bool False default
        Add channel ids text on the probe
    unit_colors :  dict or None
        If given, a dictionary with unit ids as keys and colors as values
    hide_unit_selector : bool
        If True, the unit selector is not displayed.
        Default False (sortingview backend)
    plot_all_units : bool
        If True, all units are plotted. The unselected ones (not in unit_ids),
        are plotted in grey. Default True (matplotlib backend)
    plot_legend : bool
        If True, the legend is plotted. Default False (matplotlib backend)
    hide_axis : bool
        If True, the axis is set to off. Default False (matplotlib backend)
    """
    possible_backends = {}

    def __init__(self, waveform_extractor: WaveformExtractor, 
                 unit_ids=None, with_channel_ids=False,
                 unit_colors=None, hide_unit_selector=False,
                 plot_all_units=True, plot_legend=False, hide_axis=False,
                 backend=None, **backend_kwargs):
        self.check_extensions(waveform_extractor, "unit_locations")
        ulc = waveform_extractor.load_extension("unit_locations")
        unit_locations = ulc.get_data(outputs="by_unit")

        sorting = waveform_extractor.sorting
        
        channel_ids = waveform_extractor.channel_ids
        channel_locations = waveform_extractor.get_channel_locations()
        probegroup = waveform_extractor.get_probegroup()
        
        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)
        
        if unit_ids is None:
            unit_ids = sorting.unit_ids

        plot_data = dict(
            all_unit_ids=sorting.unit_ids,
            unit_locations=unit_locations,
            sorting=sorting,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
            unit_colors=unit_colors,
            channel_locations=channel_locations,
            probegroup_dict=probegroup.to_dict(),
            with_channel_ids=with_channel_ids,
            hide_unit_selector=hide_unit_selector,
            plot_all_units=plot_all_units,
            plot_legend=plot_legend,
            hide_axis=hide_axis
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)



