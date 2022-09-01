import numpy as np
from typing import Union

from .base import BaseWidget
from ..core.waveform_extractor import WaveformExtractor
from ..core.basesorting import BaseSorting
from ..postprocessing import compute_correlograms


class CrossCorrelogramsWidget(BaseWidget):
    """
    Plots unit cross correlograms.

    Parameters
    ----------
    waveform_or_sorting_extractor : WaveformExtractor or BaseSorting
        The object to compute/get crosscorrelograms from
    unit_ids: list
        List of unit ids.
    window_ms : float
        Window for CCGs in ms, by default 100 ms
    bin_ms : float
        Bin size in ms, by default 1 ms
    hide_unit_selector : bool
        For sortingview backend, if True the unit selector is not displayed
    """
    possible_backends = {}

    def __init__(self, waveform_or_sorting_extractor: Union[WaveformExtractor, BaseSorting], 
                 unit_ids=None, window_ms=100.0, bin_ms=1.0, hide_unit_selector=False,
                 backend=None, **backend_kwargs):
        if isinstance(waveform_or_sorting_extractor, WaveformExtractor):
            sorting = waveform_or_sorting_extractor.sorting
            self.check_extensions(waveform_or_sorting_extractor, "correlograms")
            ccc = waveform_or_sorting_extractor.load_extension("correlograms")
            ccgs, bins = ccc.get_data()
        else:
            sorting = waveform_or_sorting_extractor
            ccgs, bins = compute_correlograms(sorting,
                                              window_ms=window_ms,
                                              bin_ms=bin_ms, symmetrize=True)
            
        if unit_ids is None:
            unit_ids = sorting.unit_ids
            correlograms = ccgs
        else:
            unit_indices = sorting.ids_to_indices(unit_ids)
            correlograms = ccgs[unit_indices][:, unit_indices]

        plot_data = dict(
            correlograms=correlograms,
            bins=bins,
            unit_ids=unit_ids,
            hide_unit_selector=hide_unit_selector
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)



