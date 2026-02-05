from __future__ import annotations

from pathlib import Path

import numpy as np

from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class
from .matlabhelpers import MatlabHelper


class WaveClusSortingExtractor(MatlabHelper, BaseSorting):
    """Load WaveClus format data as a sorting extractor.

    Parameters
    ----------
    file_path : str or Path
        Path to the WaveClus file.
    keep_good_only : bool, default: True
        Whether to only keep good units.

    Returns
    -------
    extractor : WaveClusSortingExtractor
        Loaded data.
    """

    def __init__(self, file_path, keep_good_only=True):
        MatlabHelper.__init__(self, file_path)

        cluster_classes = self._getfield("cluster_class")
        classes = cluster_classes[:, 0]
        spike_times = cluster_classes[:, 1]
        sampling_frequency = float(self._getfield("par/sr"))
        unit_ids = np.unique(classes).astype("int")
        if keep_good_only:
            unit_ids = unit_ids[unit_ids > 0]
        spiketrains = {}
        for unit_id in unit_ids:
            mask = classes == unit_id
            spiketrains[unit_id] = np.rint(spike_times[mask] * (sampling_frequency / 1000))

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        self.add_sorting_segment(WaveClustSortingSegment(unit_ids, spiketrains))
        self.set_property("unsorted", np.array([c == 0 for c in unit_ids]))
        self._kwargs = {"file_path": str(Path(file_path).absolute()), "keep_good_only": keep_good_only}


class WaveClustSortingSegment(BaseSortingSegment):
    def __init__(self, unit_ids, spiketrains):
        BaseSortingSegment.__init__(self)
        self._unit_ids = list(unit_ids)
        self._spiketrains = spiketrains

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        times = self._spiketrains[unit_id]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times


read_waveclus = define_function_from_class(source_class=WaveClusSortingExtractor, name="read_waveclus")
