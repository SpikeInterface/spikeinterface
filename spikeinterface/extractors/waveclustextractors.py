from pathlib import Path
import numpy as np

from spikeinterface.core import (BaseRecording, BaseSorting,
                                 BaseRecordingSegment, BaseSortingSegment)
from .matlabhelpers import MatlabHelper


class WaveClusSortingExtractor(MatlabHelper, BaseSorting):
    extractor_name = "WaveClusSortingExtractor"
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path, keep_good_only=True):
        MatlabHelper.__init__(self, file_path)

        cluster_classes = self._getfield("cluster_class")
        classes = cluster_classes[:, 0]
        spike_times = cluster_classes[:, 1]
        par = self._getfield("par")
        sampling_frequency = par[0, 0][np.where(np.array(par.dtype.names) == 'sr')[0][0]][0][0]
        unit_ids = np.unique(classes).astype('int')
        if keep_good_only:
            unit_ids = unit_ids[unit_ids > 0]
        spiketrains = {}
        for unit_id in unit_ids:
            mask = (classes == unit_id)
            spiketrains[unit_id] = np.rint(spike_times[mask] * (sampling_frequency / 1000))

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        self.add_sorting_segment(WaveClustSortingSegment(unit_ids, spiketrains))
        self.set_property('unsorted', np.array([c == 0 for c in unit_ids]))
        self._kwargs = {'file_path': str(Path(file_path).absolute())}


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


def read_waveclust(*args, **kwargs):
    sorting = WaveClusSortingExtractor(*args, **kwargs)
    return sorting


read_waveclust.__doc__ = WaveClusSortingExtractor.__doc__
