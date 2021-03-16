from pathlib import Path
import numpy as np

from spikeinterface.core import (BaseRecording, BaseSorting,
                                BaseRecordingSegment, BaseSortingSegment)
from .matlabhelpers import MatlabHelper



class WaveClusSortingExtractor(MatlabHelper, BaseSorting):
    extractor_name = "WaveClusSortingExtractor"
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path):
        MatlabHelper.__init__(self, file_path)
        
        cluster_classes = self._getfield("cluster_class")
        classes = cluster_classes[:, 0]
        spike_times = cluster_classes[:, 1]
        par = self._getfield("par")
        sampling_frequency = par[0, 0][np.where(np.array(par.dtype.names) == 'sr')[0][0]][0][0]
        unit_ids = np.unique(classes[classes > 0]).astype('int')

        spiketrains = {}
        for unit_id in self._unit_ids:
            mask = (classes == unit_id)
            spiketrains[unit_id] = np.rint(spike_times[mask] * (sampling_frequency/1000))
        self._unsorted_train = np.rint(spike_times[classes == 0] * (sampling_frequency / 1000))

        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        
        self.add_sorting_segment(WaveClustSortingSegment(unit_ids, spiketrains))




    #~ @check_valid_unit_id
    #~ def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        #~ start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)

        #~ start_frame = start_frame or 0
        #~ end_frame = end_frame or np.infty
        #~ st = self._spike_trains[unit_id]
        #~ return st[(st >= start_frame) & (st < end_frame)]

    #~ def get_unit_ids(self):
        #~ return self._unit_ids.tolist()

    #~ def get_unsorted_spike_train(self, start_frame=None, end_frame=None):
        #~ start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)

        #~ start_frame = start_frame or 0
        #~ end_frame = end_frame or np.infty
        #~ u = self._unsorted_train
        #~ return u[(u >= start_frame) & (u < end_frame)]


class WaveClustSortingSegment(BaseSortingSegment):
    def __init__(self):
        BaseSortingSegment.__init__(self, unit_ids, spiketrains)
        self._unit_ids = list(unit_ids)
        self._spiketrains  = spiketrains

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        times = self._spiketrains[unit_id]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times




