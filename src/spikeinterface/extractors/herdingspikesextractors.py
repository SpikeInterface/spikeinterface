from __future__ import annotations

from pathlib import Path

import numpy as np

from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


class HerdingspikesSortingExtractor(BaseSorting):
    """Load HerdingSpikes format data as a sorting extractor.

    Parameters
    ----------
    file_path : str or Path
        Path to the ALF folder.
    load_unit_info : bool, default: True
        Whether to load the unit info from the file.

    Returns
    -------
    extractor : HerdingSpikesSortingExtractor
        The loaded data.
    """

    installation_mesg = "To use the HS2SortingExtractor install h5py: \n\n pip install h5py\n\n"

    def __init__(self, file_path, load_unit_info=True):
        try:
            import h5py
        except ImportError:
            raise ImportError(self.installation_mesg)

        self._recording_file = file_path
        self._rf = h5py.File(self._recording_file, mode="r")
        if "Sampling" in self._rf:
            if self._rf["Sampling"][()] == 0:
                sampling_frequency = None
            else:
                sampling_frequency = self._rf["Sampling"][()]

        spike_ids = self._rf["cluster_id"][()]
        unit_ids = np.unique(spike_ids)
        spike_times = self._rf["times"][()]

        if load_unit_info:
            self.load_unit_info()

        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        self.add_sorting_segment(HerdingspikesSortingSegment(unit_ids, spike_times, spike_ids))
        self._kwargs = {"file_path": str(Path(file_path).absolute()), "load_unit_info": load_unit_info}

        self.extra_requirements.append("h5py")

    def load_unit_info(self):
        # TODO
        """
        if 'centres' in self._rf.keys() and len(self._spike_times) > 0:
            self._unit_locs = self._rf['centres'][()]  # cache for faster access
            for u_i, unit_id in enumerate(self._unit_ids):
                self.set_unit_property(unit_id, property_name='unit_location', value=self._unit_locs[u_i])
        inds = []  # get these only once
        for unit_id in self._unit_ids:
            inds.append(np.where(self._cluster_id == unit_id)[0])
        if 'data' in self._rf.keys() and len(self._spike_times) > 0:
            d = self._rf['data'][()]
            for i, unit_id in enumerate(self._unit_ids):
                self.set_unit_spike_features(unit_id, 'spike_location', d[:, inds[i]].T)
        if 'ch' in self._rf.keys() and len(self._spike_times) > 0:
            d = self._rf['ch'][()]
            for i, unit_id in enumerate(self._unit_ids):
                self.set_unit_spike_features(unit_id, 'max_channel', d[inds[i]])
        """


# alias for backward compatiblity
HS2SortingExtractor = HerdingspikesSortingExtractor


class HerdingspikesSortingSegment(BaseSortingSegment):
    def __init__(self, unit_ids, spike_times, spike_ids):
        BaseSortingSegment.__init__(self)
        # spike_times is a dict
        self._unit_ids = list(unit_ids)
        self._spike_times = spike_times
        self._spike_ids = spike_ids

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        mask = self._spike_ids == unit_id
        times = self._spike_times[mask]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times

    """
    @staticmethod
    def write_sorting(sorting, save_path):
        assert HAVE_HS2SX, HS2SortingExtractor.installation_mesg
        unit_ids = sorting.get_unit_ids()
        times_list = []
        labels_list = []
        for i in range(len(unit_ids)):
            unit = unit_ids[i]
            times = sorting.get_unit_spike_train(unit_id=unit)
            times_list.append(times)
            labels_list.append(np.ones(times.shape, dtype=int) * unit)
        all_times = np.concatenate(times_list)
        all_labels = np.concatenate(labels_list)

        rf = h5py.File(save_path, mode='w')
        if sorting.get_sampling_frequency() is not None:
            rf.create_dataset("Sampling", data=sorting.get_sampling_frequency())
        else:
            rf.create_dataset("Sampling", data=0)
        if 'unit_location' in sorting.get_shared_unit_property_names():
            spike_centres = [sorting.get_unit_property(u, 'unit_location') for u in sorting.get_unit_ids()]
            spike_centres = np.array(spike_centres)
            rf.create_dataset("centres", data=spike_centres)
        if 'spike_location' in sorting.get_shared_unit_spike_feature_names():
            spike_loc_x = []
            spike_loc_y = []
            for u in sorting.get_unit_ids():
                l = sorting.get_unit_spike_features(u, 'spike_location')
                spike_loc_x.append(l[:, 0])
                spike_loc_y.append(l[:, 1])
            spike_loc = np.vstack((np.concatenate(spike_loc_x), np.concatenate(spike_loc_y)))
            rf.create_dataset("data", data=spike_loc)
        if 'max_channel' in sorting.get_shared_unit_spike_feature_names():
            spike_max_channel = np.concatenate(
                [sorting.get_unit_spike_features(u, 'max_channel') for u in sorting.get_unit_ids()])
            rf.create_dataset("ch", data=spike_max_channel)

        rf.create_dataset("times", data=all_times)
        rf.create_dataset("cluster_id", data=all_labels)
        rf.close()
    """


read_herdingspikes = define_function_from_class(source_class=HerdingspikesSortingExtractor, name="read_herdingspikes")
