from __future__ import annotations

from pathlib import Path
import numpy as np

from .basesorting import BaseSorting, BaseSortingSegment
from .core_tools import define_function_from_class


class NpzSortingExtractor(BaseSorting):
    """
    Dead simple and super light format based on the NPZ numpy format.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html#numpy.savez

    It is in fact an archive of several .npy format.
    All spike are store in two columns maner index+labels
    """

    def __init__(self, file_path):
        self.npz_filename = file_path

        npz = np.load(file_path)
        num_segment = int(npz["num_segment"][0])
        unit_ids = npz["unit_ids"]
        sampling_frequency = float(npz["sampling_frequency"][0])

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        for seg_index in range(num_segment):
            spike_indexes = npz[f"spike_indexes_seg{seg_index}"]
            spike_labels = npz[f"spike_labels_seg{seg_index}"]
            sorting_segment = NpzSortingSegment(spike_indexes, spike_labels)
            self.add_sorting_segment(sorting_segment)

        self._kwargs = {"file_path": str(Path(file_path).absolute())}

    @staticmethod
    def write_sorting(sorting, save_path):
        d = {}
        units_ids = np.array(sorting.get_unit_ids())
        d["unit_ids"] = units_ids
        d["num_segment"] = np.array([sorting.get_num_segments()], dtype="int64")
        d["sampling_frequency"] = np.array([sorting.get_sampling_frequency()], dtype="float64")

        for seg_index in range(sorting.get_num_segments()):
            spike_indexes = []
            spike_labels = []
            for unit_id in units_ids:
                sp_ind = sorting.get_unit_spike_train(unit_id, segment_index=seg_index)
                spike_indexes.append(sp_ind.astype("int64"))
                # spike_labels.append(np.ones(sp_ind.size, dtype='int64')*unit_id)
                spike_labels.append(np.array([unit_id] * sp_ind.size))

            # order times
            if len(spike_indexes) > 0:
                spike_indexes = np.concatenate(spike_indexes)
                spike_labels = np.concatenate(spike_labels)
                order = np.argsort(spike_indexes)
                spike_indexes = spike_indexes[order]
                spike_labels = spike_labels[order]
            else:
                spike_indexes = np.array([], dtype="int64")
                spike_labels = np.array([], dtype="int64")
            d[f"spike_indexes_seg{seg_index}"] = spike_indexes
            d[f"spike_labels_seg{seg_index}"] = spike_labels

        np.savez(save_path, **d)


class NpzSortingSegment(BaseSortingSegment):
    def __init__(self, spike_indexes, spike_labels):
        BaseSortingSegment.__init__(self)

        self.spike_indexes = spike_indexes
        self.spike_labels = spike_labels

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        spike_times = self.spike_indexes[self.spike_labels == unit_id]
        if start_frame is not None:
            spike_times = spike_times[spike_times >= start_frame]
        if end_frame is not None:
            spike_times = spike_times[spike_times < end_frame]
        return spike_times.astype("int64")


read_npz_sorting = define_function_from_class(source_class=NpzSortingExtractor, name="read_npz_sorting")
