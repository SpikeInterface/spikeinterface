import pytest
import numpy as np

from spikeinterface.core import NpzSortingExtractor


def test_NpzSortingExtractor():
    num_seg = 2
    file_path = 'test_NpzSortingExtractor.npz'
    
    # create a NPZ sorting file
    d = {}
    d['unit_ids'] = np.array([0,1,2], dtype='int64')
    d['num_segment'] = np.array([2], dtype='int64')
    d['sampling_frequency'] = np.array([30000.], dtype='float64')
    for seg_index in range(num_seg):
        spike_indexes = np.arange(0, 1000, 10)
        spike_labels = np.zeros(spike_indexes.size, dtype='int64')
        spike_labels[0::3] = 0
        spike_labels[1::3] = 1
        spike_labels[2::3] = 2
        d[f'spike_indexes_seg{seg_index}'] = spike_indexes
        d[f'spike_labels_seg{seg_index}'] = spike_labels
    np.savez(file_path, **d)
    
    
    sorting = NpzSortingExtractor(file_path)
    
    for segment_index in range(num_seg):
        for unit_id in (0, 1, 2):
            st = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
    
    file_path_copy = 'test_NpzSortingExtractor_copy.npz'
    NpzSortingExtractor.write_sorting(sorting,  file_path_copy)
    sorting_copy = NpzSortingExtractor(file_path_copy)
    

if __name__ == '__main__':
    test_NpzSortingExtractor()


