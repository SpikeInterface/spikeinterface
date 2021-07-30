import pytest
import numpy as np

from spikeinterface.core import NpzSortingExtractor
from spikeinterface.core.tests.testing_tools import create_sorting_npz


def test_NpzSortingExtractor():
    num_seg = 2
    file_path = 'test_NpzSortingExtractor.npz'

    create_sorting_npz(num_seg, file_path)

    sorting = NpzSortingExtractor(file_path)

    for segment_index in range(num_seg):
        for unit_id in (0, 1, 2):
            st = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)

    file_path_copy = 'test_NpzSortingExtractor_copy.npz'
    NpzSortingExtractor.write_sorting(sorting, file_path_copy)
    sorting_copy = NpzSortingExtractor(file_path_copy)


if __name__ == '__main__':
    test_NpzSortingExtractor()
