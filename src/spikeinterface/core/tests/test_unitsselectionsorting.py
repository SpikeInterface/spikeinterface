import pytest
import numpy as np
from pathlib import Path

from spikeinterface.core import UnitsSelectionSorting

from spikeinterface.core import NpzSortingExtractor, load_extractor
from spikeinterface.core.base import BaseExtractor

from spikeinterface.core import create_sorting_npz


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_unitsselectionsorting():
    num_seg = 2
    file_path = cache_folder / 'test_BaseSorting.npz'

    create_sorting_npz(num_seg, file_path)

    sorting = NpzSortingExtractor(file_path)
    print(sorting)
    print(sorting.unit_ids)

    sorting2 = UnitsSelectionSorting(sorting, unit_ids=[0, 2])
    print(sorting2.unit_ids)
    assert np.array_equal(sorting2.unit_ids, [0, 2])

    sorting3 = UnitsSelectionSorting(sorting, unit_ids=[0, 2], renamed_unit_ids=['a', 'b'])
    print(sorting3.unit_ids)
    assert np.array_equal(sorting3.unit_ids, ['a', 'b'])

    assert np.array_equal(sorting.get_unit_spike_train(0, segment_index=0),
                          sorting2.get_unit_spike_train(0, segment_index=0))
    assert np.array_equal(sorting.get_unit_spike_train(0, segment_index=0),
                          sorting3.get_unit_spike_train('a', segment_index=0))

    assert np.array_equal(sorting.get_unit_spike_train(2, segment_index=0),
                          sorting2.get_unit_spike_train(2, segment_index=0))
    assert np.array_equal(sorting.get_unit_spike_train(2, segment_index=0),
                          sorting3.get_unit_spike_train('b', segment_index=0))


if __name__ == '__main__':
    test_unitsselectionsorting()
