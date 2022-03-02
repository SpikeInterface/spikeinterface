"""
test for BaseSorting are done with NpzSortingExtractor.
but check only for BaseRecording general methods.
"""
import shutil
from pathlib import Path
import pytest
import numpy as np

from spikeinterface.core import NpzSortingExtractor, load_extractor
from spikeinterface.core.base import BaseExtractor

from spikeinterface.core.testing_tools import create_sorting_npz

if getattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_BaseSorting():
    num_seg = 2
    file_path = cache_folder / 'test_BaseSorting.npz'

    create_sorting_npz(num_seg, file_path)

    sorting = NpzSortingExtractor(file_path)
    print(sorting)

    assert sorting.get_num_segments() == 2
    assert sorting.get_num_units() == 3

    # annotations / properties
    sorting.annotate(yep='yop')
    assert sorting.get_annotation('yep') == 'yop'

    sorting.set_property('amplitude', [-20, -40., -55.5])
    values = sorting.get_property('amplitude')
    assert np.all(values == [-20, -40., -55.5])

    # dump/load dict
    d = sorting.to_dict()
    sorting2 = BaseExtractor.from_dict(d)
    sorting3 = load_extractor(d)

    # dump/load json
    sorting.dump_to_json(cache_folder / 'test_BaseSorting.json')
    sorting2 = BaseExtractor.load(cache_folder / 'test_BaseSorting.json')
    sorting3 = load_extractor(cache_folder / 'test_BaseSorting.json')

    # dump/load pickle
    sorting.dump_to_pickle(cache_folder / 'test_BaseSorting.pkl')
    sorting2 = BaseExtractor.load(cache_folder / 'test_BaseSorting.pkl')
    sorting3 = load_extractor(cache_folder / 'test_BaseSorting.pkl')

    # cache
    folder = cache_folder / 'simple_sorting'
    sorting.save(folder=folder)
    sorting2 = BaseExtractor.load_from_folder(folder)
    # but also possible
    sorting3 = BaseExtractor.load(folder)

    spikes = sorting.get_all_spike_trains()
    # print(spikes)

    spikes = sorting.to_spike_vector()
    # print(spikes)


if __name__ == '__main__':
    test_BaseSorting()
