"""
test for BaseSorting are done with NpzSortingExtractor.
but check only for BaseRecording general methods.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_raises

from spikeinterface.core import (
    NpzSortingExtractor,
    NumpyRecording,
    NumpySorting,
    create_sorting_npz,
    generate_sorting,
    load_extractor,
)
from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.testing import check_sorted_arrays_equal, check_sortings_equal

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_BaseSorting():
    num_seg = 2
    file_path = cache_folder / "test_BaseSorting.npz"

    create_sorting_npz(num_seg, file_path)

    sorting = NpzSortingExtractor(file_path)
    # print(sorting)

    assert sorting.get_num_segments() == 2
    assert sorting.get_num_units() == 3

    # annotations / properties
    sorting.annotate(yep="yop")
    assert sorting.get_annotation("yep") == "yop"

    sorting.set_property("amplitude", [-20, -40.0, -55.5])
    values = sorting.get_property("amplitude")
    assert np.all(values == [-20, -40.0, -55.5])

    # dump/load dict
    d = sorting.to_dict(include_annotations=True, include_properties=True)
    sorting2 = BaseExtractor.from_dict(d)
    sorting3 = load_extractor(d)
    check_sortings_equal(sorting, sorting2, check_annotations=True, check_properties=True)
    check_sortings_equal(sorting, sorting3, check_annotations=True, check_properties=True)

    # dump/load json
    sorting.dump_to_json(cache_folder / "test_BaseSorting.json")
    sorting2 = BaseExtractor.load(cache_folder / "test_BaseSorting.json")
    sorting3 = load_extractor(cache_folder / "test_BaseSorting.json")
    check_sortings_equal(sorting, sorting2, check_annotations=True, check_properties=False)
    check_sortings_equal(sorting, sorting3, check_annotations=True, check_properties=False)

    # dump/load pickle
    sorting.dump_to_pickle(cache_folder / "test_BaseSorting.pkl")
    sorting2 = BaseExtractor.load(cache_folder / "test_BaseSorting.pkl")
    sorting3 = load_extractor(cache_folder / "test_BaseSorting.pkl")
    check_sortings_equal(sorting, sorting2, check_annotations=True, check_properties=True)
    check_sortings_equal(sorting, sorting3, check_annotations=True, check_properties=True)

    # cache
    folder = cache_folder / "simple_sorting"
    sorting.set_property("test", np.ones(len(sorting.unit_ids)))
    sorting.save(folder=folder)
    sorting2 = BaseExtractor.load_from_folder(folder)
    # but also possible
    sorting3 = BaseExtractor.load(folder)
    check_sortings_equal(sorting, sorting2, check_annotations=True, check_properties=True)
    check_sortings_equal(sorting, sorting3, check_annotations=True, check_properties=True)

    # save to memory
    sorting4 = sorting.save(format="memory")
    check_sortings_equal(sorting, sorting4, check_annotations=True, check_properties=True)

    spikes = sorting.get_all_spike_trains()
    # print(spikes)

    spikes = sorting.to_spike_vector()
    # print(spikes)
    spikes = sorting.to_spike_vector(extremum_channel_inds={0: 15, 1: 5, 2: 18})
    # print(spikes)

    # select units
    keep_units = [0, 1]
    sorting_select = sorting.select_units(unit_ids=keep_units)
    for unit in sorting_select.get_unit_ids():
        assert unit in keep_units

    # remove empty units
    empty_units = [1, 3]
    sorting_empty = generate_sorting(empty_units=empty_units)
    sorting_clean = sorting_empty.remove_empty_units()
    for unit in sorting_clean.get_unit_ids():
        assert unit not in empty_units


def test_npy_sorting():
    sfreq = 10
    spike_times_0 = {
        "0": np.array([0, 1, 9]),  # Max sample idx is 9 for a rec of length 10
        "1": np.array([2, 5]),
    }
    spike_times_1 = {
        "0": np.array([0, 1]),
        "1": np.array([], dtype="int64"),
    }
    sorting = NumpySorting.from_dict(
        [spike_times_0, spike_times_1],
        sfreq,
    )

    assert sorting.get_num_segments() == 2
    assert set(sorting.get_unit_ids()) == set(["0", "1"])
    check_sorted_arrays_equal(sorting.get_unit_spike_train(segment_index=0, unit_id="1"), [2, 5])

    # Check registering a recording
    seg_nframes = [10, 5]
    rec = NumpyRecording([np.zeros((nframes, 10)) for nframes in seg_nframes], sampling_frequency=sfreq)
    sorting.register_recording(rec)
    assert sorting.get_num_samples(segment_index=0) == 10
    assert sorting.get_num_samples(segment_index=1) == 5
    assert sorting.get_total_samples() == 15

    # Registering too short a recording raises a warning
    seg_nframes = [9, 5]
    rec = NumpyRecording([np.zeros((nframes, 10)) for nframes in seg_nframes], sampling_frequency=sfreq)
    # assert_raises(Exception, sorting.register_recording, rec)
    with pytest.warns():
        sorting.register_recording(rec)

    # Registering a rec with too many segments
    seg_nframes = [9, 5, 10]
    rec = NumpyRecording([np.zeros((nframes, 10)) for nframes in seg_nframes], sampling_frequency=sfreq)
    assert_raises(Exception, sorting.register_recording, rec)


def test_empty_sorting():
    sorting = NumpySorting.from_dict({}, 30000)

    assert len(sorting.unit_ids) == 0

    spikes = sorting.get_all_spike_trains()
    assert len(spikes) == 1
    assert len(spikes[0][0]) == 0
    assert len(spikes[0][1]) == 0

    spikes = sorting.to_spike_vector()
    assert spikes.shape == (0,)


if __name__ == "__main__":
    test_BaseSorting()
    test_npy_sorting()
    test_empty_sorting()
