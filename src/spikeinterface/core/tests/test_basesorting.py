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
    SharedMemorySorting,
    NpzFolderSorting,
    NumpyFolderSorting,
    create_sorting_npz,
    generate_sorting,
    load_extractor,
)
from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.testing import check_sorted_arrays_equal, check_sortings_equal
from spikeinterface.core.generate import generate_sorting


def test_BaseSorting(create_cache_folder):
    cache_folder = create_cache_folder
    num_seg = 2
    file_path = cache_folder / "test_BaseSorting.npz"
    file_path.parent.mkdir(exist_ok=True)

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

    # cache old format : npz_folder
    folder = cache_folder / "simple_sorting_npz_folder"
    sorting.set_property("test", np.ones(len(sorting.unit_ids)))
    sorting.save(folder=folder, format="npz_folder")
    sorting2 = BaseExtractor.load_from_folder(folder)
    assert isinstance(sorting2, NpzFolderSorting)

    # cache new format : numpy_folder
    folder = cache_folder / "simple_sorting_numpy_folder"
    sorting.set_property("test", np.ones(len(sorting.unit_ids)))
    sorting.save(folder=folder, format="numpy_folder")
    sorting2 = BaseExtractor.load_from_folder(folder)
    assert isinstance(sorting2, NumpyFolderSorting)

    # but also possible
    sorting3 = BaseExtractor.load(folder)
    check_sortings_equal(sorting, sorting2, check_annotations=True, check_properties=True)
    check_sortings_equal(sorting, sorting3, check_annotations=True, check_properties=True)

    # save to memory
    sorting4 = sorting.save(format="memory")
    check_sortings_equal(sorting, sorting4, check_annotations=True, check_properties=True)

    with pytest.warns(DeprecationWarning):
        num_spikes = sorting.get_all_spike_trains()
    # print(spikes)

    spikes = sorting.to_spike_vector()
    # print(spikes)
    assert sorting._cached_spike_vector is not None
    spikes = sorting.to_spike_vector(extremum_channel_inds={0: 15, 1: 5, 2: 18})
    # print(spikes)

    num_spikes_per_unit = sorting.count_num_spikes_per_unit(outputs="dict")
    num_spikes_per_unit = sorting.count_num_spikes_per_unit(outputs="array")
    total_spikes = sorting.count_total_num_spikes()

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

    sorting4 = sorting.to_numpy_sorting()
    sorting5 = sorting.to_multiprocessing(n_jobs=2)
    # create a clone with the same share mem buffer
    sorting6 = load_extractor(sorting5.to_dict())
    assert isinstance(sorting6, SharedMemorySorting)
    del sorting6
    del sorting5

    # test save to zarr
    # compressor = get_default_zarr_compressor()
    sorting_zarr = sorting.save(format="zarr", folder=cache_folder / "sorting")
    sorting_zarr_loaded = load_extractor(cache_folder / "sorting.zarr")
    # annotations is False because Zarr adds compression ratios
    check_sortings_equal(sorting, sorting_zarr, check_annotations=False, check_properties=True)
    check_sortings_equal(sorting_zarr, sorting_zarr_loaded, check_annotations=False, check_properties=True)
    for annotation_name in sorting.get_annotation_keys():
        assert sorting.get_annotation(annotation_name) == sorting_zarr.get_annotation(annotation_name)
        assert sorting.get_annotation(annotation_name) == sorting_zarr_loaded.get_annotation(annotation_name)


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
    sorting = NumpySorting.from_unit_dict(
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
    with pytest.warns(UserWarning):
        sorting.register_recording(rec)

    # Registering a rec with too many segments
    seg_nframes = [9, 5, 10]
    rec = NumpyRecording([np.zeros((nframes, 10)) for nframes in seg_nframes], sampling_frequency=sfreq)
    assert_raises(Exception, sorting.register_recording, rec)


def test_rename_units_method():
    num_units = 2
    durations = [1.0, 1.0]

    sorting = generate_sorting(num_units=num_units, durations=durations)

    new_unit_ids = ["a", "b"]
    new_sorting = sorting.rename_units(new_unit_ids=new_unit_ids)

    assert np.array_equal(new_sorting.get_unit_ids(), new_unit_ids)


def test_empty_sorting():
    sorting = NumpySorting.from_unit_dict({}, 30000)

    assert len(sorting.unit_ids) == 0

    with pytest.warns(DeprecationWarning):
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
