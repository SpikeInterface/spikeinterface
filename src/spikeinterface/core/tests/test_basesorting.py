"""
test for BaseSorting are done with NpzSortingExtractor.
but check only for BaseRecording general methods.
"""

import importlib.util
import time
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
    generate_ground_truth_recording,
    generate_recording,
    generate_sorting,
    create_sorting_npz,
    generate_sorting,
    load,
)
from spikeinterface.core.base import BaseExtractor, minimum_spike_dtype, unit_period_dtype
from spikeinterface.core.basesorting import LEXSORT_UNIT_COMPACT
from spikeinterface.core.testing import check_sorted_arrays_equal, check_sortings_equal


def _make_sorting_with_shuffled_ties(num_units, num_segments, seed=42):
    """Build a NumpySorting whose cotemporal spikes are in arbitrary unit_index order.

    A spike vector is only guaranteed to be segment-blocked and sample_index-ascending within each
    segment; the unit_index order among spikes sharing a sample_index is unspecified (see #4606).
    Building via `NumpySorting.from_unit_dict` happens to produce unit-ascending ties, so it
    can't test the shuffled tie case.
    """
    rng = np.random.default_rng(seed)
    num_spikes = 2_000

    # A sample range far smaller than num_spikes, so cotemporal spikes are abundant -- including
    # repeats of the same (segment, sample, unit), the tie that np.lexsort itself cannot break.
    spikes = np.empty(num_spikes, dtype=minimum_spike_dtype)
    spikes["sample_index"] = rng.integers(0, 200, size=num_spikes)
    spikes["unit_index"] = rng.integers(0, num_units, size=num_spikes)
    spikes["segment_index"] = rng.integers(0, num_segments, size=num_spikes)

    # Order by segment then sample, breaking ties randomly rather than by unit_index.
    spikes = spikes[np.lexsort((rng.random(num_spikes), spikes["sample_index"], spikes["segment_index"]))]

    sorting = NumpySorting(spikes, 30_000.0, np.arange(num_units))
    assert sorting.get_num_segments() == num_segments
    return sorting


def test_BaseSorting(create_cache_folder):
    num_seg = 2
    file_path = create_cache_folder / "test_BaseSorting.npz"
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
    sorting3 = load(d)
    check_sortings_equal(sorting, sorting2, check_annotations=True, check_properties=True)
    check_sortings_equal(sorting, sorting3, check_annotations=True, check_properties=True)

    # dump/load json
    sorting.dump_to_json(create_cache_folder / "test_BaseSorting.json")
    sorting2 = load(create_cache_folder / "test_BaseSorting.json")
    sorting3 = load(create_cache_folder / "test_BaseSorting.json")
    check_sortings_equal(sorting, sorting2, check_annotations=True, check_properties=False)
    check_sortings_equal(sorting, sorting3, check_annotations=True, check_properties=False)

    # dump/load pickle
    sorting.dump_to_pickle(create_cache_folder / "test_BaseSorting.pkl")
    sorting2 = load(create_cache_folder / "test_BaseSorting.pkl")
    sorting3 = load(create_cache_folder / "test_BaseSorting.pkl")
    check_sortings_equal(sorting, sorting2, check_annotations=True, check_properties=True)
    check_sortings_equal(sorting, sorting3, check_annotations=True, check_properties=True)

    # cache old format : npz_folder
    folder = create_cache_folder / "simple_sorting_npz_folder"
    sorting.set_property("test", np.ones(len(sorting.unit_ids)))
    sorting.save(folder=folder, format="npz_folder")
    sorting2 = load(folder)
    assert isinstance(sorting2, NpzFolderSorting)

    # cache new format : binary
    folder = create_cache_folder / "simple_sorting_binary"
    sorting.set_property("test", np.ones(len(sorting.unit_ids)))
    sorting.save(folder=folder, format="numpy_folder")
    sorting2 = load(folder)
    assert isinstance(sorting2, NumpyFolderSorting)

    # but also possible
    sorting3 = load(folder)
    check_sortings_equal(sorting, sorting2, check_annotations=True, check_properties=True)
    check_sortings_equal(sorting, sorting3, check_annotations=True, check_properties=True)

    # save to memory
    sorting4 = sorting.save(format="memory")
    check_sortings_equal(sorting, sorting4, check_annotations=True, check_properties=True)

    # print(spikes)

    spikes = sorting.to_spike_vector()
    # print(spikes)
    assert sorting._cached_time_ordered is not None

    spikes = sorting.to_spike_vector()
    ordered_spikes, order, slices = sorting.to_reordered_spike_vector(
        lexsort=("sample_index", "segment_index", "unit_index"),
        return_order=True,
        return_slices=True,
    )
    assert np.array_equal(spikes[order], ordered_spikes)

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
    sorting6 = load(sorting5.to_dict())
    assert isinstance(sorting6, SharedMemorySorting)
    del sorting6
    del sorting5

    # test save to zarr
    # compressor = get_default_zarr_compressor()
    sorting_zarr = sorting.save(format="zarr", folder=create_cache_folder / "sorting.zarr")
    sorting_zarr_loaded = load(create_cache_folder / "sorting.zarr")
    # annotations is False because Zarr adds compression ratios
    check_sortings_equal(sorting, sorting_zarr, check_annotations=False, check_properties=True)
    check_sortings_equal(sorting_zarr, sorting_zarr_loaded, check_annotations=False, check_properties=True)
    for annotation_name in sorting.get_annotation_keys():
        assert sorting.get_annotation(annotation_name) == sorting_zarr.get_annotation(annotation_name)
        assert sorting.get_annotation(annotation_name) == sorting_zarr_loaded.get_annotation(annotation_name)


def test_zarr_save_with_sharding(create_cache_folder):
    sorting = generate_sorting(durations=[600, 600])
    # 1 MB target chunk size for Zarr
    zarr_target_bytes = 1 * 1024 * 1024
    shard_factor = 3
    sorting_zarr = sorting.save(
        format="zarr",
        folder=create_cache_folder / "sorting_sharded.zarr",
        target_chunk_size_bytes=zarr_target_bytes,
        shard_factor=shard_factor,
    )
    sorting_zarr_loaded = load(create_cache_folder / "sorting_sharded.zarr")
    check_sortings_equal(sorting, sorting_zarr, check_annotations=False, check_properties=True)
    check_sortings_equal(sorting_zarr, sorting_zarr_loaded, check_annotations=False, check_properties=True)

    # check that chunks and shards are correctly set
    spikes_group = sorting_zarr._root["spikes"]
    for field in spikes_group:
        if field not in ("segment_slices", "sample_index_chunk_firsts"):
            array = spikes_group[field]
            assert array.chunks is not None
            assert array.shards is not None
            assert array.shards[0] == array.chunks[0] * shard_factor
            assert array.chunks[0] * array.dtype.itemsize <= zarr_target_bytes
            print(f"Field: {field}, Chunks: {array.chunks}, Shards: {array.shards}")
            print(
                f"Field: {field}, Chunk size in bytes: {array.chunks[0] * array.dtype.itemsize} - Target: {zarr_target_bytes}"
            )


@pytest.mark.parametrize("use_numba", [True, False], ids=["numba", "numpy"])
def test_to_reordered_spike_vector(use_numba, monkeypatch):
    """`to_reordered_spike_vector` should group spikes by (unit, segment) without disturbing them."""
    if use_numba and importlib.util.find_spec("numba") is None:
        pytest.skip("numba not installed")
    monkeypatch.setattr("spikeinterface.core.sorting_tools.HAVE_NUMBA", use_numba)

    num_units, num_segments = 6, 3
    sorting = _make_sorting_with_shuffled_ties(num_units, num_segments)
    spikes = sorting.to_spike_vector()

    # Make sure the input genuinely violates the old full lexsort
    assert not np.array_equal(
        spikes, spikes[np.lexsort((spikes["unit_index"], spikes["sample_index"], spikes["segment_index"]))]
    )

    ordered_spikes, order, slices = sorting.to_reordered_spike_vector(
        lexsort=LEXSORT_UNIT_COMPACT, return_order=True, return_slices=True
    )

    # The buckets, unit by unit and segment by segment within each unit.
    assert slices.shape == (num_units, num_segments, 2)
    groups = [(u, s) for u in range(num_units) for s in range(num_segments)]
    masks = [(spikes["unit_index"] == u) & (spikes["segment_index"] == s) for u, s in groups]

    assert np.array_equal(ordered_spikes, np.concatenate([spikes[mask] for mask in masks]))

    # `order` must reproduce the reordering.
    assert np.array_equal(spikes[order], ordered_spikes)

    # `slices` must delimit each bucket, and together tile the whole vector.
    stops = np.cumsum([mask.sum() for mask in masks])
    expected_slices = np.stack([stops - [mask.sum() for mask in masks], stops], axis=-1)
    assert np.array_equal(slices.reshape(-1, 2), expected_slices)
    assert slices.reshape(-1, 2)[0, 0] == 0 and slices.reshape(-1, 2)[-1, 1] == spikes.size


@pytest.mark.parametrize(
    "unit_dict, num_units",
    [({"0": np.array([], dtype="int64")}, 1), ({}, 0)],
    ids=["unit_with_no_spikes", "no_units"],
)
def test_to_reordered_spike_vector_empty(unit_dict, num_units):
    """Empty sortings must round-trip.

    A sorting with *no units at all* is a valid degenerate case, not an error
    (see `test_empty_sorting`). It simply has no buckets.
    """
    sorting = NumpySorting.from_unit_dict(unit_dict, 30_000.0)
    assert len(sorting.unit_ids) == num_units

    ordered_spikes, order, slices = sorting.to_reordered_spike_vector(
        lexsort=("sample_index", "segment_index", "unit_index"),
        return_order=True,
        return_slices=True,
    )
    assert ordered_spikes.size == 0
    assert order.size == 0
    assert np.array_equal(slices, np.zeros((num_units, 1, 2), dtype=np.int64))

    # The methods that build the reordering internally must survive it too.
    sorting.precompute_spike_trains()
    assert len(sorting.count_num_spikes_per_unit(outputs="dict")) == num_units


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

    spikes = sorting.to_spike_vector()
    assert spikes.shape == (0,)


def test_time_slice():

    sampling_frequency = 10_000.0

    # no recording attached to sorting
    sorting = generate_sorting(durations=[1.0], sampling_frequency=sampling_frequency)

    sliced_sorting_times = sorting.time_slice(start_time=0.1, end_time=0.8)
    sliced_sorting_frames = sorting.frame_slice(start_frame=1000, end_frame=8000)

    assert np.allclose(
        sliced_sorting_times.to_spike_vector()["sample_index"], sliced_sorting_frames.to_spike_vector()["sample_index"]
    )

    # with recording
    recording, sorting = generate_ground_truth_recording(durations=[1.0], sampling_frequency=sampling_frequency)
    sorting.register_recording(recording)

    sliced_sorting_times = sorting.time_slice(start_time=0.1, end_time=0.8)
    sliced_sorting_frames = sorting.frame_slice(start_frame=1000, end_frame=8000)

    assert np.allclose(
        sliced_sorting_times.to_spike_vector()["sample_index"], sliced_sorting_frames.to_spike_vector()["sample_index"]
    )


def test_select_periods():
    sampling_frequency = 10_000.0
    duration = 100
    num_samples = int(sampling_frequency * duration)
    num_units = 1000
    sorting = generate_sorting(
        durations=[duration, duration], sampling_frequency=sampling_frequency, num_units=num_units
    )

    rng = np.random.default_rng()

    # number of random periods
    n_periods = 1_000
    # generate random periods
    segment_indices = rng.integers(0, sorting.get_num_segments(), n_periods)
    start_samples = rng.integers(0, num_samples, n_periods)
    durations = rng.integers(100, 100_000, n_periods)
    end_samples = start_samples + durations
    valid_periods = end_samples < num_samples
    segment_indices = segment_indices[valid_periods]
    start_samples = start_samples[valid_periods]
    end_samples = end_samples[valid_periods]
    unit_index = rng.integers(0, num_units - 1, len(segment_indices))

    periods = np.zeros(len(segment_indices), dtype=unit_period_dtype)
    periods["segment_index"] = segment_indices
    periods["start_sample_index"] = start_samples
    periods["end_sample_index"] = end_samples
    periods["unit_index"] = unit_index
    periods = np.sort(periods, order=["segment_index", "start_sample_index"])

    t_start = time.perf_counter()
    sliced_sorting = sorting.select_periods(periods=periods)
    t_stop = time.perf_counter()
    elapsed = t_stop - t_start
    print(f"select_periods took {elapsed:.2f} seconds for {len(periods)} periods")

    # Check that all spikes in the sliced sorting are within the periods
    for segment_index in range(sorting.get_num_segments()):
        periods_in_segment = periods[periods["segment_index"] == segment_index]
        for unit_index, unit_id in enumerate(sorting.unit_ids):
            spiketrain = sorting.get_unit_spike_train(segment_index=segment_index, unit_id=unit_id)

            periods_for_unit = periods_in_segment[periods_in_segment["unit_index"] == unit_index]
            spiketrain_in_periods = []
            for period in periods_for_unit:
                start_sample = period["start_sample_index"]
                end_sample = period["end_sample_index"]
                spiketrain_in_periods.append(spiketrain[(spiketrain >= start_sample) & (spiketrain < end_sample)])
            if len(spiketrain_in_periods) == 0:
                spiketrain_in_periods = np.array([], dtype=spiketrain.dtype)
            else:
                spiketrain_in_periods = np.unique(np.concatenate(spiketrain_in_periods))

            spiketrain_sliced = sliced_sorting.get_unit_spike_train(segment_index=segment_index, unit_id=unit_id)
            assert len(spiketrain_in_periods) == len(spiketrain_sliced)

    # now test with input as numpy array with shape (n_periods, 4)
    periods_array = np.zeros((len(periods), 4), dtype="int64")
    periods_array[:, 0] = periods["segment_index"]
    periods_array[:, 1] = periods["start_sample_index"]
    periods_array[:, 2] = periods["end_sample_index"]
    periods_array[:, 3] = periods["unit_index"]

    sliced_sorting_array = sorting.select_periods(periods=periods_array)
    np.testing.assert_array_equal(sliced_sorting.to_spike_vector(), sliced_sorting_array.to_spike_vector())


def _make_sorting_for_select_periods(register_recording=True):
    """Two-segment sorting, with a property, an annotation and a recording that does not start at t = 0."""
    sampling_frequency = 1000.0
    sorting = NumpySorting.from_unit_dict(
        [
            {"a": np.array([10, 2000, 5000]), "b": np.array([30, 3000])},
            {"a": np.array([100, 7000]), "b": np.array([40])},
        ],
        sampling_frequency=sampling_frequency,
    )
    sorting.set_property("quality", np.array(["good", "mua"]))
    sorting.annotate(probe_name="NP2.0")
    if register_recording:
        recording = generate_recording(
            num_channels=2, sampling_frequency=sampling_frequency, durations=[10.0, 10.0], seed=0
        )
        recording.shift_times(500.0, segment_index=0)
        recording.shift_times(900.0, segment_index=1)
        sorting.register_recording(recording)
    return sorting


def _spike_times_of_selection(sorting, segment_index):
    """Spike times of unit "a" after selection, which keeps samples [10, 2000] and [100, 7000] of segments 0 and 1."""
    periods = np.array([(0, 0, 4000, 0), (1, 0, 8000, 0)], dtype=unit_period_dtype)
    selected = sorting.select_periods(periods)
    return selected.get_unit_spike_train("a", segment_index=segment_index, return_times=True)


def test_select_periods_keeps_metadata():
    sorting = _make_sorting_for_select_periods()
    periods = np.array([(0, 0, 4000, 0), (0, 0, 4000, 1), (1, 0, 8000, 0)], dtype=unit_period_dtype)

    selected = sorting.select_periods(periods)

    assert selected is not sorting
    np.testing.assert_array_equal(selected.unit_ids, sorting.unit_ids)
    np.testing.assert_array_equal(selected.get_property("quality"), ["good", "mua"])
    assert selected.get_annotation("probe_name") == "NP2.0"
    assert selected.has_recording()
    assert selected._recording is sorting._recording


def test_select_periods_keeps_times():
    # start times of the recording
    sorting = _make_sorting_for_select_periods()
    np.testing.assert_allclose(_spike_times_of_selection(sorting, 0), [500.01, 502.0])
    np.testing.assert_allclose(_spike_times_of_selection(sorting, 1), [900.1, 907.0])

    # shift of the sorting, on top of the times of the recording
    sorting.shift_times(2.0, segment_index=1)
    np.testing.assert_allclose(_spike_times_of_selection(sorting, 0), [500.01, 502.0])
    np.testing.assert_allclose(_spike_times_of_selection(sorting, 1), [902.1, 909.0])

    # time vector of the recording: irregular, so that a start time cannot stand in for it
    sorting = _make_sorting_for_select_periods()
    times = 100.0 + (np.arange(10_000) / 1000.0) ** 2
    recording = sorting._recording
    recording.set_times(times, segment_index=0)
    sorting.register_recording(recording)
    np.testing.assert_allclose(_spike_times_of_selection(sorting, 0), times[[10, 2000]])

    # shift of the sorting, without any recording
    sorting = _make_sorting_for_select_periods(register_recording=False)
    sorting.shift_times(7.0, segment_index=0)
    np.testing.assert_allclose(_spike_times_of_selection(sorting, 0), [7.01, 9.0])
    np.testing.assert_allclose(_spike_times_of_selection(sorting, 1), [0.1, 7.0])


@pytest.mark.parametrize(
    "periods, expected_num_spikes",
    [
        pytest.param([(0, 0, 4000, 0), (0, 0, 4000, 1)], 4, id="no_period_in_last_segment"),
        pytest.param([(0, 0, 1, 0)], 0, id="no_spike_in_periods"),
        pytest.param([], 0, id="no_period"),
    ],
)
def test_select_periods_keeps_segments_without_spikes(periods, expected_num_spikes):
    sorting = _make_sorting_for_select_periods()

    selected = sorting.select_periods(np.array(periods, dtype=unit_period_dtype))

    assert selected.to_spike_vector().size == expected_num_spikes
    assert selected.get_num_segments() == sorting.get_num_segments()
    assert selected.has_recording()
    assert selected.get_start_time(segment_index=1) == 900.0
    assert selected.get_unit_spike_train("a", segment_index=1).size == 0
    assert selected.get_unit_spike_train("a", segment_index=1, return_times=True).size == 0


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdirname:
        create_cache_folder = Path(tmpdirname)

    test_BaseSorting(create_cache_folder)
    test_npy_sorting()
    test_empty_sorting()
    test_select_periods()
    test_select_periods_keeps_metadata()
    test_select_periods_keeps_times()
