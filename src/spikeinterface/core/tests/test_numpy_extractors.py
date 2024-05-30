import pytest
import numpy as np

from spikeinterface.core import (
    BaseRecording,
    NumpyRecording,
    SharedMemoryRecording,
    NumpySorting,
    SharedMemorySorting,
    NumpyEvent,
    create_sorting_npz,
    load_extractor,
    NpzSortingExtractor,
    generate_recording,
)

from spikeinterface.core.basesorting import minimum_spike_dtype


@pytest.fixture(scope="module")
def setup_NumpyRecording(tmp_path_factory):
    sampling_frequency = 30000
    timeseries_list = []
    for seg_index in range(3):
        traces = np.zeros((1000, 5), dtype="float64")
        timeseries_list.append(traces)

    rec = NumpyRecording(timeseries_list, sampling_frequency)
    # print(rec)

    times1 = rec.get_times(1)
    cache_folder = tmp_path_factory.mktemp("cache_folder")
    rec.save(folder=cache_folder / "test_NumpyRecording")
    return cache_folder


def test_SharedMemoryRecording():
    rec0 = generate_recording(num_channels=2, durations=[4.0, 3.0])
    # print(rec0)
    job_kwargs = dict(n_jobs=1, progress_bar=True)
    rec = SharedMemoryRecording.from_recording(rec0, **job_kwargs)

    d = rec.to_dict()
    rec_clone = load_extractor(d)
    traces = rec_clone.get_traces(start_frame=0, end_frame=30000, segment_index=0)

    assert rec.shms[0].name == rec_clone.shms[0].name

    del traces
    del rec_clone
    del rec


def test_NumpySorting(setup_NumpyRecording):
    sampling_frequency = 30000

    # empty
    unit_ids = []
    spikes = np.zeros(0, dtype=minimum_spike_dtype)
    sorting = NumpySorting(spikes, sampling_frequency, unit_ids)
    # print(sorting)

    # 2 columns
    times = np.arange(0, 1000, 10)
    labels = np.zeros(times.size, dtype="int64")
    labels[0::3] = 0
    labels[1::3] = 1
    labels[2::3] = 2
    sorting = NumpySorting.from_times_labels(times, labels, sampling_frequency)
    # print(sorting)
    assert sorting.get_num_segments() == 1

    sorting = NumpySorting.from_times_labels([times] * 3, [labels] * 3, sampling_frequency)
    # print(sorting)
    assert sorting.get_num_segments() == 3

    # from other extracrtor
    num_seg = 2

    cache_folder = setup_NumpyRecording

    file_path = cache_folder / "test_NpzSortingExtractor.npz"
    create_sorting_npz(num_seg, file_path)
    other_sorting = NpzSortingExtractor(file_path)

    sorting = NumpySorting.from_sorting(other_sorting)
    # print(sorting)

    # construct back from kwargs keep the same array
    sorting2 = load_extractor(sorting.to_dict())
    assert np.shares_memory(sorting2._cached_spike_vector, sorting._cached_spike_vector)


def test_SharedMemorySorting():
    sampling_frequency = 30000
    unit_ids = ["a", "b", "c"]
    spikes = np.zeros(100, dtype=minimum_spike_dtype)
    spikes["sample_index"][:] = np.arange(0, 1000, 10, dtype="int64")
    spikes["unit_index"][0::3] = 0
    spikes["unit_index"][1::3] = 1
    spikes["unit_index"][2::3] = 2
    np_sorting = NumpySorting(spikes, sampling_frequency, unit_ids)
    # print(np_sorting)

    sorting = SharedMemorySorting.from_sorting(np_sorting)
    # print(sorting)
    assert sorting._cached_spike_vector is not None

    # print(sorting.to_spike_vector())
    d = sorting.to_dict()

    sorting_reload = load_extractor(d)
    # print(sorting_reload)
    # print(sorting_reload.to_spike_vector())

    assert sorting.shm.name == sorting_reload.shm.name


def test_NumpyEvent():
    # one segment - dtype simple
    d = {
        "trig0": np.array([1, 10, 100]),
        "trig1": np.array([1, 50, 150]),
    }
    event = NumpyEvent.from_dict(d)

    times = event.get_events("trig0")
    assert times[2] == 100

    times = event.get_events("trig1")
    assert times[2] == 150

    # 2 segments - dtype simple
    event = NumpyEvent.from_dict([d, d])
    times = event.get_events("trig1", segment_index=1)
    assert times[2] == 150

    # 2 segments - dtype structured for one trig
    d = {
        "trig0": np.array([1, 10, 100]),
        "trig1": np.array([1, 50, 150]),
        "trig3": np.array([(1, 20), (50, 30), (150, 60)], dtype=[("time", "int64"), ("duration", "int64")]),
    }
    event = NumpyEvent.from_dict([d, d])
    times = event.get_events("trig1", segment_index=1)
    assert times[2] == 150

    times = event.get_events("trig3", segment_index=1)
    assert times.dtype.fields is not None
    assert times["time"][2] == 150
    assert times["duration"][2] == 60

    times = event.get_events("trig3", segment_index=1, end_time=100)
    assert times.size == 2


if __name__ == "__main__":
    # test_NumpyRecording()
    test_SharedMemoryRecording()
    # test_NumpySorting()
    # test_SharedMemorySorting()
    # test_NumpyEvent()
