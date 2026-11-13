import numpy as np
import spikeinterface.core.recording_tools as recording_tools

from spikeinterface.core import NumpyRecording, generate_recording, MockRecording

from spikeinterface.core.binaryrecordingextractor import BinaryRecordingExtractor


from spikeinterface.core.recording_tools import (
    write_binary_recording,
    write_memory_recording,
    get_random_data_chunks,
    get_chunk_with_margin,
    get_closest_channels,
    get_noise_levels,
    order_channels_by_depth,
    do_recording_attributes_match,
    get_rec_attributes,
)


def test_write_binary_recording(tmp_path):
    # Test write_binary_recording() with loop (n_jobs=1)
    # Setup
    sampling_frequency = 30_000
    num_channels = 2
    dtype = "float32"

    durations = [10.0]
    recording = MockRecording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        strategy="tile_pregenerated",
    )
    file_paths = [tmp_path / "binary01.raw"]

    # Write binary recording
    job_kwargs = dict(n_jobs=1)
    write_binary_recording(recording, file_paths=file_paths, dtype=dtype, verbose=False, **job_kwargs)

    # Check if written data matches original data
    recorder_binary = BinaryRecordingExtractor(
        file_paths=file_paths, sampling_frequency=sampling_frequency, num_channels=num_channels, dtype=dtype
    )
    assert np.allclose(recorder_binary.get_traces(), recording.get_traces())


def test_write_binary_recording_offset(tmp_path):
    # Test write_binary_recording() with loop (n_jobs=1)
    # Setup
    sampling_frequency = 30_000
    num_channels = 2
    dtype = "float32"

    durations = [10.0]
    recording = MockRecording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        strategy="tile_pregenerated",
    )
    file_paths = [tmp_path / "binary01.raw"]

    # Write binary recording
    job_kwargs = dict(n_jobs=1)
    byte_offset = 125
    write_binary_recording(
        recording, file_paths=file_paths, dtype=dtype, byte_offset=byte_offset, verbose=False, **job_kwargs
    )

    # Check if written data matches original data
    recorder_binary = BinaryRecordingExtractor(
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
        file_offset=byte_offset,
    )
    assert np.allclose(recorder_binary.get_traces(), recording.get_traces())


def test_write_binary_recording_parallel(tmp_path):
    # Test write_binary_recording() with parallel processing (n_jobs=2)

    # Setup
    sampling_frequency = 30_000
    num_channels = 2
    dtype = "float32"
    durations = [10.30, 3.5]
    recording = MockRecording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        dtype=dtype,
        strategy="tile_pregenerated",
    )
    file_paths = [tmp_path / "binary01.raw", tmp_path / "binary02.raw"]

    # Write binary recording
    job_kwargs = dict(n_jobs=2, chunk_memory="100k", mp_context="spawn")
    write_binary_recording(recording, file_paths=file_paths, dtype=dtype, verbose=False, **job_kwargs)

    # Check if written data matches original data
    recorder_binary = BinaryRecordingExtractor(
        file_paths=file_paths, sampling_frequency=sampling_frequency, num_channels=num_channels, dtype=dtype
    )
    for segment_index in range(recording.get_num_segments()):
        binary_traces = recorder_binary.get_traces(segment_index=segment_index)
        recording_traces = recording.get_traces(segment_index=segment_index)
        assert np.allclose(binary_traces, recording_traces)


def test_write_binary_recording_multiple_segment(tmp_path):
    # Test write_binary_recording() with multiple segments (n_jobs=2)
    # Setup
    sampling_frequency = 30_000
    num_channels = 10
    dtype = "float32"

    durations = [10.30, 3.5]
    recording = MockRecording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        strategy="tile_pregenerated",
    )
    file_paths = [tmp_path / "binary01.raw", tmp_path / "binary02.raw"]

    # Write binary recording
    job_kwargs = dict(n_jobs=2, chunk_memory="100k", mp_context="spawn")
    write_binary_recording(recording, file_paths=file_paths, dtype=dtype, verbose=False, **job_kwargs)

    # Check if written data matches original data
    recorder_binary = BinaryRecordingExtractor(
        file_paths=file_paths, sampling_frequency=sampling_frequency, num_channels=num_channels, dtype=dtype
    )

    for segment_index in range(recording.get_num_segments()):
        binary_traces = recorder_binary.get_traces(segment_index=segment_index)
        recording_traces = recording.get_traces(segment_index=segment_index)
        assert np.allclose(binary_traces, recording_traces)


def test_write_memory_recording():
    # 2 segments
    recording = MockRecording(
        num_channels=2, durations=[10.325, 3.5], sampling_frequency=30_000, strategy="tile_pregenerated"
    )

    # write with loop
    traces_list, shms = write_memory_recording(recording, dtype=None, verbose=True, n_jobs=1)

    traces_list, shms = write_memory_recording(
        recording, dtype=None, verbose=True, n_jobs=1, chunk_memory="100k", progress_bar=True
    )

    # write parallel
    traces_list, shms = write_memory_recording(recording, dtype=None, verbose=False, n_jobs=2, chunk_memory="100k")
    # need to clean the buffer
    del traces_list
    for shm in shms:
        shm.unlink()


def test_get_random_data_chunks():
    rec = generate_recording(num_channels=1, sampling_frequency=1000.0, durations=[10.0, 20.0])
    chunks = get_random_data_chunks(rec, num_chunks_per_segment=50, chunk_size=500, seed=0)
    assert chunks.shape == (50000, 1)


def test_get_closest_channels():
    rec = generate_recording(num_channels=32, sampling_frequency=1000.0, durations=[0.1])
    closest_channels_inds, distances = get_closest_channels(rec)
    closest_channels_inds, distances = get_closest_channels(rec, num_channels=4)


def test_get_noise_levels():
    job_kwargs = dict(n_jobs=1, progress_bar=True)
    rec = generate_recording(num_channels=2, sampling_frequency=1000.0, durations=[60.0])

    noise_levels_1 = get_noise_levels(rec, return_in_uV=False, **job_kwargs)
    noise_levels_2 = get_noise_levels(rec, return_in_uV=False, **job_kwargs)

    rec.set_channel_gains(0.1)
    rec.set_channel_offsets(0)
    _ = get_noise_levels(rec, return_in_uV=True, force_recompute=True, **job_kwargs)
    _ = get_noise_levels(rec, return_in_uV=True, method="std", **job_kwargs)

    # Generate a recording following a gaussian distribution to check the result of get_noise.
    std = 6.0
    seed = 0
    rng = np.random.default_rng(seed=seed)
    traces = rng.normal(loc=10.0, scale=std, size=(1_000_000, 2))
    recording = NumpyRecording(traces, 30000)

    assert np.all(noise_levels_1 == noise_levels_2)
    assert np.allclose(get_noise_levels(recording, return_in_uV=False, **job_kwargs), [std, std], rtol=1e-2, atol=1e-3)
    assert np.allclose(
        get_noise_levels(recording, method="std", return_in_uV=False, **job_kwargs), [std, std], rtol=1e-2, atol=1e-3
    )


def test_get_noise_levels_output():
    # Generate a recording following a gaussian distribution to check the result of get_noise.
    std = 6.0
    seed = 0
    rng = np.random.default_rng(seed=seed)
    num_channels = 2
    num_samples = 10_000
    sampling_frequency = 30_000.0
    traces = rng.normal(loc=10.0, scale=std, size=(num_samples, num_channels))
    recording = NumpyRecording(traces_list=traces, sampling_frequency=sampling_frequency)

    std_estimated_with_mad = get_noise_levels(
        recording,
        method="mad",
        return_in_uV=False,
        random_slices_kwargs=dict(num_chunks_per_segment=40, chunk_size=1_000, seed=seed),
    )
    print(std_estimated_with_mad)
    assert np.allclose(std_estimated_with_mad, [std, std], rtol=1e-2, atol=1e-3)

    std_estimated_with_std = get_noise_levels(
        recording,
        method="std",
        return_in_uV=False,
        random_slices_kwargs=dict(num_chunks_per_segment=40, chunk_size=1_000, seed=seed),
    )
    assert np.allclose(std_estimated_with_std, [std, std], rtol=1e-2, atol=1e-3)


def test_get_noise_levels_cache_parameters(monkeypatch):
    """Sampling parameters must be part of the noise-level cache semantics."""
    recording = generate_recording(num_channels=2, sampling_frequency=1000.0, durations=[2.0])
    computed_chunks = []

    def fake_noise_level_chunk(segment_index, start_frame, end_frame, worker_ctx):
        value = float(len(computed_chunks) + 1)
        computed_chunks.append((segment_index, start_frame, end_frame))
        return np.full(recording.get_num_channels(), value)

    monkeypatch.setattr(recording_tools, "_noise_level_chunk", fake_noise_level_chunk)
    sampling_kwargs = {"num_chunks_per_segment": 1, "chunk_size": 100, "seed": 42}

    first = get_noise_levels(
        recording, return_in_uV=False, random_slices_kwargs=sampling_kwargs, n_jobs=1, progress_bar=False
    )
    np.testing.assert_array_equal(first, [1.0, 1.0])

    # Identical parameters hit the cache, and seed=None is a wildcard for the seed.
    cached = get_noise_levels(
        recording, return_in_uV=False, random_slices_kwargs=sampling_kwargs, n_jobs=1, progress_bar=False
    )
    cached_any_seed = get_noise_levels(
        recording,
        return_in_uV=False,
        random_slices_kwargs={**sampling_kwargs, "seed": None},
        n_jobs=1,
        progress_bar=False,
    )
    np.testing.assert_array_equal(cached, first)
    np.testing.assert_array_equal(cached_any_seed, first)
    assert len(computed_chunks) == 1

    # A different explicit seed and a different chunk layout each trigger recomputation.
    different_seed = get_noise_levels(
        recording,
        return_in_uV=False,
        random_slices_kwargs={**sampling_kwargs, "seed": 123},
        n_jobs=1,
        progress_bar=False,
    )
    np.testing.assert_array_equal(different_seed, [2.0, 2.0])
    different_layout = get_noise_levels(
        recording,
        return_in_uV=False,
        random_slices_kwargs={**sampling_kwargs, "seed": 123, "num_chunks_per_segment": 2},
        n_jobs=1,
        progress_bar=False,
    )
    np.testing.assert_array_equal(different_layout, [3.5, 3.5])
    assert len(computed_chunks) == 4

    # The new layout can be reused when the caller does not care about the seed.
    cached_any_seed = get_noise_levels(
        recording,
        return_in_uV=False,
        random_slices_kwargs={**sampling_kwargs, "seed": None, "num_chunks_per_segment": 2},
        n_jobs=1,
        progress_bar=False,
    )
    np.testing.assert_array_equal(cached_any_seed, different_layout)
    assert len(computed_chunks) == 4

    # force_recompute remains an unconditional override.
    forced = get_noise_levels(
        recording,
        return_in_uV=False,
        force_recompute=True,
        random_slices_kwargs={**sampling_kwargs, "seed": 123, "num_chunks_per_segment": 2},
        n_jobs=1,
        progress_bar=False,
    )
    np.testing.assert_array_equal(forced, [5.5, 5.5])
    assert len(computed_chunks) == 6

    # Overwriting a computed property turns it back into user-supplied state.
    key = "noise_level_mad_raw"
    manual_values = np.full(recording.get_num_channels(), 9.0)
    recording.set_property(key, manual_values)
    assert key not in recording._noise_levels_params
    np.testing.assert_array_equal(
        get_noise_levels(recording, return_in_uV=False, n_jobs=1, progress_bar=False), manual_values
    )

    # An explicit sampling request recomputes the manually supplied value.
    get_noise_levels(recording, return_in_uV=False, random_slices_kwargs=sampling_kwargs, n_jobs=1, progress_bar=False)
    assert key in recording._noise_levels_params
    recording.delete_property(key)
    assert key not in recording._noise_levels_params


def test_noise_levels_cache_invalidated_on_gain_change():
    recording = generate_recording(num_channels=4, durations=[5.0])
    recording.set_channel_gains([1.0] * 4)
    recording.set_channel_offsets([0.0] * 4)

    sampling_kwargs = {"seed": 42}

    scaled_before = {}
    raw_before = {}

    for method in ("mad", "std", "rms"):
        scaled_before[method] = get_noise_levels(
            recording,
            method=method,
            return_in_uV=True,
            random_slices_kwargs=sampling_kwargs,
            n_jobs=1,
            progress_bar=False,
        ).copy()

        raw_before[method] = get_noise_levels(
            recording,
            method=method,
            return_in_uV=False,
            random_slices_kwargs=sampling_kwargs,
            n_jobs=1,
            progress_bar=False,
        ).copy()

    recording.set_channel_gains([10.0] * 4)

    for method in ("mad", "std", "rms"):
        scaled_after = get_noise_levels(
            recording,
            method=method,
            return_in_uV=True,
            random_slices_kwargs=sampling_kwargs,
            n_jobs=1,
            progress_bar=False,
        )

        raw_after = get_noise_levels(
            recording,
            method=method,
            return_in_uV=False,
            random_slices_kwargs=sampling_kwargs,
            n_jobs=1,
            progress_bar=False,
        )

        np.testing.assert_allclose(scaled_after, scaled_before[method] * 10, rtol=1e-4)
        np.testing.assert_array_equal(raw_after, raw_before[method])


def test_noise_levels_cache_invalidated_on_offset_change():
    recording = generate_recording(num_channels=4, durations=[5.0])
    recording.set_channel_gains([1.0] * 4)
    recording.set_channel_offsets([0.0] * 4)

    sampling_kwargs = {"seed": 42}

    rms_before = get_noise_levels(
        recording,
        method="rms",
        return_in_uV=True,
        random_slices_kwargs=sampling_kwargs,
        n_jobs=1,
        progress_bar=False,
    ).copy()

    std_before = get_noise_levels(
        recording,
        method="std",
        return_in_uV=True,
        random_slices_kwargs=sampling_kwargs,
        n_jobs=1,
        progress_bar=False,
    ).copy()

    mad_before = get_noise_levels(
        recording,
        method="mad",
        return_in_uV=True,
        random_slices_kwargs=sampling_kwargs,
        n_jobs=1,
        progress_bar=False,
    ).copy()

    recording.set_channel_offsets([100.0] * 4)

    rms_after = get_noise_levels(
        recording,
        method="rms",
        return_in_uV=True,
        random_slices_kwargs=sampling_kwargs,
        n_jobs=1,
        progress_bar=False,
    ).copy()

    std_after = get_noise_levels(
        recording,
        method="std",
        return_in_uV=True,
        random_slices_kwargs=sampling_kwargs,
        n_jobs=1,
        progress_bar=False,
    ).copy()

    mad_after = get_noise_levels(
        recording,
        method="mad",
        return_in_uV=True,
        random_slices_kwargs=sampling_kwargs,
        n_jobs=1,
        progress_bar=False,
    ).copy()

    assert not np.allclose(rms_before, rms_after)
    np.testing.assert_allclose(std_after, std_before, rtol=1e-5)
    np.testing.assert_allclose(mad_after, mad_before, rtol=1e-5)


def test_noise_levels_cache_invalidated_on_calibration_property_deletion():
    recording = generate_recording(num_channels=4, durations=[5.0])
    recording.set_channel_gains([1.0] * 4)
    recording.set_channel_offsets([0.0] * 4)

    sampling_kwargs = {"seed": 42}

    for method in ("mad", "std", "rms"):
        get_noise_levels(
            recording,
            method=method,
            return_in_uV=True,
            random_slices_kwargs=sampling_kwargs,
            n_jobs=1,
            progress_bar=False,
        )

    scaled_keys = [
        "noise_level_mad_scaled",
        "noise_level_std_scaled",
        "noise_level_rms_scaled",
    ]

    for key in scaled_keys:
        assert key in recording.get_property_keys()

    recording.delete_property("gain_to_uV")

    for key in scaled_keys:
        assert key not in recording.get_property_keys()


def test_get_noise_levels_normalizes_chunk_duration(monkeypatch):
    """Equivalent duration- and frame-based chunk specifications share the cache."""
    recording = generate_recording(num_channels=1, sampling_frequency=1000.0, durations=[2.0])
    computed_chunks = []

    def fake_noise_level_chunk(segment_index, start_frame, end_frame, worker_ctx):
        computed_chunks.append((segment_index, start_frame, end_frame))
        return np.ones(recording.get_num_channels())

    monkeypatch.setattr(recording_tools, "_noise_level_chunk", fake_noise_level_chunk)
    by_duration = get_noise_levels(
        recording,
        return_in_uV=False,
        random_slices_kwargs={"num_chunks_per_segment": 1, "seed": 0},
        n_jobs=1,
        progress_bar=False,
    )
    by_frames = get_noise_levels(
        recording,
        return_in_uV=False,
        random_slices_kwargs={"num_chunks_per_segment": 1, "chunk_size": 500, "seed": 0},
        n_jobs=1,
        progress_bar=False,
    )
    np.testing.assert_array_equal(by_frames, by_duration)
    assert len(computed_chunks) == 1


def test_get_chunk_with_margin():
    rec = generate_recording(num_channels=1, sampling_frequency=1000.0, durations=[10.0])
    rec_seg = rec.segments[0]
    length = rec_seg.get_num_samples()

    #  rec_segment, start_frame, end_frame, channel_indices, sample_margin

    traces, l, r = get_chunk_with_margin(rec_seg, None, None, None, 10)
    assert l == 0 and r == 0

    traces, l, r = get_chunk_with_margin(rec_seg, 5, None, None, 10)
    assert l == 5 and r == 0

    traces, l, r = get_chunk_with_margin(rec_seg, length - 1000, length - 5, None, 10)
    assert l == 10 and r == 5
    assert traces.shape[0] == 1010

    traces, l, r = get_chunk_with_margin(rec_seg, 2000, 3000, None, 10)
    assert l == 10 and r == 10
    assert traces.shape[0] == 1020

    # add zeros
    traces, l, r = get_chunk_with_margin(rec_seg, 5, 1005, None, 10, add_zeros=True)
    assert traces.shape[0] == 1020
    assert l == 10
    assert r == 10
    assert np.all(traces[:5] == 0)

    traces, l, r = get_chunk_with_margin(rec_seg, length - 1005, length - 5, None, 10, add_zeros=True)
    assert traces.shape[0] == 1020
    assert np.all(traces[-5:] == 0)
    assert l == 10
    assert r == 10

    traces, l, r = get_chunk_with_margin(rec_seg, length - 500, length + 500, None, 10, add_zeros=True)
    assert traces.shape[0] == 1020
    assert np.all(traces[-510:] == 0)
    assert l == 10
    assert r == 510

    # add zeros + window and/or dtype
    traces_windowed, l, r = get_chunk_with_margin(rec_seg, 5, 1005, None, 20, add_zeros=True, window_on_margin=True)
    traces_windowed, l, r = get_chunk_with_margin(
        rec_seg, length - 1005, length - 5, None, 20, add_zeros=True, window_on_margin=True
    )
    traces_windowed, l, r = get_chunk_with_margin(
        rec_seg, length - 500, length + 500, None, 10, add_zeros=True, window_on_margin=True
    )
    traces, l, r = get_chunk_with_margin(rec_seg, length - 1005, length - 5, None, 20, add_zeros=True, dtype="float64")
    assert traces.dtype == "float64"

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(traces_windowed[:, 0], color='r', ls='--')
    # plt.show()


def test_order_channels_by_depth():
    rec = generate_recording(num_channels=10, sampling_frequency=1000.0, durations=[10.0], set_probe=False)
    locations = np.zeros((10, 2))
    locations[:, 1] = np.arange(10) // 2 * 50
    locations[:, 0] = np.arange(10) % 2 * 30

    locations_copy = locations.copy()
    locations_copy = locations_copy[::-1]
    rec.set_channel_locations(locations_copy)

    order_1d, _ = order_channels_by_depth(rec, dimensions="y")
    order_2d, order_r2d = order_channels_by_depth(rec, dimensions=("x", "y"))
    order_2d_fliped, _ = order_channels_by_depth(rec, dimensions=("x", "y"), flip=True)

    assert np.array_equal(locations[:, 1], locations_copy[order_1d][:, 1])
    assert np.array_equal(locations_copy[order_1d][:, 1], locations_copy[order_2d][:, 1])
    assert np.array_equal(locations, locations_copy[order_2d])
    assert np.array_equal(locations_copy, locations_copy[order_2d][order_r2d])
    assert np.array_equal(order_2d[::-1], order_2d_fliped)


def test_do_recording_attributes_match():
    recording = MockRecording(
        num_channels=2, durations=[10.325, 3.5], sampling_frequency=30_000, strategy="tile_pregenerated"
    )
    rec_attributes = get_rec_attributes(recording)
    do_match, _ = do_recording_attributes_match(recording, rec_attributes)
    assert do_match

    rec_attributes = get_rec_attributes(recording)
    rec_attributes["sampling_frequency"] = 1.0
    do_match, exc = do_recording_attributes_match(recording, rec_attributes)
    assert not do_match
    assert "sampling_frequency" in exc

    # check dtype options
    rec_attributes = get_rec_attributes(recording)
    rec_attributes["dtype"] = "int16"
    do_match, exc = do_recording_attributes_match(recording, rec_attributes)
    assert not do_match
    assert "dtype" in exc
    do_match, exc = do_recording_attributes_match(recording, rec_attributes, check_dtype=False)
    assert do_match

    # check missing dtype
    rec_attributes.pop("dtype")
    do_match, exc = do_recording_attributes_match(recording, rec_attributes)
    assert do_match


if __name__ == "__main__":
    # Create a temporary folder using the standard library
    # import tempfile

    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     tmp_path = Path(tmpdirname)
    #     test_write_binary_recording(tmp_path)
    # test_write_memory_recording()

    # test_get_random_data_chunks()
    # test_get_closest_channels()
    # test_get_noise_levels()
    # test_get_noise_levels_output()
    test_order_channels_by_depth()
