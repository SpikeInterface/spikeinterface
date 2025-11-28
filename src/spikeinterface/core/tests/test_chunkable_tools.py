import numpy as np

from spikeinterface.core import generate_recording

from spikeinterface.core.binaryrecordingextractor import BinaryRecordingExtractor
from spikeinterface.core.generate import NoiseGeneratorRecording


from spikeinterface.core.chunkable_tools import (
    write_binary,
    write_memory,
    get_random_sample_slices,
    get_chunks,
)


def test_write_binary(tmp_path):
    # Test write_binary() with loop (n_jobs=1)
    # Setup
    sampling_frequency = 30_000
    num_channels = 2
    dtype = "float32"

    durations = [10.0]
    recording = NoiseGeneratorRecording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        strategy="tile_pregenerated",
    )
    file_paths = [tmp_path / "binary01.raw"]

    # Write binary recording
    job_kwargs = dict(n_jobs=1)
    write_binary(recording, file_paths=file_paths, dtype=dtype, verbose=False, **job_kwargs)

    # Check if written data matches original data
    recorder_binary = BinaryRecordingExtractor(
        file_paths=file_paths, sampling_frequency=sampling_frequency, num_channels=num_channels, dtype=dtype
    )
    assert np.allclose(recorder_binary.get_traces(), recording.get_traces())


def test_write_binary_offset(tmp_path):
    # Test write_binary() with loop (n_jobs=1)
    # Setup
    sampling_frequency = 30_000
    num_channels = 2
    dtype = "float32"

    durations = [10.0]
    recording = NoiseGeneratorRecording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        strategy="tile_pregenerated",
    )
    file_paths = [tmp_path / "binary01.raw"]

    # Write binary recording
    job_kwargs = dict(n_jobs=1)
    byte_offset = 125
    write_binary(recording, file_paths=file_paths, dtype=dtype, byte_offset=byte_offset, verbose=False, **job_kwargs)

    # Check if written data matches original data
    recorder_binary = BinaryRecordingExtractor(
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
        file_offset=byte_offset,
    )
    assert np.allclose(recorder_binary.get_traces(), recording.get_traces())


def test_write_binary_parallel(tmp_path):
    # Test write_binary() with parallel processing (n_jobs=2)

    # Setup
    sampling_frequency = 30_000
    num_channels = 2
    dtype = "float32"
    durations = [10.30, 3.5]
    recording = NoiseGeneratorRecording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        dtype=dtype,
        strategy="tile_pregenerated",
    )
    file_paths = [tmp_path / "binary01.raw", tmp_path / "binary02.raw"]

    # Write binary recording
    job_kwargs = dict(n_jobs=2, chunk_memory="100k", mp_context="spawn")
    write_binary(recording, file_paths=file_paths, dtype=dtype, verbose=False, **job_kwargs)

    # Check if written data matches original data
    recorder_binary = BinaryRecordingExtractor(
        file_paths=file_paths, sampling_frequency=sampling_frequency, num_channels=num_channels, dtype=dtype
    )
    for segment_index in range(recording.get_num_segments()):
        binary_traces = recorder_binary.get_traces(segment_index=segment_index)
        recording_traces = recording.get_traces(segment_index=segment_index)
        assert np.allclose(binary_traces, recording_traces)


def test_write_binary_multiple_segment(tmp_path):
    # Test write_binary() with multiple segments (n_jobs=2)
    # Setup
    sampling_frequency = 30_000
    num_channels = 10
    dtype = "float32"

    durations = [10.30, 3.5]
    recording = NoiseGeneratorRecording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        strategy="tile_pregenerated",
    )
    file_paths = [tmp_path / "binary01.raw", tmp_path / "binary02.raw"]

    # Write binary recording
    job_kwargs = dict(n_jobs=2, chunk_memory="100k", mp_context="spawn")
    write_binary(recording, file_paths=file_paths, dtype=dtype, verbose=False, **job_kwargs)

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
    recording = NoiseGeneratorRecording(
        num_channels=2, durations=[10.325, 3.5], sampling_frequency=30_000, strategy="tile_pregenerated"
    )
    recording = recording.save()

    # write with loop
    traces_list, shms = write_memory(recording, dtype=None, verbose=True, n_jobs=1)

    traces_list, shms = write_memory(
        recording, dtype=None, verbose=True, n_jobs=1, chunk_memory="100k", progress_bar=True
    )

    # write parallel
    traces_list, shms = write_memory(recording, dtype=None, verbose=False, n_jobs=2, chunk_memory="100k")
    # need to clean the buffer
    del traces_list
    for shm in shms:
        shm.unlink()


def test_get_random_sample_slices():
    rec = generate_recording(num_channels=1, sampling_frequency=1000.0, durations=[10.0, 20.0])
    rec_slices = get_random_sample_slices(
        rec, method="full_random", num_chunks_per_segment=20, chunk_duration="500ms", margin_frames=0, seed=0
    )
    assert len(rec_slices) == 40
    for seg_ind, start, stop in rec_slices:
        assert stop - start == 500
        assert seg_ind in (0, 1)


def test_get_chunks():
    rec = generate_recording(num_channels=1, sampling_frequency=1000.0, durations=[10.0, 20.0])
    chunks = get_chunks(rec, num_chunks_per_segment=50, chunk_size=500, seed=0)
    assert chunks.shape == (50000, 1)
