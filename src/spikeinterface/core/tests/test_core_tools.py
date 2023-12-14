import platform
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import importlib
import pytest
import numpy as np

from spikeinterface.core.core_tools import (
    write_binary_recording,
    write_memory_recording,
    recursive_path_modifier,
    make_paths_relative,
    make_paths_absolute,
    check_paths_relative,
)
from spikeinterface.core.binaryrecordingextractor import BinaryRecordingExtractor
from spikeinterface.core.generate import NoiseGeneratorRecording
from spikeinterface.core.numpyextractors import NumpySorting


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_write_binary_recording(tmp_path):
    # Test write_binary_recording() with loop (n_jobs=1)
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
    job_kwargs = dict(verbose=False, n_jobs=1)
    write_binary_recording(recording, file_paths=file_paths, dtype=dtype, **job_kwargs)

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
    recording = NoiseGeneratorRecording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        strategy="tile_pregenerated",
    )
    file_paths = [tmp_path / "binary01.raw"]

    # Write binary recording
    job_kwargs = dict(verbose=False, n_jobs=1)
    byte_offset = 125
    write_binary_recording(recording, file_paths=file_paths, dtype=dtype, byte_offset=byte_offset, **job_kwargs)

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
    recording = NoiseGeneratorRecording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        dtype=dtype,
        strategy="tile_pregenerated",
    )
    file_paths = [tmp_path / "binary01.raw", tmp_path / "binary02.raw"]

    # Write binary recording
    job_kwargs = dict(verbose=False, n_jobs=2, chunk_memory="100k", mp_context="spawn")
    write_binary_recording(recording, file_paths=file_paths, dtype=dtype, **job_kwargs)

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
    recording = NoiseGeneratorRecording(
        durations=durations,
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        strategy="tile_pregenerated",
    )
    file_paths = [tmp_path / "binary01.raw", tmp_path / "binary02.raw"]

    # Write binary recording
    job_kwargs = dict(verbose=False, n_jobs=2, chunk_memory="100k", mp_context="spawn")
    write_binary_recording(recording, file_paths=file_paths, dtype=dtype, **job_kwargs)

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
    write_memory_recording(recording, dtype=None, verbose=True, n_jobs=1)

    write_memory_recording(recording, dtype=None, verbose=True, n_jobs=1, chunk_memory="100k", progress_bar=True)

    if platform.system() != "Windows":
        # write parrallel
        write_memory_recording(recording, dtype=None, verbose=False, n_jobs=2, chunk_memory="100k")

        # write parrallel
        write_memory_recording(recording, dtype=None, verbose=False, n_jobs=2, total_memory="200k", progress_bar=True)


def test_path_utils_functions():
    if platform.system() != "Windows":
        # posix path
        d = {
            "kwargs": {
                "path": "/yep/sub/path1",
                "recording": {
                    "module": "mock_module",
                    "class": "mock_class",
                    "version": "1.2",
                    "annotations": {},
                    "kwargs": {"path": "/yep/sub/path2"},
                },
            }
        }

        d2 = recursive_path_modifier(d, lambda p: p.replace("/yep", "/yop"))
        assert d2["kwargs"]["path"].startswith("/yop")
        assert d2["kwargs"]["recording"]["kwargs"]["path"].startswith("/yop")

        d3 = make_paths_relative(d, Path("/yep"))
        assert d3["kwargs"]["path"] == "sub/path1"
        assert d3["kwargs"]["recording"]["kwargs"]["path"] == "sub/path2"

        d4 = make_paths_absolute(d3, "/yop")
        assert d4["kwargs"]["path"].startswith("/yop")
        assert d4["kwargs"]["recording"]["kwargs"]["path"].startswith("/yop")

    if platform.system() == "Windows":
        # test for windows Path
        d = {
            "kwargs": {
                "path": r"c:\yep\sub\path1",
                "recording": {
                    "module": "mock_module",
                    "class": "mock_class",
                    "version": "1.2",
                    "annotations": {},
                    "kwargs": {"path": r"c:\yep\sub\path2"},
                },
            }
        }

        d2 = make_paths_relative(d, "c:\\yep")
        # the str be must unix like path even on windows for more portability
        assert d2["kwargs"]["path"] == "sub/path1"
        assert d2["kwargs"]["recording"]["kwargs"]["path"] == "sub/path2"

        # same drive
        assert check_paths_relative(d, r"c:\yep")
        # not the same drive
        assert not check_paths_relative(d, r"d:\yep")

        d = {
            "kwargs": {
                "path": r"\\host\share\yep\sub\path1",
            }
        }
        # UNC cannot be relative to d: drive
        assert not check_paths_relative(d, r"d:\yep")

        # UNC can be relative to the same UNC
        assert check_paths_relative(d, r"\\host\share")


if __name__ == "__main__":
    # Create a temporary folder using the standard library
    # import tempfile

    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     tmp_path = Path(tmpdirname)
    #     test_write_binary_recording(tmp_path)
    # test_write_memory_recording()

    test_path_utils_functions()
