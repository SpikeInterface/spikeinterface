import pytest
import numpy as np
from pathlib import Path

from spikeinterface.core import BinaryRecordingExtractor
from spikeinterface.core.numpyextractors import NumpyRecording
from spikeinterface.core.core_tools import measure_memory_allocation
from spikeinterface.core.generate import NoiseGeneratorRecording


def test_BinaryRecordingExtractor(create_cache_folder):
    cache_folder = create_cache_folder
    num_seg = 2
    num_channels = 3
    num_samples = 30
    sampling_frequency = 10000
    dtype = "int16"

    file_paths = [cache_folder / f"test_BinaryRecordingExtractor_{i}.raw" for i in range(num_seg)]
    for i in range(num_seg):
        np.memmap(file_paths[i], dtype=dtype, mode="w+", shape=(num_samples, num_channels))

    rec = BinaryRecordingExtractor(
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
    )

    file_paths = [cache_folder / f"test_BinaryRecordingExtractor_copied_{i}.raw" for i in range(num_seg)]
    BinaryRecordingExtractor.write_recording(rec, file_paths)

    file_paths = [cache_folder / f"test_BinaryRecordingExtractor_{i}.raw" for i in range(num_seg)]
    assert (cache_folder / "test_BinaryRecordingExtractor_copied_0.raw").is_file()


def test_round_trip(tmp_path):
    num_channels = 10
    num_samples = 500
    traces_list = [np.ones(shape=(num_samples, num_channels), dtype="int32")]
    sampling_frequency = 30_000.0
    recording = NumpyRecording(traces_list=traces_list, sampling_frequency=sampling_frequency)

    file_path = tmp_path / "test_BinaryRecordingExtractor.raw"
    dtype = recording.get_dtype()
    BinaryRecordingExtractor.write_recording(recording=recording, dtype=dtype, file_paths=file_path)

    sampling_frequency = recording.get_sampling_frequency()
    num_channels = recording.get_num_channels()
    binary_recorder = BinaryRecordingExtractor(
        file_paths=file_path,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
    )

    # Test for full traces
    assert np.allclose(recording.get_traces(), binary_recorder.get_traces())

    # Ttest for a sub-set of the traces
    start_frame = 20
    end_frame = 40
    smaller_traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    binary_smaller_traces = binary_recorder.get_traces(start_frame=start_frame, end_frame=end_frame)

    np.allclose(smaller_traces, binary_smaller_traces)


@pytest.fixture(scope="module")
def folder_with_binary_files(tmpdir_factory):
    tmp_path = Path(tmpdir_factory.mktemp("spike_interface_test"))
    folder = tmp_path / "test_binary_recording"
    num_channels = 32
    sampling_frequency = 30_000.0
    dtype = "float32"
    recording = NoiseGeneratorRecording(
        durations=[1.0],
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
    )
    dtype = recording.get_dtype()
    recording.save(folder=folder, overwrite=True)

    return folder


def test_sequential_reading_of_small_traces(folder_with_binary_files):
    # Test that memmap is readed correctly when pointing to specific frames
    folder = folder_with_binary_files
    num_channels = 32
    sampling_frequency = 30_000.0
    dtype = "float32"

    file_paths = [folder / "traces_cached_seg0.raw"]
    recording = BinaryRecordingExtractor(
        num_chan=num_channels,
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        dtype=dtype,
    )

    full_traces = recording.get_traces()

    # Test for a sub-set of the traces
    start_frame = 10
    end_frame = 15
    small_traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    expected_traces = full_traces[start_frame:end_frame, :]
    assert np.allclose(small_traces, expected_traces)

    # Test for a sub-set of the traces
    start_frame = 1000
    end_frame = 1100
    small_traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    expected_traces = full_traces[start_frame:end_frame, :]
    assert np.allclose(small_traces, expected_traces)

    # Test for a sub-set of the traces
    start_frame = 10_000
    end_frame = 11_000
    small_traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    expected_traces = full_traces[start_frame:end_frame, :]
    assert np.allclose(small_traces, expected_traces)


if __name__ == "__main__":
    test_BinaryRecordingExtractor()
