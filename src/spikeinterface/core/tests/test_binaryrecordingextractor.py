import pytest
import numpy as np
from pathlib import Path

from spikeinterface.core import BinaryRecordingExtractor
from spikeinterface.core.numpyextractors import NumpyRecording

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_BinaryRecordingExtractor():
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
    num_samples = 50
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

    assert np.allclose(recording.get_traces(), binary_recorder.get_traces())

    start_frame = 200
    end_frame = 500
    smaller_traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    binary_smaller_traces = binary_recorder.get_traces(start_frame=start_frame, end_frame=end_frame)

    np.allclose(smaller_traces, binary_smaller_traces)


if __name__ == "__main__":
    test_BinaryRecordingExtractor()
