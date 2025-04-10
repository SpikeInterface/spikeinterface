import pytest
import numpy as np

from spikeinterface.extractors import WhiteMatterRecordingExtractor, BinaryRecordingExtractor
from spikeinterface.core.numpyextractors import NumpyRecording


def test_round_trip(tmp_path):
    num_channels = 10
    num_samples = 500
    traces_list = [np.ones(shape=(num_samples, num_channels), dtype="int16")]
    sampling_frequency = 30_000.0
    recording = NumpyRecording(traces_list=traces_list, sampling_frequency=sampling_frequency)

    file_path = tmp_path / "test_WhiteMatterRecordingExtractor.raw"
    BinaryRecordingExtractor.write_recording(recording=recording, file_paths=file_path, dtype="int16", byte_offset=8)

    sampling_frequency = recording.get_sampling_frequency()
    num_channels = recording.get_num_channels()
    binary_recorder = WhiteMatterRecordingExtractor(
        file_path=file_path,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
    )

    # Test for full traces
    assert np.allclose(recording.get_traces(), binary_recorder.get_traces())

    # Ttest for a sub-set of the traces
    start_frame = 20
    end_frame = 40
    smaller_traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    binary_smaller_traces = binary_recorder.get_traces(start_frame=start_frame, end_frame=end_frame)

    np.allclose(smaller_traces, binary_smaller_traces)


def test_sequential_reading_of_small_traces(tmp_path):
    # Test that memmap is readed correctly when pointing to specific frames
    num_channels = 10
    num_samples = 12_000
    traces_list = [np.ones(shape=(num_samples, num_channels), dtype="int16")]
    sampling_frequency = 30_000.0
    recording = NumpyRecording(traces_list=traces_list, sampling_frequency=sampling_frequency)

    file_path = tmp_path / "test_WhiteMatterRecordingExtractor.raw"
    BinaryRecordingExtractor.write_recording(recording=recording, file_paths=file_path, dtype="int16", byte_offset=8)

    sampling_frequency = recording.get_sampling_frequency()
    num_channels = recording.get_num_channels()
    recording = WhiteMatterRecordingExtractor(
        file_path=file_path,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
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
