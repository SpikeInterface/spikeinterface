import pytest
import numpy as np
from pathlib import Path

from spikeinterface.extractors import WhiteMatterRecordingExtractor, BinaryRecordingExtractor
from spikeinterface.core.numpyextractors import NumpyRecording
from spikeinterface import get_global_dataset_folder, download_dataset


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

    # Test for a sub-set of the traces
    start_frame = 20
    end_frame = 40
    smaller_traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    binary_smaller_traces = binary_recorder.get_traces(start_frame=start_frame, end_frame=end_frame)

    np.allclose(smaller_traces, binary_smaller_traces)


gin_repo = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data"
local_folder = get_global_dataset_folder() / "ephy_testing_data"
remote_path = Path("whitematter") / "HSW_2024_12_12__10_28_23__70min_17sec__hsamp_64ch_25000sps_stub.bin"


def test_on_data():
    file_path = download_dataset(
        repo=gin_repo, remote_path=remote_path, local_folder=local_folder, update_if_exists=True
    )

    sampling_frequency = 25_000.0
    num_channels = 64
    recording = WhiteMatterRecordingExtractor(
        file_path=file_path,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
    )
    assert recording.get_sampling_frequency() == sampling_frequency
    assert recording.get_num_channels() == num_channels
    assert recording.get_duration() == 1.0


def test_kwargs():
    file_path = download_dataset(
        repo=gin_repo, remote_path=remote_path, local_folder=local_folder, update_if_exists=True
    )

    sampling_frequency = 25_000.0
    num_channels = 64
    recording1 = WhiteMatterRecordingExtractor(
        file_path=file_path,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        is_filtered=True,
    )
    recording2 = WhiteMatterRecordingExtractor(**recording1._kwargs)

    assert recording1.get_sampling_frequency() == recording2.get_sampling_frequency()
    assert recording1.get_num_channels() == recording2.get_num_channels()
    assert recording1.get_duration() == recording2.get_duration()
