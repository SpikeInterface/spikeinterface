import pytest
import numpy as np
from spikeinterface.core.testing_tools import generate_recording
from spikeinterface.preprocessing import scale_to_uV, CenterRecording


def test_scale_to_uV():
    # Create a sample recording extractor with fake gains and offsets
    num_channels = 4
    sampling_frequency = 30_000.0
    durations = [1.0, 1.0]  # seconds
    recording = generate_recording(
        num_channels=num_channels,
        durations=durations,
        sampling_frequency=sampling_frequency,
    )

    rng = np.random.default_rng(0)
    gains = rng.random(size=(num_channels)).astype(np.float32)
    offsets = rng.random(size=(num_channels)).astype(np.float32)
    recording.set_channel_gains(gains)
    recording.set_channel_offsets(offsets)

    # Apply the preprocessor
    scaled_recording = scale_to_uV(recording=recording)

    # Check if the traces are indeed scaled
    expected_traces = recording.get_traces(return_scaled=True, segment_index=0)
    scaled_traces = scaled_recording.get_traces(segment_index=0)

    np.testing.assert_allclose(scaled_traces, expected_traces)

    # Test for the error when recording doesn't have scaleable traces
    recording.set_channel_gains(None)  # Remove gains to make traces unscaleable
    with pytest.raises(RuntimeError):
        scale_to_uV(recording)


def test_scaling_in_preprocessing_chain():

    # Create a sample recording extractor with fake gains and offsets
    num_channels = 4
    sampling_frequency = 30_000.0
    durations = [1.0]  # seconds
    recording = generate_recording(
        num_channels=num_channels,
        durations=durations,
        sampling_frequency=sampling_frequency,
    )

    rng = np.random.default_rng(0)
    gains = rng.random(size=(num_channels)).astype(np.float32)
    offsets = rng.random(size=(num_channels)).astype(np.float32)

    recording.set_channel_gains(gains)
    recording.set_channel_offsets(offsets)

    centered_recording = CenterRecording(scale_to_uV(recording=recording))
    traces_scaled_with_argument = centered_recording.get_traces(return_scaled=True)

    # Chain preprocessors
    centered_recording_scaled = CenterRecording(scale_to_uV(recording=recording))
    traces_scaled_with_preprocessor = centered_recording_scaled.get_traces()

    np.testing.assert_allclose(traces_scaled_with_argument, traces_scaled_with_preprocessor)

    # Test if the scaling is not done twice
    traces_scaled_with_preprocessor_and_argument = centered_recording_scaled.get_traces(return_scaled=True)

    np.testing.assert_allclose(traces_scaled_with_preprocessor, traces_scaled_with_preprocessor_and_argument)
