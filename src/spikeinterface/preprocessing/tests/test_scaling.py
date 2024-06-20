import pytest
import numpy as np
from spikeinterface.core.testing_tools import generate_recording
from spikeinterface.preprocessing import ScaleTouV  # Replace 'your_module' with your actual module name


def test_scale_to_uv():
    # Create a sample recording extractor with fake gains and offsets
    num_channels = 4
    sampling_frequency = 30_000.0
    durations = [1]  # seconds
    recording = generate_recording(
        num_channels=num_channels,
        durations=durations,
        sampling_frequency=sampling_frequency,
    )

    gains = np.ones(shape=(num_channels))
    offsets = np.zeros(shape=(num_channels))
    recording.set_channel_gains(gains)  # Random gains
    recording.set_channel_offsets(offsets)  # Random offsets

    # Apply the preprocessor
    scaled_recording = ScaleTouV(recording=recording)

    # Check if the traces are indeed scaled
    expected_traces = recording.get_traces(return_scaled=True)
    scaled_traces = scaled_recording.get_traces()

    np.testing.assert_allclose(scaled_traces, expected_traces)

    # Test for the error when recording doesn't have scaleable traces
    recording.set_channel_gains(None)  # Remove gains to make traces unscaleable
    with pytest.raises(AssertionError):
        ScaleTouV(recording)
