import pytest
import numpy as np
from spikeinterface.core.testing_tools import generate_recording
from spikeinterface.preprocessing import ScaleTouVRecording  # Replace 'your_module' with your actual module name


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

    rng = np.random.default_rng(0)
    gains = rng.random(size=(num_channels)).astype(np.float32)
    offsets = rng.random(size=(num_channels)).astype(np.float32)
    recording.set_channel_gains(gains)
    recording.set_channel_offsets(offsets)

    # Apply the preprocessor
    scaled_recording = ScaleTouVRecording(recording=recording)

    # Check if the traces are indeed scaled
    expected_traces = recording.get_traces(return_scaled=True)
    scaled_traces = scaled_recording.get_traces()

    np.testing.assert_allclose(scaled_traces, expected_traces)

    # Test for the error when recording doesn't have scaleable traces
    recording.set_channel_gains(None)  # Remove gains to make traces unscaleable
    with pytest.raises(AssertionError):
        ScaleTouVRecording(recording)
