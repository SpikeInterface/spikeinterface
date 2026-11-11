import pytest
import numpy as np
from spikeinterface.core.testing_tools import generate_recording
from spikeinterface.preprocessing.preprocessing_classes import scale_to_uV, CenterRecording, scale_to_physical_units


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
    expected_traces = recording.get_traces(return_in_uV=True, segment_index=0)
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

    centered_recording = CenterRecording(scale_to_uV(recording=recording), seed=2205)
    traces_scaled_with_argument = centered_recording.get_traces(return_in_uV=True)

    # Chain preprocessors
    centered_recording_scaled = CenterRecording(scale_to_uV(recording=recording), seed=2205)
    traces_scaled_with_preprocessor = centered_recording_scaled.get_traces()

    np.testing.assert_allclose(traces_scaled_with_argument, traces_scaled_with_preprocessor)

    # Test if the scaling is not done twice
    traces_scaled_with_preprocessor_and_argument = centered_recording_scaled.get_traces(return_in_uV=True)

    np.testing.assert_allclose(traces_scaled_with_preprocessor, traces_scaled_with_preprocessor_and_argument)


def test_scale_to_physical_units():
    # Create a sample recording extractor with fake physical unit gains and offsets
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

    # Set physical unit gains/offsets instead of regular gains/offsets
    recording.set_property("gain_to_physical_unit", gains)
    recording.set_property("offset_to_physical_unit", offsets)

    # Apply the preprocessor
    scaled_recording = scale_to_physical_units(recording=recording)

    # Get raw traces and apply scaling manually
    raw_traces = recording.get_traces(segment_index=0)
    expected_traces = raw_traces * gains + offsets

    # Get scaled traces
    scaled_traces = scaled_recording.get_traces(segment_index=0)

    # Check if the traces are scaled correctly
    np.testing.assert_allclose(scaled_traces, expected_traces)

    # Test for the error when recording doesn't have physical unit properties
    recording_no_gains = generate_recording(
        num_channels=num_channels,
        durations=durations,
        sampling_frequency=sampling_frequency,
    )
    with pytest.raises(ValueError):
        scale_to_physical_units(recording_no_gains)


if __name__ == "__main__":
    test_scale_to_uV()
    test_scaling_in_preprocessing_chain()
