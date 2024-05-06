import numpy as np
import pytest

from spikeinterface.core.generate import generate_recording
from spikeinterface.preprocessing import GenericPreprocessor


def test_basic_use():

    recording = generate_recording(num_channels=4, durations=[1.0])
    recording = recording.rename_channels(["a", "b", "c", "d"])
    function = np.mean  # function to apply to the traces

    # Initialize the preprocessor
    preprocessor = GenericPreprocessor(recording, function)

    traces = preprocessor.get_traces(channel_ids=["a", "d"])
    expected_traces = np.mean(recording.get_traces(channel_ids=["a", "d"]))

    np.testing.assert_allclose(traces, expected_traces)
