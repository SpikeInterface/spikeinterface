import pytest
from pathlib import Path
import numpy as np

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording
from spikeinterface.core.numpyextractors import NumpyRecording

from spikeinterface.preprocessing import zero_channel_pad
from spikeinterface.preprocessing.zero_channel_pad import ZeroTracePaddedRecording

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"

set_global_tmp_folder(cache_folder)


def test_zero_paddin_channel():
    num_original_channels = 4
    num_padded_channels = num_original_channels + 8
    rec = generate_recording(num_channels=num_original_channels, durations=[10])

    rec2 = zero_channel_pad(rec, num_channels=num_padded_channels)
    rec2.save(verbose=False)

    print(rec2)

    assert rec2.get_num_channels() == num_padded_channels

    tr = rec2.get_traces()
    assert np.allclose(
        tr[:, num_original_channels:], np.zeros((rec2.get_num_samples(), num_padded_channels - num_original_channels))
    )
    assert np.allclose(tr[:, :num_original_channels], rec.get_traces())


def test_zero_padding_trace():
    num_channels = 4
    num_samples = 10
    traces = np.ones((num_samples, num_channels))
    traces_list = [traces]
    recording = NumpyRecording(traces_list=traces_list, sampling_frequency=30_000)

    padding_left = 5
    padding_right = 5
    padded_recording = ZeroTracePaddedRecording(
        parent_recording=recording, padding_left=padding_left, padding_right=padding_right
    )
    padded_traces = padded_recording.get_traces()

    # Padd traces with zeros on the left and right
    assert np.allclose(padded_traces[:padding_left, :], np.zeros((padding_left, num_channels)))
    assert np.allclose(padded_traces[padding_left:-padding_right, :], traces)
    assert np.allclose(padded_traces[-padding_right:, :], np.zeros((padding_right, num_channels)))


if __name__ == "__main__":
    test_zero_paddin_channel()
