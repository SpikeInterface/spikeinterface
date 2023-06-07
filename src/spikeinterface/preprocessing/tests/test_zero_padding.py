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


@pytest.mark.parametrize("padding_left, padding_right", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_trace_padded_recording_full_trace(padding_left, padding_right):
    num_channels = 4
    num_samples = 10
    rng = np.random.default_rng(seed=0)
    traces = rng.random(size=(num_samples, num_channels))
    traces_list = [traces]
    recording = NumpyRecording(traces_list=traces_list, sampling_frequency=30_000)

    padded_recording = ZeroTracePaddedRecording(
        parent_recording=recording,
        padding_left=padding_left,
        padding_right=padding_right,
    )
    padded_traces = padded_recording.get_traces()

    # Until padding_left the traces should be filled with zeros
    assert np.allclose(padded_traces[:padding_left, :], np.zeros((padding_left, num_channels)))
    last_frame_of_original_trace = num_samples + padding_left
    # Then from padding_left to the number of samples plus padding_left it should be the original traces
    assert np.allclose(padded_traces[padding_left:last_frame_of_original_trace, :], traces)
    # After the original trace is over it should have zeros until the end of the padding right
    assert np.allclose(padded_traces[last_frame_of_original_trace:, :], np.zeros((padding_right, num_channels)))


@pytest.mark.parametrize("padding_left, padding_right", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_trace_padded_recording_retrieve_original_trace(padding_left, padding_right):
    num_channels = 4
    num_samples = 10
    rng = np.random.default_rng(seed=0)
    traces = rng.random(size=(num_samples, num_channels))
    traces_list = [traces]
    recording = NumpyRecording(traces_list=traces_list, sampling_frequency=30_000)

    padded_recording = ZeroTracePaddedRecording(
        parent_recording=recording,
        padding_left=padding_left,
        padding_right=padding_right,
    )

    # These are the limits of the original trace
    start_frame = padding_left
    end_frame = num_samples + padding_left
    padded_traces = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)

    assert np.allclose(padded_traces, traces)


@pytest.mark.parametrize("padding_left, padding_right", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_trace_padded_recording_retrieve_partial_original_trace(padding_left, padding_right):
    num_channels = 4
    num_samples = 10
    rng = np.random.default_rng(seed=0)
    traces = rng.random(size=(num_samples, num_channels))
    traces_list = [traces]
    recording = NumpyRecording(traces_list=traces_list, sampling_frequency=30_000)

    padded_recording = ZeroTracePaddedRecording(
        parent_recording=recording,
        padding_left=padding_left,
        padding_right=padding_right,
    )

    # These are the limits of the original trace
    start_frame = padding_left + 2
    end_frame = num_samples + padding_left - 1
    padded_traces = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)

    assert np.allclose(padded_traces, traces)


@pytest.mark.parametrize("padding_left, padding_right", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_trace_padded_recording_retrieve_traces_with_partial_padding(padding_left, padding_right):
    num_channels = 4
    num_samples = 10
    rng = np.random.default_rng(seed=0)
    traces = rng.random(size=(num_samples, num_channels))
    traces_list = [traces]
    recording = NumpyRecording(traces_list=traces_list, sampling_frequency=30_000)

    padded_recording = ZeroTracePaddedRecording(
        parent_recording=recording,
        padding_left=padding_left,
        padding_right=padding_right,
    )

    # Extract the traces with partial padding in the left and to the right
    number_of_padded_frames_at_start = 2
    start_frame = padding_left - number_of_padded_frames_at_start

    number_of_paded_frames_at_end = 2
    last_frame_of_original_trace = num_samples + padding_left
    end_frame = last_frame_of_original_trace + number_of_paded_frames_at_end

    padded_traces = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)

    # Find that there as many frames padded a the start as expected
    padded_traces_start = padded_traces[:number_of_padded_frames_at_start, :]
    expected_zeros = np.zeros((number_of_padded_frames_at_start, num_channels))
    assert np.allclose(padded_traces_start, expected_zeros)

    # Then from padding_left to the number of samples plus padding_left it should be the original traces
    assert np.allclose(padded_traces[number_of_padded_frames_at_start:-number_of_paded_frames_at_end, :], traces)

    # Find that there as many frames padded at the end as expected
    padded_traces_end = padded_traces[-number_of_paded_frames_at_end:, :]
    expected_zeros = np.zeros((number_of_paded_frames_at_end, num_channels))
    assert np.allclose(padded_traces_end, expected_zeros)


if __name__ == "__main__":
    test_zero_paddin_channel()
