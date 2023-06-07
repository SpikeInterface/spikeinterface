import pytest
from pathlib import Path
import numpy as np

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording
from spikeinterface.core.numpyextractors import NumpyRecording

from spikeinterface.preprocessing import zero_channel_pad
from spikeinterface.preprocessing.zero_channel_pad import TracePaddedRecording

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


@pytest.fixture(scope="module")
def recording_numpy():
    num_channels = 4
    num_samples = 10
    rng = np.random.default_rng(seed=0)
    traces = rng.random(size=(num_samples, num_channels))
    traces_list = [traces]
    recording = NumpyRecording(traces_list=traces_list, sampling_frequency=30_000)
    return recording


@pytest.fixture(scope="module")
def recording_mearec():
    from spikeinterface.core.datasets import download_dataset
    from spikeinterface.extractors.neoextractors.mearec import MEArecRecordingExtractor

    mearec_path = download_dataset()
    recording = MEArecRecordingExtractor(file_path=mearec_path)

    return recording


@pytest.fixture(scope="module")
def recording(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "recording, padding_left, padding_right",
    [
        ("recording_numpy", 5, 5),
        ("recording_numpy", 0, 5),
        ("recording_numpy", 5, 0),
        ("recording_numpy", 0, 0),
        ("recording_mearec", 5, 5),
        ("recording_mearec", 0, 5),
        ("recording_mearec", 5, 0),
        ("recording_mearec", 0, 0),
    ],
    indirect=["recording"],
)
def test_trace_padded_recording_full_trace(recording, padding_left, padding_right):
    num_channels = recording.get_num_channels()
    num_samples = recording.get_num_samples()

    padded_recording = TracePaddedRecording(
        parent_recording=recording,
        padding_left=padding_left,
        padding_right=padding_right,
    )
    padded_traces = padded_recording.get_traces()

    # Until padding_left the traces should be filled with zeros
    assert np.allclose(padded_traces[:padding_left, :], np.zeros((padding_left, num_channels)))
    last_frame_of_original_trace = num_samples + padding_left

    # Then from padding_left to the number of samples plus padding_left it should be the original traces
    original_traces = recording.get_traces()
    original_traces_from_padding = padded_traces[padding_left:last_frame_of_original_trace, :]
    assert np.allclose(original_traces_from_padding, original_traces)

    # After the original trace is over it should have zeros until the end of the padding right
    assert np.allclose(padded_traces[last_frame_of_original_trace:, :], np.zeros((padding_right, num_channels)))


@pytest.mark.parametrize(
    "recording, padding_left, padding_right",
    [
        ("recording_numpy", 5, 5),
        ("recording_numpy", 0, 5),
        ("recording_numpy", 5, 0),
        ("recording_numpy", 0, 0),
        ("recording_mearec", 5, 5),
        ("recording_mearec", 0, 5),
        ("recording_mearec", 5, 0),
        ("recording_mearec", 0, 0),
    ],
    indirect=["recording"],
)
def test_trace_padded_recording_retrieve_original_trace(recording, padding_left, padding_right):
    num_samples = recording.get_num_samples()

    padded_recording = TracePaddedRecording(
        parent_recording=recording,
        padding_left=padding_left,
        padding_right=padding_right,
    )

    # These are the limits of the original trace
    start_frame = padding_left
    end_frame = num_samples + padding_left
    padded_traces = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)

    original_traces = recording.get_traces()
    assert np.allclose(padded_traces, original_traces)


@pytest.mark.parametrize(
    "recording, padding_left, padding_right",
    [
        ("recording_numpy", 5, 5),
        ("recording_numpy", 0, 5),
        ("recording_numpy", 5, 0),
        ("recording_numpy", 0, 0),
        ("recording_mearec", 5, 5),
        ("recording_mearec", 0, 5),
        ("recording_mearec", 5, 0),
        ("recording_mearec", 0, 0),
    ],
    indirect=["recording"],
)
def test_trace_padded_recording_retrieve_partial_original_trace(recording, padding_left, padding_right):
    num_samples = recording.get_num_samples()

    padded_recording = TracePaddedRecording(
        parent_recording=recording,
        padding_left=padding_left,
        padding_right=padding_right,
    )

    # These are the limits of the original trace
    start_frame_original_traces = 2
    end_frame_original_traces = num_samples - 2

    start_frame = padding_left + start_frame_original_traces
    end_frame = padding_left + end_frame_original_traces
    padded_traces = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    original_traces = recording.get_traces(start_frame=start_frame_original_traces, end_frame=end_frame_original_traces)

    assert np.allclose(padded_traces, original_traces)


@pytest.mark.parametrize(
    "recording, padding_left, padding_right",
    [
        ("recording_numpy", 5, 5),
        ("recording_numpy", 0, 5),
        ("recording_numpy", 5, 0),
        ("recording_numpy", 0, 0),
        ("recording_mearec", 5, 5),
        ("recording_mearec", 0, 5),
        ("recording_mearec", 5, 0),
        ("recording_mearec", 0, 0),
    ],
    indirect=["recording"],
)
def test_trace_padded_recording_retrieve_traces_with_partial_padding(recording, padding_left, padding_right):
    num_samples = recording.get_num_samples()
    num_channels = recording.get_num_channels()

    padded_recording = TracePaddedRecording(
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
    original_traces_from_padding = padded_traces[number_of_padded_frames_at_start:-number_of_paded_frames_at_end, :]
    original_traces = recording.get_traces()
    assert np.allclose(original_traces_from_padding, original_traces)

    # Find that there as many frames padded at the end as expected
    padded_traces_end = padded_traces[-number_of_paded_frames_at_end:, :]
    expected_zeros = np.zeros((number_of_paded_frames_at_end, num_channels))
    assert np.allclose(padded_traces_end, expected_zeros)


if __name__ == "__main__":
    test_zero_paddin_channel()
