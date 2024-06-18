import pytest
from pathlib import Path
import numpy as np

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording
from spikeinterface.core.numpyextractors import NumpyRecording

from spikeinterface.preprocessing import zero_channel_pad, bandpass_filter, phase_shift
from spikeinterface.preprocessing.zero_channel_pad import TracePaddedRecording


def test_zero_padding_channel():
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


@pytest.fixture
def recording():
    num_channels = 4
    num_samples = 10000
    rng = np.random.default_rng(seed=0)
    traces = rng.random(size=(num_samples, num_channels))
    traces_list = [traces]
    recording = NumpyRecording(traces_list=traces_list, sampling_frequency=30_000)
    return recording


@pytest.mark.parametrize("padding_start, padding_end", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_trace_padded_recording_full_trace(recording, padding_start, padding_end):
    num_channels = recording.get_num_channels()
    num_samples = recording.get_num_samples()

    padded_recording = TracePaddedRecording(
        recording=recording,
        padding_start=padding_start,
        padding_end=padding_end,
    )
    padded_traces = padded_recording.get_traces()

    # Until padding_start the traces should be filled with zeros
    padded_traces_start = padded_traces[:padding_start, :]
    expected_padding = np.zeros((padding_start, num_channels))
    assert np.allclose(padded_traces_start, expected_padding)

    # Then from padding_start to the number of samples plus padding_start it should be the original traces
    original_traces = recording.get_traces()
    last_frame_of_original_trace = num_samples + padding_start
    original_traces_from_padding = padded_traces[padding_start:last_frame_of_original_trace, :]
    assert np.allclose(original_traces_from_padding, original_traces)

    # After the original trace is over it should have zeros until the end of the padding right
    padded_traces_end = padded_traces[last_frame_of_original_trace:, :]
    expected_padding = np.zeros((padding_end, num_channels))
    assert np.allclose(padded_traces_end, expected_padding)


@pytest.mark.parametrize("padding_start, padding_end", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_trace_padded_recording_full_trace_with_channel_indices(recording, padding_start, padding_end):
    num_samples = recording.get_num_samples()

    padded_recording = TracePaddedRecording(
        recording=recording,
        padding_start=padding_start,
        padding_end=padding_end,
    )

    channel_indices = [2, 0]
    num_channels_sub_straces = len(channel_indices)
    padded_traces = padded_recording.get_traces(channel_ids=channel_indices)

    # Until padding_start the traces should be filled with zeros
    assert np.allclose(padded_traces[:padding_start, :], np.zeros((padding_start, num_channels_sub_straces)))
    last_frame_of_original_trace = num_samples + padding_start

    # Then from padding_start to the number of samples plus padding_start it should be the original traces
    original_traces = recording.get_traces(channel_ids=channel_indices)
    original_traces_from_padding = padded_traces[padding_start:last_frame_of_original_trace, :]
    assert np.allclose(original_traces_from_padding, original_traces)

    # After the original trace is over it should have zeros until the end of the padding right
    padded_traces_end = padded_traces[last_frame_of_original_trace:, :]
    expected_padding = np.zeros((padding_end, num_channels_sub_straces))
    assert np.allclose(padded_traces_end, expected_padding)


@pytest.mark.parametrize("padding_start, padding_end", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_trace_padded_recording_retrieve_original_trace(recording, padding_start, padding_end):
    num_samples = recording.get_num_samples()

    padded_recording = TracePaddedRecording(
        recording=recording,
        padding_start=padding_start,
        padding_end=padding_end,
    )

    # These are the limits of the original trace
    start_frame = padding_start
    end_frame = num_samples + padding_start
    padded_traces = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)

    original_traces = recording.get_traces()
    assert np.allclose(padded_traces, original_traces)


@pytest.mark.parametrize("padding_start, padding_end", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_trace_padded_recording_retrieve_partial_original_trace(recording, padding_start, padding_end):
    num_samples = recording.get_num_samples()

    padded_recording = TracePaddedRecording(
        recording=recording,
        padding_start=padding_start,
        padding_end=padding_end,
    )

    # We extract a trace that is smaller than the original trace
    start_frame_original_traces = 2
    end_frame_original_traces = num_samples - 3

    # Adjust the start and end frame to take into account the padding in the padded recoderer
    start_frame = padding_start + start_frame_original_traces
    end_frame = padding_start + end_frame_original_traces
    padded_traces = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)

    # Test for a match
    original_traces = recording.get_traces(start_frame=start_frame_original_traces, end_frame=end_frame_original_traces)
    assert np.allclose(padded_traces, original_traces)


@pytest.mark.parametrize("padding_start, padding_end", [(5, 5), (5, 0)])
def test_trace_padded_recording_retrieve_start_padding_and_partial_original_trace(
    recording, padding_start, padding_end
):
    num_samples = recording.get_num_samples()
    num_channels = recording.get_num_channels()

    padded_recording = TracePaddedRecording(
        recording=recording,
        padding_start=padding_start,
        padding_end=padding_end,
    )

    # We extract a traces with padding at the beginning and smaller than original trace at the end
    start_padding_to_retrieve = 2
    end_frame_original_traces = num_samples - 3

    # Adjust the start and end frame to take into account the padding in the padded recoderer
    start_frame = padding_start - start_padding_to_retrieve
    end_frame = padding_start + end_frame_original_traces
    padded_traces = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)

    # Check the the beginning of the trace is is padded with zeros
    start_padding = padded_traces[:start_padding_to_retrieve, :]
    expected_padding = np.zeros((start_padding_to_retrieve, num_channels))
    assert np.allclose(start_padding, expected_padding)

    # Check the partial retrieval of the original series
    partial_original_traces = recording.get_traces(start_frame=0, end_frame=end_frame_original_traces)
    partial_orignal_traces_from_padded_traces = padded_traces[start_padding_to_retrieve:, :]

    assert np.allclose(partial_original_traces, partial_orignal_traces_from_padded_traces)


@pytest.mark.parametrize("padding_start, padding_end", [(5, 5), (0, 5)])
def test_trace_padded_recording_retrieve_end_padding_and_partial_original_trace(recording, padding_start, padding_end):
    num_samples = recording.get_num_samples()
    num_channels = recording.get_num_channels()

    padded_recording = TracePaddedRecording(
        recording=recording,
        padding_start=padding_start,
        padding_end=padding_end,
    )

    # We extract traces that are smaller than original at the start and then padded at the end
    start_frame_original_traces = 2
    end_padding_to_retrieve = 3
    end_frame_original_traces = num_samples

    # Adjust the start and end frame to take into account the padding in the padded recoderer
    start_frame = padding_start + start_frame_original_traces
    end_frame = padding_start + end_frame_original_traces + end_padding_to_retrieve
    padded_traces = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)

    # Check the partial retrieval of the original series at the start
    partial_original_traces = recording.get_traces(
        start_frame=start_frame_original_traces, end_frame=end_frame_original_traces
    )
    partial_orignal_traces_from_padded_traces = padded_traces[:-end_padding_to_retrieve, :]
    assert np.allclose(partial_orignal_traces_from_padded_traces, partial_original_traces)

    # Check that the end of the padded traces is padded with zeros
    last_padding = padded_traces[end_frame_original_traces:, :]
    expected_padding = np.zeros((end_padding_to_retrieve, num_channels))
    assert np.allclose(last_padding, expected_padding)


@pytest.mark.parametrize("padding_start, padding_end", [(5, 5), (0, 5), (5, 0)])
def test_trace_padded_recording_retrieve_traces_with_partial_padding(recording, padding_start, padding_end):
    num_samples = recording.get_num_samples()
    num_channels = recording.get_num_channels()

    padded_recording = TracePaddedRecording(
        recording=recording,
        padding_start=padding_start,
        padding_end=padding_end,
    )

    # Define how many frames are padded at the start and end (take into account they could be 0)
    number_of_padded_frames_at_start = 2 if padding_start > 0 else 0
    number_of_paded_frames_at_end = 3 if padding_end > 0 else 0

    # Start before the padding starts and ensure the end includes padding by going
    # further than the number of samples of the original recording
    start_frame = padding_start - number_of_padded_frames_at_start
    end_frame = padding_start + num_samples + number_of_paded_frames_at_end

    # Retrieve the traces which should have partial padding at the start and end
    padded_traces = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)

    # Find that there as many frames padded a the start as expected
    padded_traces_start = padded_traces[:number_of_padded_frames_at_start, :]
    expected_zeros = np.zeros((number_of_padded_frames_at_start, num_channels))
    assert np.allclose(padded_traces_start, expected_zeros)

    # Then from padding_start to the number of samples plus padding_start this should be the original traces
    frame_where_original_trace_ends = number_of_padded_frames_at_start + num_samples
    original_traces_from_padded_traces = padded_traces[
        number_of_padded_frames_at_start:frame_where_original_trace_ends, :
    ]
    original_traces = recording.get_traces()
    assert np.allclose(original_traces_from_padded_traces, original_traces)

    # Find that there as many frames padded at the end as expected
    padded_traces_end = padded_traces[frame_where_original_trace_ends:, :]
    expected_zeros = np.zeros((number_of_paded_frames_at_end, num_channels))
    assert np.allclose(padded_traces_end, expected_zeros)


@pytest.mark.parametrize("padding_start, padding_end", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_trace_padded_recording_retrieve_only_start_padding(recording, padding_start, padding_end):
    num_samples = recording.get_num_samples()
    num_channels = recording.get_num_channels()

    padded_recording = TracePaddedRecording(
        recording=recording,
        padding_start=padding_start,
        padding_end=padding_end,
    )

    # Retrieve the padding at the start and test it
    padded_traces_start = padded_recording.get_traces(start_frame=0, end_frame=padding_start)
    expected_traces = np.zeros((padding_start, num_channels))
    assert np.allclose(padded_traces_start, expected_traces)


@pytest.mark.parametrize("padding_start, padding_end", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_trace_padded_recording_retrieve_only_end_padding(recording, padding_start, padding_end):
    num_samples = recording.get_num_samples()
    num_channels = recording.get_num_channels()

    padded_recording = TracePaddedRecording(
        recording=recording,
        padding_start=padding_start,
        padding_end=padding_end,
    )

    # Retrieve the padding at the end and test it
    start_frame = padding_start + num_samples
    end_frame = padding_start + num_samples + padding_end
    padded_traces_end = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    expected_traces = np.zeros((padding_end, num_channels))
    assert np.allclose(padded_traces_end, expected_traces)


@pytest.mark.parametrize("preprocessing", ["bandpass_filter", "phase_shift"])
@pytest.mark.parametrize("padding_start, padding_end", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_trace_padded_recording_retrieve_only_end_padding_with_preprocessing(
    recording, padding_start, padding_end, preprocessing
):
    """This is a tmeporary test to check that this works when the recording is called out of bonds. It should be removed
    when more general test are added in that direction"""

    num_samples = recording.get_num_samples()
    num_channels = recording.get_num_channels()

    if preprocessing == "bandpass_filter":
        recording = bandpass_filter(recording, freq_min=300, freq_max=6000)
    else:
        sample_shift_size = 0.4
        inter_sample_shift = np.arange(recording.get_num_channels()) * sample_shift_size
        recording.set_property("inter_sample_shift", inter_sample_shift)
        recording = phase_shift(recording)

    padded_recording = TracePaddedRecording(
        recording=recording,
        padding_start=padding_start,
        padding_end=padding_end,
    )

    # Retrieve the padding at the end and test it
    start_frame = padding_start + num_samples
    end_frame = padding_start + num_samples + padding_end
    padded_traces_end = padded_recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    expected_traces = np.zeros((padding_end, num_channels))
    assert np.allclose(padded_traces_end, expected_traces)


if __name__ == "__main__":
    test_zero_padding_channel()
