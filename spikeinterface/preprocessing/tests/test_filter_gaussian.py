from spikeinterface.core.testing_tools import generate_recording
from spikeinterface.preprocessing import gaussian_filter


def test_filter_gaussian():
	rec = generate_recording(num_channels=3)
	rec = gaussian_filter(rec)

	assert rec.get_traces(segment_index=0, end_frame=600).shape == (600, 3)
	assert rec.get_traces(segment_index=0, start_frame=100, end_frame=600).shape == (500, 3)
	assert rec.get_traces(segment_index=1, start_frame=rec.get_num_frames(1) - 200).shape == (200, 3)
