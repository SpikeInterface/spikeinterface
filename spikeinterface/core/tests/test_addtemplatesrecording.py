from spikeinterface.core import extract_waveforms, AddTemplatesRecording
from spikeinterface.core.testing_tools import generate_recording, generate_sorting


def test_add_templates_recording():
	recording = generate_recording(num_channels = 4)
	recording.annotate(is_filtered=True)
	sorting = generate_sorting()
	wvf_extractor = extract_waveforms(recording, sorting, mode="memory", ms_before=3., ms_after=3.)
	templates = wvf_extractor.get_all_templates()
	templates[:, 0] = templates[:, -1] = 0.0  # Go around the check for the edge, this is just testing.

	# parent_recording = None
	add_templates_recording = AddTemplatesRecording(sorting, templates, nbefore=wvf_extractor.nbefore,
													num_samples=[recording.get_num_frames(seg_ind) for seg_ind in range(recording.get_num_segments())])

	assert add_templates_recording.get_traces(end_frame=600, segment_index=0).shape == (600, 4)
	assert add_templates_recording.get_traces(start_frame=100, end_frame=600, segment_index=1).shape == (500, 4)
	assert add_templates_recording.get_traces(start_frame=recording.get_num_frames(0) - 200, segment_index=0).shape == (200, 4)

	# parent_recording != None
	add_templates_recording = AddTemplatesRecording(sorting, templates, nbefore=wvf_extractor.nbefore,parent_recording=recording)

	assert add_templates_recording.get_traces(end_frame=600, segment_index=0).shape == (600, 4)
	assert add_templates_recording.get_traces(start_frame=100, end_frame=600, segment_index=1).shape == (500, 4)
	assert add_templates_recording.get_traces(start_frame=recording.get_num_frames(0) - 200, segment_index=0).shape == (200, 4)
