from spikeinterface.core import extract_waveforms
from spikeinterface.comparison import create_hybrid_units_recording, create_hybrid_spikes_recording, generate_injected_sorting
from spikeinterface.extractors import toy_example
from spikeinterface.preprocessing import bandpass_filter


recording, sorting = toy_example(duration=60, num_channels=4, num_units=5,
								 num_segments=2, average_peak_amplitude=-1000)
recording_f = bandpass_filter(recording, freq_min=300, freq_max=6000)
wvf_extractor = extract_waveforms(recording_f, sorting, mode="memory", ms_before=10., ms_after=10.)


def test_hybrid_units_recording():
	hybrid_units_recording = create_hybrid_units_recording(recording_f, wvf_extractor.get_all_templates(),
														   nbefore=wvf_extractor.nbefore)

	assert hybrid_units_recording.get_traces(end_frame=600, segment_index=0).shape == (600, 4)
	assert hybrid_units_recording.get_traces(start_frame=100, end_frame=600, segment_index=1).shape == (500, 4)
	assert hybrid_units_recording.get_traces(start_frame=recording.get_num_frames(0) - 200, segment_index=0).shape == (200, 4)


def test_hybrid_spikes_recording():
	hybrid_spikes_recording = create_hybrid_spikes_recording(wvf_extractor)
	hybrid_spikes_recording = create_hybrid_spikes_recording(wvf_extractor, unit_ids=sorting.unit_ids[:3])

	assert hybrid_spikes_recording.get_traces(end_frame=600, segment_index=0).shape == (600, 4)
	assert hybrid_spikes_recording.get_traces(start_frame=100, end_frame=600, segment_index=1).shape == (500, 4)
	assert hybrid_spikes_recording.get_traces(start_frame=recording.get_num_frames(0) - 200, segment_index=0).shape == (200, 4)


def test_generate_injected_sorting():
	injected_sorting = generate_injected_sorting(sorting, [recording.get_num_frames(seg_index) for seg_index in range(recording.get_num_segments())])
