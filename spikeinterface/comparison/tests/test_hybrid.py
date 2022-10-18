import pytest
from pathlib import Path
from spikeinterface.core import WaveformExtractor, extract_waveforms, load_extractor, set_global_tmp_folder, waveform_extractor
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.comparison import create_hybrid_units_recording, create_hybrid_spikes_recording, generate_injected_sorting
from spikeinterface.extractors import toy_example
from spikeinterface.preprocessing import bandpass_filter


if hasattr(pytest, "global_test_folder"):
	cache_folder = pytest.global_test_folder / "comparison" / "hybrid"
else:
	cache_folder = Path("cache_folder") / "comparison" / "hybrid"

hybrid_folder = cache_folder / "hybrid"


def setup_module():
	hybrid_folder.mkdir(parents=True, exist_ok=True)
	recording, sorting = toy_example(duration=60, num_channels=4, num_units=5,
									 num_segments=2, average_peak_amplitude=-1000)
	recording = bandpass_filter(recording, freq_min=300, freq_max=6000)
	recording = recording.save(folder=hybrid_folder / "recording")
	sorting = sorting.save(folder=hybrid_folder / "sorting")

	wvf_extractor = extract_waveforms(recording, sorting, folder=hybrid_folder / "wvf_extractor",
									  ms_before=10., ms_after=10.)


def test_hybrid_units_recording():
	wvf_extractor = WaveformExtractor.load_from_folder(hybrid_folder / "wvf_extractor")
	recording = wvf_extractor.recording
	templates = wvf_extractor.get_all_templates()
	templates[:, 0, :] = 0
	templates[:, -1, :] = 0
	hybrid_units_recording = create_hybrid_units_recording(recording, templates, nbefore=wvf_extractor.nbefore,
														injected_sorting_folder=hybrid_folder / "injected0")

	assert hybrid_units_recording.get_traces(end_frame=600, segment_index=0).shape == (600, 4)
	assert hybrid_units_recording.get_traces(start_frame=100, end_frame=600, segment_index=1).shape == (500, 4)
	assert hybrid_units_recording.get_traces(start_frame=recording.get_num_frames(0) - 200, segment_index=0).shape == (200, 4)

	# Check dumpability
	saved_loaded = load_extractor(hybrid_units_recording.to_dict())
	check_recordings_equal(hybrid_units_recording, saved_loaded, return_scaled=False)

	saved_1job = hybrid_units_recording.save(folder=cache_folder / "units_1job")
	saved_2job = hybrid_units_recording.save(folder=cache_folder / "units_2job", n_jobs=2, chunk_duration='1s')
	check_recordings_equal(hybrid_units_recording, saved_1job, return_scaled=False)
	check_recordings_equal(hybrid_units_recording, saved_2job, return_scaled=False)


def test_hybrid_spikes_recording():
	wvf_extractor = WaveformExtractor.load_from_folder(hybrid_folder / "wvf_extractor")
	recording = wvf_extractor.recording
	sorting = wvf_extractor.sorting
	hybrid_spikes_recording = create_hybrid_spikes_recording(wvf_extractor,
															 injected_sorting_folder=hybrid_folder / "injected1")
	hybrid_spikes_recording = create_hybrid_spikes_recording(wvf_extractor, unit_ids=sorting.unit_ids[:3],
															 injected_sorting_folder=hybrid_folder / "injected2")

	assert hybrid_spikes_recording.get_traces(end_frame=600, segment_index=0).shape == (600, 4)
	assert hybrid_spikes_recording.get_traces(start_frame=100, end_frame=600, segment_index=1).shape == (500, 4)
	assert hybrid_spikes_recording.get_traces(start_frame=recording.get_num_frames(0) - 200, segment_index=0).shape == (200, 4)

	# Check dumpability
	saved_loaded = load_extractor(hybrid_spikes_recording.to_dict())
	check_recordings_equal(hybrid_spikes_recording, saved_loaded, return_scaled=False)

	saved_1job = hybrid_spikes_recording.save(folder=cache_folder / "spikes_1job")
	saved_2job = hybrid_spikes_recording.save(folder=cache_folder / "spikes_2job", n_jobs=2, chunk_duration='1s')
	check_recordings_equal(hybrid_spikes_recording, saved_1job, return_scaled=False)
	check_recordings_equal(hybrid_spikes_recording, saved_2job, return_scaled=False)


def test_generate_injected_sorting():
	recording = load_extractor(hybrid_folder / "recording")
	sorting = load_extractor(hybrid_folder / "sorting")
	injected_sorting = generate_injected_sorting(sorting, [recording.get_num_frames(seg_index) for seg_index in range(recording.get_num_segments())])


if __name__ == "__main__":
    setup_module()
    test_generate_injected_sorting()
    test_hybrid_units_recording()
    test_hybrid_spikes_recording()
