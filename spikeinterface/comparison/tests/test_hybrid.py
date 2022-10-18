import pytest
from pathlib import Path
from spikeinterface.core import NpzSortingExtractor, extract_waveforms, load_extractor, set_global_tmp_folder
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.comparison import create_hybrid_units_recording, create_hybrid_spikes_recording, generate_injected_sorting
from spikeinterface.extractors import toy_example
from spikeinterface.preprocessing import bandpass_filter


if hasattr(pytest, "global_test_folder"):
	cache_folder = pytest.global_test_folder / "comparison" / "hybrid"
else:
	cache_folder = Path("cache_folder") / "comparison" / "hybrid"

set_global_tmp_folder(cache_folder)
cache_folder.mkdir(parents=True, exist_ok=True)


recording, sorting = toy_example(duration=60, num_channels=4, num_units=5,
								 num_segments=2, average_peak_amplitude=-1000)
recording = recording.save(folder=cache_folder / "recording")
recording_f = bandpass_filter(recording, freq_min=300, freq_max=6000)

npz_filename = cache_folder / "sorting.npz"
NpzSortingExtractor.write_sorting(sorting, npz_filename)
sorting = NpzSortingExtractor(npz_filename)

wvf_extractor = extract_waveforms(recording_f, sorting, folder=cache_folder / "wvf_extractor", ms_before=10., ms_after=10.)


def test_hybrid_units_recording():
	hybrid_units_recording = create_hybrid_units_recording(recording_f, wvf_extractor.get_all_templates(),
														   nbefore=wvf_extractor.nbefore, filename=cache_folder / "hybrid_units.npz")

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
	hybrid_spikes_recording = create_hybrid_spikes_recording(wvf_extractor)
	hybrid_spikes_recording = create_hybrid_spikes_recording(wvf_extractor, unit_ids=sorting.unit_ids[:3], filename=cache_folder / "hybrid_spikes.npz")

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
	injected_sorting = generate_injected_sorting(sorting, [recording.get_num_frames(seg_index) for seg_index in range(recording.get_num_segments())])
