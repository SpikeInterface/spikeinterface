import pytest
from pathlib import Path
from spikeinterface.core import extract_waveforms, AddTemplatesRecording, NpzSortingExtractor, load_extractor, set_global_tmp_folder
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.core.testing_tools import generate_recording, create_sorting_npz


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core" / "add_templates_recording"
else:
    cache_folder = Path("cache_folder") / "core" / "add_templates_recording"

set_global_tmp_folder(cache_folder)
cache_folder.mkdir(parents=True, exist_ok=True)


def test_add_templates_recording():
    recording = generate_recording(num_channels = 4)
    recording.annotate(is_filtered=True)
    recording = recording.save(folder=cache_folder / "recording")

    npz_filename = cache_folder / "sorting.npz"
    sorting_npz = create_sorting_npz(num_seg=2, file_path=npz_filename)
    sorting = NpzSortingExtractor(npz_filename)

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

    # Check dumpability
    saved_loaded = load_extractor(add_templates_recording.to_dict())
    check_recordings_equal(add_templates_recording, saved_loaded, return_scaled=False)

    saved_1job = add_templates_recording.save(folder=cache_folder / "1job")
    saved_2job = add_templates_recording.save(folder=cache_folder / "2job", n_jobs=2, chunk_duration='1s')
    check_recordings_equal(add_templates_recording, saved_1job, return_scaled=False)
    check_recordings_equal(add_templates_recording, saved_2job, return_scaled=False)
