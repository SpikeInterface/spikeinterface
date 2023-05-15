import numpy as np
import pytest
from pathlib import Path
from spikeinterface.core import load_extractor, set_global_tmp_folder
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.core.generate import generate_recording
from spikeinterface.preprocessing import gaussian_bandpass_filter


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing" / "gaussian_bandpass_filter"
else:
    cache_folder = Path("cache_folder") / "preprocessing" / "gaussian_bandpass_filter"

set_global_tmp_folder(cache_folder)
cache_folder.mkdir(parents=True, exist_ok=True)


def test_filter_gaussian():
    recording = generate_recording(num_channels=3)
    recording.annotate(is_filtered=True)
    recording = recording.save(folder=cache_folder / "recording")

    rec_filtered = gaussian_bandpass_filter(recording)

    assert rec_filtered.dtype == recording.dtype
    assert rec_filtered.get_traces(segment_index=0, end_frame=100).dtype == rec_filtered.dtype
    assert rec_filtered.get_traces(segment_index=0, end_frame=600).shape == (600, 3)
    assert rec_filtered.get_traces(segment_index=0, start_frame=100, end_frame=600).shape == (500, 3)
    assert rec_filtered.get_traces(segment_index=1, start_frame=rec_filtered.get_num_frames(1) - 200).shape == (200, 3)

    # Check dumpability
    saved_loaded = load_extractor(rec_filtered.to_dict())
    check_recordings_equal(rec_filtered, saved_loaded, return_scaled=False)

    saved_1job = rec_filtered.save(folder=cache_folder / "1job")
    saved_2job = rec_filtered.save(folder=cache_folder / "2job", n_jobs=2, chunk_duration='1s')

    for seg_idx in range(rec_filtered.get_num_segments()):
        original_trace = rec_filtered.get_traces(seg_idx)
        saved1_trace = saved_1job.get_traces(seg_idx)
        saved2_trace = saved_2job.get_traces(seg_idx)

        assert np.allclose(original_trace[60:-60], saved1_trace[60:-60], rtol=1e-3, atol=1e-3)
        assert np.allclose(original_trace[60:-60], saved2_trace[60:-60], rtol=1e-3, atol=1e-3)
