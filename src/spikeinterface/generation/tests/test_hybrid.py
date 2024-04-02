import pytest
import shutil
from pathlib import Path
from spikeinterface.core import create_sorting_analyzer, load_waveforms, load_extractor
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.comparison import (
    create_hybrid_units_recording,
    create_hybrid_spikes_recording,
)
from spikeinterface.extractors import toy_example
from spikeinterface.preprocessing import bandpass_filter


def _generate_analyzer():
    recording, sorting = toy_example(
        duration=60, num_channels=4, num_units=5, num_segments=2, average_peak_amplitude=-1000, seed=0
    )
    analyzer = create_sorting_analyzer(sorting, recording, format="memory", ms_before=10.0, ms_after=10.0)
    analyzer.compute(["random_spikes", "waveforms", "templates"])

    return analyzer


@pytest.fixture
def generate_analyzer():
    return _generate_analyzer()


def test_hybrid_units_recording(generate_analyzer, tmp_path):
    analyzer = generate_analyzer

    recording = analyzer.recording
    templates_ext = analyzer.get_extension("templates")
    templates = templates_ext.get_templates()
    templates[:, 0, :] = 0
    templates[:, -1, :] = 0
    hybrid_units_recording = create_hybrid_units_recording(recording, templates, nbefore=templates_ext.nbefore)

    assert hybrid_units_recording.get_traces(end_frame=600, segment_index=0).shape == (600, 4)
    assert hybrid_units_recording.get_traces(start_frame=100, end_frame=600, segment_index=1).shape == (500, 4)
    assert hybrid_units_recording.get_traces(start_frame=recording.get_num_frames(0) - 200, segment_index=0).shape == (
        200,
        4,
    )

    # Check dumpability
    saved_loaded = load_extractor(hybrid_units_recording.to_dict())
    check_recordings_equal(hybrid_units_recording, saved_loaded, return_scaled=False)

    saved_1job = hybrid_units_recording.save(folder=tmp_path / "units_1job")
    saved_2job = hybrid_units_recording.save(folder=tmp_path / "units_2job", n_jobs=2, chunk_duration="1s")
    check_recordings_equal(hybrid_units_recording, saved_1job, return_scaled=False)
    check_recordings_equal(hybrid_units_recording, saved_2job, return_scaled=False)


def test_hybrid_spikes_recording(generate_analyzer, tmp_path):
    analyzer = generate_analyzer
    recording = analyzer.recording
    sorting = analyzer.sorting
    hybrid_spikes_recording = create_hybrid_spikes_recording(analyzer)
    hybrid_spikes_recording = create_hybrid_spikes_recording(analyzer, unit_ids=sorting.unit_ids[:3])

    assert hybrid_spikes_recording.get_traces(end_frame=600, segment_index=0).shape == (600, 4)
    assert hybrid_spikes_recording.get_traces(start_frame=100, end_frame=600, segment_index=1).shape == (500, 4)
    assert hybrid_spikes_recording.get_traces(start_frame=recording.get_num_frames(0) - 200, segment_index=0).shape == (
        200,
        4,
    )

    # Check dumpability
    saved_loaded = load_extractor(hybrid_spikes_recording.to_dict())
    check_recordings_equal(hybrid_spikes_recording, saved_loaded, return_scaled=False)

    saved_1job = hybrid_spikes_recording.save(folder=tmp_path / "spikes_1job")
    saved_2job = hybrid_spikes_recording.save(folder=tmp_path / "spikes_2job", n_jobs=2, chunk_duration="1s")
    check_recordings_equal(hybrid_spikes_recording, saved_1job, return_scaled=False)
    check_recordings_equal(hybrid_spikes_recording, saved_2job, return_scaled=False)


if __name__ == "__main__":
    test_hybrid_units_recording(_generate_analyzer(), Path("./tmp_path"))
    test_hybrid_spikes_recording(_generate_analyzer(), Path("./tmp_path"))
