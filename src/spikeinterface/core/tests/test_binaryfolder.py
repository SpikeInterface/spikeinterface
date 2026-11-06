import pytest

import json
from pathlib import Path
import shutil

import numpy as np

from spikeinterface.core import load
from spikeinterface.core import generate_recording


def test_BinaryFolderRecording(create_cache_folder):
    cache_folder = create_cache_folder
    rec = generate_recording(num_channels=10, durations=[2.0, 2.0])
    folder = cache_folder / "binary_folder_1"

    if folder.is_dir():
        shutil.rmtree(folder)

    saved_rec = rec.save(folder=folder)
    print(saved_rec)

    loaded_rec = load(folder)
    print(loaded_rec)


@pytest.mark.parametrize("other_segment_t_start", [None, 100.0])
def test_save_time_vector_on_one_segment(tmp_path, monkeypatch, other_segment_t_start):
    recording = generate_recording(num_channels=2, durations=[0.1, 0.2], sampling_frequency=1000.0, seed=0)
    times = 7.0 + np.arange(recording.get_num_samples(segment_index=0)) / 500.0
    recording.set_times(times, segment_index=0)
    if other_segment_t_start is not None:
        recording.shift_times(other_segment_t_start, segment_index=1)

    monkeypatch.chdir(tmp_path)
    recording.save(format="binary", folder="saved")

    with open(Path("saved") / "binary.json", "r") as f:
        binary_dict = json.load(f)
    assert binary_dict["relative_paths"]
    assert binary_dict["kwargs"]["file_timestamps_paths"] == ["times_cached_seg0.raw", None]
    assert not (Path("saved") / "times_cached_seg1.raw").exists()

    # In a copy, the time vector is only found if the paths are relative to the folder
    shutil.copytree("saved", "copied")
    copied_folder = tmp_path / "copied"
    # A folder gives a BinaryFolderRecording and the json file gives a BinaryRecordingExtractor
    for loaded_rec in (load(copied_folder), load(copied_folder / "binary.json", base_folder=True)):
        assert loaded_rec.segments[0].file_timestamps_path.parent.name == "copied"
        assert loaded_rec.has_time_vector(segment_index=0)
        np.testing.assert_array_equal(loaded_rec.get_times(segment_index=0), times)
        assert not loaded_rec.has_time_vector(segment_index=1)
        assert loaded_rec.segments[1].file_timestamps_path is None
        assert loaded_rec.get_start_time(segment_index=1) == (other_segment_t_start or 0.0)
        assert loaded_rec.get_segment_t_starts() == (None if other_segment_t_start is None else [None, 100.0])
        for segment_index in range(2):
            np.testing.assert_array_equal(
                loaded_rec.get_traces(segment_index=segment_index), recording.get_traces(segment_index=segment_index)
            )


if __name__ == "__main__":
    test_BinaryFolderRecording()
