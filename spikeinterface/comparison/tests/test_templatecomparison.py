import shutil
import pytest
from pathlib import Path
import numpy as np

from spikeinterface.core import extract_waveforms
from spikeinterface.extractors import toy_example
from spikeinterface.comparison import compare_templates, compare_multiple_templates


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "comparison"
else:
    cache_folder = Path("cache_folder") / "comparison"


test_dir = cache_folder / "temp_comp_test"


def setup_module():
    if test_dir.is_dir():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)


def test_compare_multiple_templates():
    test_dir = Path("temp_comp_test")

    duration = 60
    num_channels = 8

    rec, sort = toy_example(
        duration=duration, num_segments=1, num_channels=num_channels)
    rec = rec.save(folder=test_dir / "rec")
    sort = sort.save(folder=test_dir / "sort")

    # split recording in 3 equal slices
    fs = rec.get_sampling_frequency()
    rec1 = rec.frame_slice(start_frame=0*fs,
                           end_frame=duration / 3 * fs)
    rec2 = rec.frame_slice(start_frame=duration / 3*fs,
                           end_frame=2 / 3 * duration * fs)
    rec3 = rec.frame_slice(start_frame=2 / 3 * duration * fs,
                           end_frame=duration*fs)
    sort1 = sort.frame_slice(start_frame=0*fs,
                             end_frame=duration / 3 * fs)
    sort2 = sort.frame_slice(start_frame=duration / 3*fs,
                             end_frame=2 / 3 * duration * fs)
    sort3 = sort.frame_slice(start_frame=2 / 3 * duration * fs,
                             end_frame=duration*fs)
    # compute waveforms
    we1 = extract_waveforms(rec1, sort1, test_dir / "wf1", n_jobs=1)
    we2 = extract_waveforms(rec2, sort2, test_dir / "wf2", n_jobs=1)
    we3 = extract_waveforms(rec3, sort3, test_dir / "wf3", n_jobs=1)

    # paired comparison
    temp_cmp = compare_templates(we1, we2)

    for u1 in temp_cmp.hungarian_match_12.index.values:
        u2 = temp_cmp.hungarian_match_12[u1]
        if u2 != -1:
            assert u1 == u2

    # multi-comparison
    temp_mcmp = compare_multiple_templates([we1, we2, we3])
    # assert unit ids are the same across sessions (because of initial slicing)
    for unit_dict in temp_mcmp.units.values():
        unit_ids = unit_dict["unit_ids"].values()
        if len(unit_ids) > 1:
            assert len(np.unique(unit_ids)) == 1


if __name__ == '__main__':
    test_compare_multiple_templates()
