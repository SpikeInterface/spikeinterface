import pytest
import shutil
from pathlib import Path


from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms, NumpySorting, set_global_tmp_folder
from spikeinterface.extractors import toy_example


from spikeinterface.core.generate import inject_some_split_units
from spikeinterface.curation import get_potential_auto_merge


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"

set_global_tmp_folder(cache_folder)

def test_get_auto_merge_list():

    rec, sorting = toy_example(num_segments=1, num_units=5, duration=[600.])
    

    sorting_with_split = inject_some_split_units(sorting, split_ids=sorting.unit_ids[:2], num_split=2,)
    print(sorting_with_split)
    print(sorting_with_split.unit_ids)

    rec = rec.save()
    sorting_with_split = sorting_with_split.save()
    wf_folder = cache_folder / 'wf_auto_merge'
    if wf_folder.exists():
        shutil.rmtree(wf_folder)
    we = extract_waveforms(rec, sorting_with_split, mode='folder', folder=wf_folder, n_jobs=1)

    # we = extract_waveforms(rec, sorting_with_split, mode='memory', folder=None, n_jobs=1)
    print(we)

    potential_merges = get_potential_auto_merge(we,
                minimum_spikes=1000, maximum_distance_um=150.,
                peak_sign="neg",
                bin_ms=0.25, window_ms=100.,
                corr_diff_thresh=0.16,
                template_diff_thresh=0.25,
                censored_period_ms=0., refractory_period_ms=4.0,
                sigma_smooth_ms = 0.6,
                contamination_threshold=0.2,
                adaptative_window_threshold=0.5,
                num_channels=5,
                num_shift=5,
                firing_contamination_balance=1.5,
                extra_outputs=False,
                )
    print(potential_merges)


    


    
if __name__ == '__main__':
    test_get_auto_merge_list()
