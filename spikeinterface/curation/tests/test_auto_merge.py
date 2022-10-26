import pytest
import shutil
from pathlib import Path


from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms, NumpySorting, set_global_tmp_folder
from spikeinterface.extractors import toy_example


from spikeinterface.core.generate import inject_some_duplicat_units
from spikeinterface.curation import get_potential_auto_merge


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"

set_global_tmp_folder(cache_folder)

def test_get_auto_merge_list():

    rec, sorting = toy_example(num_segments=1, num_units=5, duration=[10.])
    
    sorting_with_dup = inject_some_duplicat_units(sorting, num=3, max_shift=2, ratio=0.7, seed=None)
    
    we = extract_waveforms(rec, sorting_with_dup, mode='memory', folder=None, n_jobs=1)
    print(we)

    potential_merges = get_potential_auto_merge(we,
                minimum_spikes=1000, maximum_distance_um=150.,
                peak_sign="neg",
                bin_ms=0.25, window_ms=100.,
                corr_diff_thresh=0.16,
                template_diff_thresh=0.25,
                censored_period_ms=0., refractory_period_ms=1.0,
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
