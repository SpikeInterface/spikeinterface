import pytest
import shutil
from pathlib import Path

import pytest

import numpy as np

from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms, NumpySorting, set_global_tmp_folder
from spikeinterface.core.generate import inject_some_duplicate_units

from spikeinterface.extractors import toy_example

from spikeinterface.curation import remove_redundant_units


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"

set_global_tmp_folder(cache_folder)

def test_remove_redundant_units():
    rec, sorting = toy_example(num_segments=1, duration=[10.], seed=0)
    
    sorting_with_dup = inject_some_duplicate_units(sorting, ratio=0.8, num=4, seed=1)
    
    rec = rec.save()
    sorting_with_dup = sorting_with_dup.save()
    wf_folder = cache_folder / 'wf_dup'
    if wf_folder.exists():
        shutil.rmtree(wf_folder)
    we = extract_waveforms(rec, sorting_with_dup, folder=wf_folder)
    print(we)

    for remove_strategy in ('max_spikes', 'minimum_shift', 'highest_amplitude'):
        sorting_clean = remove_redundant_units(we, remove_strategy=remove_strategy)
        # print(sorting_clean)
        # print(sorting_clean.unit_ids)
        assert np.array_equal(sorting_clean.unit_ids, sorting.unit_ids)

    
    


    
if __name__ == '__main__':
    test_remove_redundant_units()
