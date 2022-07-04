import pytest
import shutil
from pathlib import Path

import pytest

import numpy as np

from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms, NumpySorting, set_global_tmp_folder
from spikeinterface.extractors import toy_example

from spikeinterface.curation import remove_redundant_units


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"

set_global_tmp_folder(cache_folder)

def test_remove_redundant_units():
    rec, sorting = toy_example(num_segments=1, duration=[10.])
    
    # sorting to dict
    d = {unit_id: sorting.get_unit_spike_train(unit_id) for unit_id in sorting.unit_ids}
    
    # inject some duplicate
    other_ids = np.arange(np.max(sorting.unit_ids) +1 , np.max(sorting.unit_ids) + 5)
    unit_peak_shifts = dict(zip(other_ids, [-5, -2, +2, +5]))
    for i, unit_id in  enumerate(other_ids):
        d[unit_id] = d[sorting.unit_ids[i]] + unit_peak_shifts[unit_id]
    sorting_with_dup = NumpySorting.from_dict(d, sampling_frequency=sorting.get_sampling_frequency())
    print(sorting_with_dup.unit_ids)

    
    rec = rec.save()
    sorting_with_dup = sorting_with_dup.save()
    wf_folder = cache_folder / 'wf_dup'
    if wf_folder.exists():
        shutil.rmtree(wf_folder)
    we = extract_waveforms(rec, sorting_with_dup, folder=wf_folder)
    print(we)
    
    sorting_clean = remove_redundant_units(we)
    print(sorting_clean)
    
    
    print(sorting_clean.unit_ids)


    
if __name__ == '__main__':
    test_remove_redundant_units()
