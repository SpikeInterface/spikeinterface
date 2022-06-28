import pytest
import shutil
from pathlib import Path

from spikeinterface import download_dataset, extract_waveforms, WaveformExtractor
from spikeinterface.extractors import read_mearec
from spikeinterface.postprocessing import compute_template_similarity, check_equal_template_with_distribution_overlap


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"


def setup_module():
    for folder_name in ('mearec_waveforms'):
        if (cache_folder / folder_name).is_dir():
            shutil.rmtree(cache_folder / folder_name)

    local_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting = read_mearec(local_path)
    print(recording)
    print(sorting)

    we = extract_waveforms(recording, sorting, cache_folder / 'mearec_waveforms',
                           ms_before=3., ms_after=4., max_spikes_per_unit=500,
                           load_if_exists=True,
                           n_jobs=1, chunk_size=30000)


def test_compute_template_similarity():
    we = WaveformExtractor.load_from_folder(cache_folder / 'mearec_waveforms')
    similarity = compute_template_similarity(we)

    # DEBUG
    #~ import matplotlib.pyplot as plt
    #~ fig, ax = plt.subplots()
    #~ ax.imshow(similarity, interpolation='nearest')
    #~ plt.show()


def test_check_equal_template_with_distribution_overlap():

    we = WaveformExtractor.load_from_folder(cache_folder / 'mearec_waveforms')
    
    
    for unit_id0 in we.sorting.unit_ids:
        waveforms0 = we.get_waveforms(unit_id0)
        for unit_id1 in we.sorting.unit_ids:
            if unit_id0 == unit_id1:
                continue
            waveforms1 = we.get_waveforms(unit_id1)
            check_equal_template_with_distribution_overlap(waveforms0, waveforms1, debug=False)
            
        
    
    
    
    


if __name__ == '__main__':
    #~ setup_module()
    #~ test_compute_template_similarity()
    test_check_equal_template_with_distribution_overlap()
