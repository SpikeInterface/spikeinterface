import unittest
import shutil
from pathlib import Path

import pytest



from spikeinterface import download_dataset, extract_waveforms, WaveformExtractor
from spikeinterface.extractors import read_mearec
from spikeinterface.toolkit import compute_template_similarity


def setup_module():
    for folder in ('mearec_waveforms'):
        if Path(folder).is_dir():
            shutil.rmtree(folder)
    
    local_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting = read_mearec(local_path)
    print(recording)
    print(sorting)
    
    we = extract_waveforms(recording, sorting, 'mearec_waveforms',
        ms_before=3., ms_after=4., max_spikes_per_unit=500,
        load_if_exists=True,
        n_jobs=1, chunk_size=30000)


def test_compute_template_similarity():
    we = WaveformExtractor.load_from_folder('mearec_waveforms')
    similarity = compute_template_similarity(we)
    
    # DEBUG
    #~ import matplotlib.pyplot as plt
    #~ fig, ax = plt.subplots()
    #~ ax.imshow(similarity, interpolation='nearest')
    #~ plt.show()

if __name__ == '__main__':
    setup_module()
    test_compute_template_similarity()
