import pytest
import numpy as np

from spikeinterface import download_dataset
from spikeinterface import extract_waveforms
from spikeinterface.sortingcomponents import find_spike_from_templates

from spikeinterface.extractors import read_mearec


def test_find_spike_from_templates():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording, gt_sorting = read_mearec(local_path)

    folder = 'waveforms_mearec'
    we = extract_waveforms(recording, gt_sorting, folder, load_if_exists=True,
                           ms_before=1, ms_after=2., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000)

    method_kwargs = {'waveform_extractor' : we}
    spikes = find_spike_from_templates(recording, method='naive', method_kwargs=method_kwargs,
                        n_jobs=1, chunk_size=30000, progress_bar=False)
    print(spikes)


if __name__ == '__main__':
    test_find_spike_from_templates()
