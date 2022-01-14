import pytest
import numpy as np

from spikeinterface import download_dataset
from spikeinterface import extract_waveforms
from spikeinterface.sortingcomponents import find_spikes_from_templates

from spikeinterface.toolkit import get_noise_levels
from spikeinterface.extractors import read_mearec


def test_find_spikes_from_templates():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording, gt_sorting = read_mearec(local_path)

    folder = 'waveforms_mearec'
    we = extract_waveforms(recording, gt_sorting, folder, load_if_exists=True,
                           ms_before=1, ms_after=2., max_spikes_per_unit=500, return_scaled=False,
                           n_jobs=1, chunk_size=10000)
    
    
    method_kwargs = {'waveform_extractor' : we}
    spikes = find_spikes_from_templates(recording, method='naive', method_kwargs=method_kwargs,
                        n_jobs=1, chunk_size=30000, progress_bar=False)

    method_kwargs = {
        'waveform_extractor' : we,
        'noise_levels' : get_noise_levels(recording),
    }
    spikes = find_spikes_from_templates(recording, method='tridesclous', method_kwargs=method_kwargs,
                        n_jobs=1, chunk_size=30000, progress_bar=False)
    
    
    

    # debug
    # import matplotlib.pyplot as plt
    # import spikeinterface.full as si
    # metrics = si.compute_quality_metrics(we, metric_names=['snr'], load_if_exists=True, )
    # sorting = si.NumpySorting.from_times_labels(spikes['sample_ind'], spikes['cluster_ind'], recording.get_sampling_frequency())
    # comp = si.compare_sorter_to_ground_truth(gt_sorting, sorting)
    # si.plot_agreement_matrix(comp)
    # si.plot_sorting_performance(comp, metrics, performance_name='accuracy', metric_name='snr',)
    # plt.show()



if __name__ == '__main__':
    test_find_spikes_from_templates()
