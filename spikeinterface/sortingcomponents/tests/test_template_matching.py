import pytest
import numpy as np
from pathlib import Path

from spikeinterface import NumpySorting
from spikeinterface import download_dataset
from spikeinterface import extract_waveforms
from spikeinterface.core import get_noise_levels
from spikeinterface.extractors import read_mearec

from spikeinterface.sortingcomponents.matching import find_spikes_from_templates, matching_methods


DEBUG = False

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sortingcomponents"
else:
    cache_folder = Path("cache_folder") / "sortingcomponents"


def test_find_spikes_from_templates():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording, gt_sorting = read_mearec(local_path)

    folder = cache_folder / 'waveforms_mearec'
    we = extract_waveforms(recording, gt_sorting, folder, load_if_exists=True,
                           ms_before=1, ms_after=2., max_spikes_per_unit=500, return_scaled=False,
                           n_jobs=1, chunk_size=10000)
    method_kwargs_all = {
        'waveform_extractor': we,
        'noise_levels': get_noise_levels(recording)
    }
    method_kwargs = {}
    method_kwargs["wobble"] = {
        'templates': we.get_all_templates(),
        'nbefore': we.nbefore,
        'nafter': we.nafter,
    }

    sampling_frequency = recording.get_sampling_frequency()

    result = {}

    for method in matching_methods.keys():
        method_kwargs_ = method_kwargs.get(method, {})
        method_kwargs_.update(method_kwargs_all)
        spikes = find_spikes_from_templates(recording, method=method, method_kwargs=method_kwargs_,
                                            n_jobs=2, chunk_size=1000, progress_bar=True)

        result[method] = NumpySorting.from_times_labels(
            spikes['sample_ind'], spikes['cluster_ind'], sampling_frequency)

    # debug
    if DEBUG:
        import matplotlib.pyplot as plt
        import spikeinterface.full as si
        plt.ion()

        metrics = si.compute_quality_metrics(we, metric_names=['snr'], load_if_exists=True, )

        comparisons = {}
        for method in matching_methods.keys():
            comp = si.compare_sorter_to_ground_truth(gt_sorting, result[method])
            comparisons[method] = comp
            si.plot_agreement_matrix(comp)
            plt.title(method)
            si.plot_sorting_performance(comp, metrics, performance_name='accuracy', metric_name='snr',)
            plt.title(method)
        plt.show()


if __name__ == '__main__':
    test_find_spikes_from_templates()
