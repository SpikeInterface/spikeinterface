import pytest
import numpy as np
from pathlib import Path

from spikeinterface import NumpySorting, start_sorting_result, get_noise_levels, compute_sparsity

from spikeinterface.sortingcomponents.matching import find_spikes_from_templates, matching_methods

from spikeinterface.sortingcomponents.tests.common import make_dataset




job_kwargs = dict(n_jobs=-1, chunk_duration="500ms", progress_bar=True)

def get_sorting_result():
    recording, sorting = make_dataset()
    sorting_result = start_sorting_result(sorting, recording, sparse=False)
    sorting_result.select_random_spikes()
    sorting_result.compute("fast_templates", **job_kwargs)
    sorting_result.compute("noise_levels")
    return sorting_result


@pytest.fixture(name="sorting_result", scope="module")
def sorting_result_fixture():
    return get_sorting_result()


@pytest.mark.parametrize("method", matching_methods.keys())
def test_find_spikes_from_templates(method, sorting_result):
    recording = sorting_result.recording
    # waveform = waveform_extractor.get_waveforms(waveform_extractor.unit_ids[0])
    # num_waveforms, _, _ = waveform.shape
    # assert num_waveforms != 0

    templates = sorting_result.get_extension("fast_templates").get_data(outputs="Templates")
    sparsity = compute_sparsity(sorting_result, method="snr", threshold=0.5)
    templates = templates.to_sparse(sparsity)

    noise_levels = sorting_result.get_extension("noise_levels").get_data()

    # sorting_result
    method_kwargs_all = {"templates": templates, "noise_levels": noise_levels}
    method_kwargs = {}
    # method_kwargs["wobble"] = {
    #     "templates": waveform_extractor.get_all_templates(),
    #     "nbefore": waveform_extractor.nbefore,
    #     "nafter": waveform_extractor.nafter,
    # }

    sampling_frequency = recording.get_sampling_frequency()

    method_kwargs_ = method_kwargs.get(method, {})
    method_kwargs_.update(method_kwargs_all)
    spikes = find_spikes_from_templates(
        recording, method=method, method_kwargs=method_kwargs_, **job_kwargs
    )

    

    # DEBUG = True
    
    # if DEBUG:
    #     import matplotlib.pyplot as plt
    #     import spikeinterface.full as si

    #     sorting_result.compute("waveforms")
    #     sorting_result.compute("templates")


    #     gt_sorting = sorting_result.sorting

    #     sorting = NumpySorting.from_times_labels(spikes["sample_index"], spikes["cluster_index"], sampling_frequency)
        
    #     metrics = si.compute_quality_metrics(sorting_result, metric_names=["snr"])

    #     fig, ax = plt.subplots()
    #     comp = si.compare_sorter_to_ground_truth(gt_sorting, sorting)
    #     si.plot_agreement_matrix(comp, ax=ax)
    #     ax.set_title(method)
    #     plt.show()


if __name__ == "__main__":
    sorting_result = get_sorting_result()
    # method = "naive"
    # method = "tdc-peeler"
    # method =  "circus"
    # method = "circus-omp-svd"
    method = "wobble"
    test_find_spikes_from_templates(method, sorting_result)

