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


def waveform_extractor(folder):
    repo = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data"
    remote_path = "mearec/mearec_test_10s.h5"
    local_path = download_dataset(repo=repo, remote_path=remote_path)
    recording, gt_sorting = read_mearec(local_path)

    waveform_extractor = extract_waveforms(
        recording,
        gt_sorting,
        folder,
        overwrite=True,
        ms_before=1,
        ms_after=2.0,
        max_spikes_per_unit=500,
        return_scaled=False,
        n_jobs=1,
        chunk_size=10000,
    )

    return waveform_extractor


@pytest.fixture(name="waveform_extractor", scope="module")
def waveform_extractor_fixture(tmp_path_factory):
    folder = tmp_path_factory.mktemp("my_temp_dir")
    return waveform_extractor(folder)


@pytest.mark.parametrize("method", matching_methods.keys())
def test_find_spikes_from_templates(method, waveform_extractor):
    recording = waveform_extractor._recording
    waveform = waveform_extractor.get_waveforms("#0")
    num_waveforms, _, _ = waveform.shape
    assert num_waveforms != 0
    method_kwargs_all = {"waveform_extractor": waveform_extractor, "noise_levels": get_noise_levels(recording)}
    method_kwargs = {}
    method_kwargs["wobble"] = {
        "templates": waveform_extractor.get_all_templates(),
        "nbefore": waveform_extractor.nbefore,
        "nafter": waveform_extractor.nafter,
    }

    sampling_frequency = recording.get_sampling_frequency()

    result = {}

    method_kwargs_ = method_kwargs.get(method, {})
    method_kwargs_.update(method_kwargs_all)
    spikes = find_spikes_from_templates(
        recording, method=method, method_kwargs=method_kwargs_, n_jobs=2, chunk_size=1000, progress_bar=True
    )

    result[method] = NumpySorting.from_times_labels(spikes["sample_index"], spikes["cluster_index"], sampling_frequency)

    # debug
    if DEBUG:
        import matplotlib.pyplot as plt
        import spikeinterface.full as si

        plt.ion()

        metrics = si.compute_quality_metrics(
            waveform_extractor,
            metric_names=["snr"],
            load_if_exists=True,
        )

        comparisons = {}
        for method in matching_methods.keys():
            comp = si.compare_sorter_to_ground_truth(gt_sorting, result[method])
            comparisons[method] = comp
            si.plot_agreement_matrix(comp)
            plt.title(method)
            si.plot_sorting_performance(
                comp,
                metrics,
                performance_name="accuracy",
                metric_name="snr",
            )
            plt.title(method)
        plt.show()


if __name__ == "__main__":
    import tempfile

    tmp_dir_main = tempfile.mkdtemp()
    waveform_extractor = waveform_extractor(tmp_dir_main)
    method = "wobble"
    test_find_spikes_from_templates(method, waveform_extractor)
