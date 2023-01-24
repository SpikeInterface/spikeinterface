import pytest
import shutil
from pathlib import Path
import numpy as np
from spikeinterface import extract_waveforms, load_waveforms
from spikeinterface.core import synthetize_spike_train_bad_isi
from spikeinterface.extractors.toy_example import toy_example
from spikeinterface.qualitymetrics.utils import create_ground_truth_pc_distributions

from spikeinterface.qualitymetrics import calculate_pc_metrics
from spikeinterface.postprocessing import compute_principal_components, compute_spike_locations, compute_spike_amplitudes

from spikeinterface.qualitymetrics import (mahalanobis_metrics, lda_metrics, nearest_neighbors_metrics, 
        compute_amplitudes_cutoff, compute_presence_ratio, compute_isi_violations, compute_firing_rate, 
        compute_num_spikes, compute_snrs, compute_refrac_period_violations, compute_amplitudes_median,
        compute_drift_metrics)

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "qualitymetrics"
else:
    cache_folder = Path("cache_folder") / "qualitymetrics"


def _simulated_data():
    max_time = 100.0
    sampling_frequency = 30000
    trains = [synthetize_spike_train_bad_isi(max_time, 10, 2),
              synthetize_spike_train_bad_isi(max_time, 5, 4),
              synthetize_spike_train_bad_isi(max_time, 5, 10)]

    labels = [np.ones((len(trains[i]),), dtype='int') * i for i in range(len(trains))]

    spike_times = np.concatenate(trains)
    spike_clusters = np.concatenate(labels)

    order = np.argsort(spike_times)
    max_num_samples = np.floor(max_time * sampling_frequency) - 1
    indexes = np.arange(0, max_time + 1, 1 / sampling_frequency)
    spike_times = np.searchsorted(indexes, spike_times[order], side="left")
    spike_clusters = spike_clusters[order]
    mask = spike_times < max_num_samples
    spike_times = spike_times[mask]
    spike_clusters = spike_clusters[mask]

    return {"duration": max_time, "times": spike_times, "labels": spike_clusters }


def setup_module():
    for folder_name in ('toy_rec', 'toy_sorting', 'toy_waveforms'):
        if (cache_folder / folder_name).is_dir():
            shutil.rmtree(cache_folder / folder_name)

    recording, sorting = toy_example(num_segments=2, num_units=10)
    recording = recording.save(folder=cache_folder / 'toy_rec')
    sorting = sorting.save(folder=cache_folder / 'toy_sorting')

    we = extract_waveforms(recording, sorting, cache_folder / 'toy_waveforms',
                           ms_before=3., ms_after=4., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000)
    pca = compute_principal_components(we, n_components=5, mode='by_channel_local')


def test_calculate_pc_metrics():
    we = load_waveforms(cache_folder / 'toy_waveforms')
    print(we)
    pca = we.load_extension('principal_components')
    print(pca)

    res = calculate_pc_metrics(pca)
    print(res)

def test_mahalanobis_metrics():
    all_pcs1, all_labels1 = create_ground_truth_pc_distributions([1, -1], [1000, 1000])
    all_pcs2, all_labels2 = create_ground_truth_pc_distributions([1, -2],
                                                                 [1000, 1000])  # increase distance between clusters

    isolation_distance1, l_ratio1 = mahalanobis_metrics(all_pcs1, all_labels1, 0)
    isolation_distance2, l_ratio2 = mahalanobis_metrics(all_pcs2, all_labels2, 0)

    assert isolation_distance1 < isolation_distance2
    assert l_ratio1 > l_ratio2


def test_lda_metrics():
    all_pcs1, all_labels1 = create_ground_truth_pc_distributions([1, -1], [1000, 1000])
    all_pcs2, all_labels2 = create_ground_truth_pc_distributions([1, -2],
                                                                 [1000, 1000])  # increase distance between clusters

    d_prime1 = lda_metrics(all_pcs1, all_labels1, 0)
    d_prime2 = lda_metrics(all_pcs2, all_labels2, 0)

    assert d_prime1 < d_prime2


def test_nearest_neighbors_metrics():
    all_pcs1, all_labels1 = create_ground_truth_pc_distributions([1, -1], [1000, 1000])
    all_pcs2, all_labels2 = create_ground_truth_pc_distributions([1, -2],
                                                                 [1000, 1000])  # increase distance between clusters

    hit_rate1, miss_rate1 = nearest_neighbors_metrics(all_pcs1, all_labels1, 0, 1000, 3)
    hit_rate2, miss_rate2 = nearest_neighbors_metrics(all_pcs2, all_labels2, 0, 1000, 3)

    assert hit_rate1 < hit_rate2
    assert miss_rate1 > miss_rate2

@pytest.fixture
def simulated_data():
    return _simulated_data()


def setup_dataset(spike_data, score_detection=1):
    recording, sorting = toy_example(duration=[spike_data["duration"]],
                                     spike_times=[spike_data["times"]],
                                     spike_labels=[spike_data["labels"]],
                                     num_segments=1,
                                     num_units=4,
                                     score_detection=score_detection,
                                     seed=10)
    folder = cache_folder / 'waveform_folder2'
    we = extract_waveforms(recording, sorting, folder,
                           ms_before=3., ms_after=4., max_spikes_per_unit=1000,
                           n_jobs=1, chunk_size=30000, overwrite=True)
    return we


def test_calculate_firing_rate_num_spikes(simulated_data):
    firing_rates_gt = {0: 10.01, 1: 5.03, 2: 5.09}
    num_spikes_gt = {0: 1001,  1: 503,  2: 509}

    we = setup_dataset(simulated_data)
    firing_rates = compute_firing_rate(we)
    num_spikes = compute_num_spikes(we)

    assert np.allclose(list(firing_rates_gt.values()), list(firing_rates.values()), rtol=0.05)
    np.testing.assert_array_equal(list(num_spikes_gt.values()), list(num_spikes.values()))


def test_calculate_amplitude_cutoff(simulated_data):
    amp_cuts_gt = {0: 0.33067210050787543, 1: 0.43482247296942045, 2: 0.43482247296942045}
    we = setup_dataset(simulated_data, score_detection=0.5)
    spike_amps = compute_spike_amplitudes(we)
    amp_cuts = compute_amplitudes_cutoff(we, num_histogram_bins=10)
    assert np.allclose(list(amp_cuts_gt.values()), list(amp_cuts.values()), rtol=0.05)


def test_calculate_amplitude_median(simulated_data):
    amp_medians_gt = {0: 130.77323354628675, 1: 130.7461997791725, 2: 130.7461997791725}
    we = setup_dataset(simulated_data, score_detection=0.5)
    spike_amps = compute_spike_amplitudes(we)
    amp_medians = compute_amplitudes_median(we)
    print(amp_medians)
    assert np.allclose(list(amp_medians_gt.values()), list(amp_medians.values()), rtol=0.05)


def test_calculate_snrs(simulated_data):
    snrs_gt = {0: 12.92, 1: 12.99, 2: 12.99}
    we = setup_dataset(simulated_data, score_detection=0.5)
    snrs = compute_snrs(we)
    print(snrs)
    assert np.allclose(list(snrs_gt.values()), list(snrs.values()), rtol=0.05)


def test_calculate_presence_ratio(simulated_data):
    ratios_gt = {0: 1.0, 1: 1.0, 2: 1.0}
    we = setup_dataset(simulated_data)
    ratios = compute_presence_ratio(we, bin_duration_s=10)
    print(ratios)
    np.testing.assert_array_equal(list(ratios_gt.values()), list(ratios.values()))


def test_calculate_isi_violations(simulated_data):
    isi_viol_gt = {0: 0.0998002996004994, 1: 0.7904857139469347, 2: 1.929898371551754}
    counts_gt = {0: 2, 1: 4, 2: 10}
    we = setup_dataset(simulated_data)
    isi_viol, counts = compute_isi_violations(we, 1, 0.0)

    print(isi_viol)
    assert np.allclose(list(isi_viol_gt.values()), list(isi_viol.values()), rtol=0.05)
    np.testing.assert_array_equal(list(counts_gt.values()), list(counts.values()))


def test_calculate_rp_violations(simulated_data):
    rp_contamination_gt = {0: 0.10534956502609294, 1: 1.0, 2: 1.0}
    counts_gt = {0: 2, 1: 4, 2: 10}
    we = setup_dataset(simulated_data)
    rp_contamination, counts = compute_refrac_period_violations(we, 1, 0.0)

    print(rp_contamination)
    assert np.allclose(list(rp_contamination_gt.values()), list(rp_contamination.values()), rtol=0.05)
    np.testing.assert_array_equal(list(counts_gt.values()), list(counts.values()))


def test_calculate_drift_metrics(simulated_data):
    drift_ptps_gt = {0: 3.8497035992743918, 1: 1.200316354668118, 2: 1.3330619152472707}
    drift_stds_gt = {0: 1.0907827238707128, 1: 0.3363447300999075, 2: 0.3607988107268864}
    drift_mads_gt = {0: 0.6769978363913438, 1: 0.2606798893916917, 2: 0.27395444544960695}

    we = setup_dataset(simulated_data)
    spike_locs = compute_spike_locations(we)
    drifts_ptps, drifts_stds, drift_mads = compute_drift_metrics(we, interval_s=10, min_spikes_per_interval=10)

    print(drifts_ptps, drifts_stds, drift_mads)
    assert np.allclose(list(drift_ptps_gt.values()), list(drifts_ptps.values()), rtol=0.05)
    assert np.allclose(list(drift_stds_gt.values()), list(drifts_stds.values()), rtol=0.05)
    assert np.allclose(list(drift_mads_gt.values()), list(drift_mads.values()), rtol=0.05)


if __name__ == '__main__':
    setup_module()
    sim_data = _simulated_data()
    # test_calculate_amplitude_cutoff(sim_data)
    # test_calculate_presence_ratio(sim_data)
    # test_calculate_amplitude_median(sim_data)
    # test_calculate_isi_violations(sim_data)
    # test_calculate_rp_violations(sim_data)
    test_calculate_drift_metrics(sim_data)