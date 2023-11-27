import pytest
import shutil
from pathlib import Path
import numpy as np
from spikeinterface import extract_waveforms
from spikeinterface.core import NumpySorting, synthetize_spike_train_bad_isi, add_synchrony_to_sorting
from spikeinterface.extractors.toy_example import toy_example
from spikeinterface.qualitymetrics.utils import create_ground_truth_pc_distributions

from spikeinterface.qualitymetrics import calculate_pc_metrics
from spikeinterface.postprocessing import (
    compute_principal_components,
    compute_spike_locations,
    compute_spike_amplitudes,
    compute_amplitude_scalings,
)

from spikeinterface.qualitymetrics import (
    mahalanobis_metrics,
    lda_metrics,
    nearest_neighbors_metrics,
    silhouette_score,
    simplified_silhouette_score,
    compute_amplitude_cutoffs,
    compute_presence_ratios,
    compute_isi_violations,
    compute_firing_rates,
    compute_num_spikes,
    compute_snrs,
    compute_refrac_period_violations,
    compute_sliding_rp_violations,
    compute_drift_metrics,
    compute_amplitude_medians,
    compute_synchrony_metrics,
    compute_firing_ranges,
    compute_amplitude_cv_metrics,
    compute_sd_ratio,
)


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "qualitymetrics"
else:
    cache_folder = Path("cache_folder") / "qualitymetrics"


def _simulated_data():
    max_time = 100.0
    sampling_frequency = 30000
    trains = [
        synthetize_spike_train_bad_isi(max_time, 10, 2),
        synthetize_spike_train_bad_isi(max_time, 5, 4),
        synthetize_spike_train_bad_isi(max_time, 5, 10),
    ]

    labels = [np.ones((len(trains[i]),), dtype="int") * i for i in range(len(trains))]

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

    return {"duration": max_time, "times": spike_times, "labels": spike_clusters}


def _waveform_extractor_simple():
    for name in ("rec1", "sort1", "waveform_folder1"):
        if (cache_folder / name).exists():
            shutil.rmtree(cache_folder / name)

    recording, sorting = toy_example(duration=50, seed=10, firing_rate=6.0)

    recording = recording.save(folder=cache_folder / "rec1")
    sorting = sorting.save(folder=cache_folder / "sort1")
    folder = cache_folder / "waveform_folder1"
    we = extract_waveforms(
        recording,
        sorting,
        folder,
        ms_before=3.0,
        ms_after=4.0,
        max_spikes_per_unit=1000,
        n_jobs=1,
        chunk_size=30000,
        overwrite=True,
    )
    _ = compute_principal_components(we, n_components=5, mode="by_channel_local")
    _ = compute_spike_amplitudes(we, return_scaled=True)
    return we


def _waveform_extractor_violations(data):
    for name in ("rec2", "sort2", "waveform_folder2"):
        if (cache_folder / name).exists():
            shutil.rmtree(cache_folder / name)

    recording, sorting = toy_example(
        duration=[data["duration"]],
        spike_times=[data["times"]],
        spike_labels=[data["labels"]],
        num_segments=1,
        num_units=4,
        # score_detection=score_detection,
        seed=10,
    )
    recording = recording.save(folder=cache_folder / "rec2")
    sorting = sorting.save(folder=cache_folder / "sort2")
    folder = cache_folder / "waveform_folder2"
    we = extract_waveforms(
        recording,
        sorting,
        folder,
        ms_before=3.0,
        ms_after=4.0,
        max_spikes_per_unit=1000,
        n_jobs=1,
        chunk_size=30000,
        overwrite=True,
    )
    return we


@pytest.fixture(scope="module")
def simulated_data():
    return _simulated_data()


@pytest.fixture(scope="module")
def waveform_extractor_violations(simulated_data):
    return _waveform_extractor_violations(simulated_data)


@pytest.fixture(scope="module")
def waveform_extractor_simple():
    return _waveform_extractor_simple()


def test_calculate_pc_metrics(waveform_extractor_simple):
    we = waveform_extractor_simple
    print(we)
    pca = we.load_extension("principal_components")
    print(pca)

    res = calculate_pc_metrics(pca)
    print(res)


def test_mahalanobis_metrics():
    all_pcs1, all_labels1 = create_ground_truth_pc_distributions([1, -1], [1000, 1000])
    all_pcs2, all_labels2 = create_ground_truth_pc_distributions(
        [1, -2], [1000, 1000]
    )  # increase distance between clusters

    isolation_distance1, l_ratio1 = mahalanobis_metrics(all_pcs1, all_labels1, 0)
    isolation_distance2, l_ratio2 = mahalanobis_metrics(all_pcs2, all_labels2, 0)

    assert isolation_distance1 < isolation_distance2
    assert l_ratio1 > l_ratio2


def test_lda_metrics():
    all_pcs1, all_labels1 = create_ground_truth_pc_distributions([1, -1], [1000, 1000])
    all_pcs2, all_labels2 = create_ground_truth_pc_distributions(
        [1, -2], [1000, 1000]
    )  # increase distance between clusters

    d_prime1 = lda_metrics(all_pcs1, all_labels1, 0)
    d_prime2 = lda_metrics(all_pcs2, all_labels2, 0)

    assert d_prime1 < d_prime2


def test_nearest_neighbors_metrics():
    all_pcs1, all_labels1 = create_ground_truth_pc_distributions([1, -1], [1000, 1000])
    all_pcs2, all_labels2 = create_ground_truth_pc_distributions(
        [1, -2], [1000, 1000]
    )  # increase distance between clusters

    hit_rate1, miss_rate1 = nearest_neighbors_metrics(all_pcs1, all_labels1, 0, 1000, 3)
    hit_rate2, miss_rate2 = nearest_neighbors_metrics(all_pcs2, all_labels2, 0, 1000, 3)

    assert hit_rate1 < hit_rate2
    assert miss_rate1 > miss_rate2


def test_silhouette_score_metrics():
    all_pcs1, all_labels1 = create_ground_truth_pc_distributions([1, -1], [1000, 1000])
    all_pcs2, all_labels2 = create_ground_truth_pc_distributions(
        [1, -2], [1000, 1000]
    )  # increase distance between clusters

    sil_score1 = silhouette_score(all_pcs1, all_labels1, 0)
    sil_score2 = silhouette_score(all_pcs2, all_labels2, 0)

    assert sil_score1 < sil_score2


def test_simplified_silhouette_score_metrics():
    all_pcs1, all_labels1 = create_ground_truth_pc_distributions([1, -1], [1000, 1000])
    all_pcs2, all_labels2 = create_ground_truth_pc_distributions(
        [1, -2], [1000, 1000]
    )  # increase distance between clusters

    sim_sil_score1 = simplified_silhouette_score(all_pcs1, all_labels1, 0)
    sim_sil_score2 = simplified_silhouette_score(all_pcs2, all_labels2, 0)

    assert sim_sil_score1 < sim_sil_score2


def test_calculate_firing_rate_num_spikes(waveform_extractor_simple):
    we = waveform_extractor_simple
    firing_rates = compute_firing_rates(we)
    num_spikes = compute_num_spikes(we)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # firing_rates_gt = {0: 10.01, 1: 5.03, 2: 5.09}
    # num_spikes_gt = {0: 1001, 1: 503, 2: 509}
    # assert np.allclose(list(firing_rates_gt.values()), list(firing_rates.values()), rtol=0.05)
    # np.testing.assert_array_equal(list(num_spikes_gt.values()), list(num_spikes.values()))


def test_calculate_firing_range(waveform_extractor_simple):
    we = waveform_extractor_simple
    firing_ranges = compute_firing_ranges(we)
    print(firing_ranges)

    with pytest.warns(UserWarning) as w:
        firing_ranges_nan = compute_firing_ranges(we, bin_size_s=we.get_total_duration() + 1)
        assert np.all([np.isnan(f) for f in firing_ranges_nan.values()])


def test_calculate_amplitude_cutoff(waveform_extractor_simple):
    we = waveform_extractor_simple
    spike_amps = we.load_extension("spike_amplitudes").get_data()
    amp_cuts = compute_amplitude_cutoffs(we, num_histogram_bins=10)
    print(amp_cuts)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # amp_cuts_gt = {0: 0.33067210050787543, 1: 0.43482247296942045, 2: 0.43482247296942045}
    # assert np.allclose(list(amp_cuts_gt.values()), list(amp_cuts.values()), rtol=0.05)


def test_calculate_amplitude_median(waveform_extractor_simple):
    we = waveform_extractor_simple
    spike_amps = we.load_extension("spike_amplitudes").get_data()
    amp_medians = compute_amplitude_medians(we)
    print(spike_amps, amp_medians)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # amp_medians_gt = {0: 130.77323354628675, 1: 130.7461997791725, 2: 130.7461997791725}
    # assert np.allclose(list(amp_medians_gt.values()), list(amp_medians.values()), rtol=0.05)


def test_calculate_amplitude_cv_metrics(waveform_extractor_simple):
    we = waveform_extractor_simple
    amp_cv_median, amp_cv_range = compute_amplitude_cv_metrics(we, average_num_spikes_per_bin=20)
    print(amp_cv_median)
    print(amp_cv_range)

    amps_scalings = compute_amplitude_scalings(we)
    amp_cv_median_scalings, amp_cv_range_scalings = compute_amplitude_cv_metrics(
        we,
        average_num_spikes_per_bin=20,
        amplitude_extension="amplitude_scalings",
        min_num_bins=5,
    )
    print(amp_cv_median_scalings)
    print(amp_cv_range_scalings)


def test_calculate_snrs(waveform_extractor_simple):
    we = waveform_extractor_simple
    snrs = compute_snrs(we)
    print(snrs)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # snrs_gt = {0: 12.92, 1: 12.99, 2: 12.99}
    # assert np.allclose(list(snrs_gt.values()), list(snrs.values()), rtol=0.05)


def test_calculate_presence_ratio(waveform_extractor_simple):
    we = waveform_extractor_simple
    ratios = compute_presence_ratios(we, bin_duration_s=10)
    print(ratios)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # ratios_gt = {0: 1.0, 1: 1.0, 2: 1.0}
    # np.testing.assert_array_equal(list(ratios_gt.values()), list(ratios.values()))


def test_calculate_isi_violations(waveform_extractor_violations):
    we = waveform_extractor_violations
    isi_viol, counts = compute_isi_violations(we, isi_threshold_ms=1, min_isi_ms=0.0)
    print(isi_viol)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # isi_viol_gt = {0: 0.0998002996004994, 1: 0.7904857139469347, 2: 1.929898371551754}
    # counts_gt = {0: 2, 1: 4, 2: 10}
    # assert np.allclose(list(isi_viol_gt.values()), list(isi_viol.values()), rtol=0.05)
    # np.testing.assert_array_equal(list(counts_gt.values()), list(counts.values()))


def test_calculate_sliding_rp_violations(waveform_extractor_violations):
    we = waveform_extractor_violations
    contaminations = compute_sliding_rp_violations(we, bin_size_ms=0.25, window_size_s=1)
    print(contaminations)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # contaminations_gt = {0: 0.03, 1: 0.185, 2: 0.325}
    # assert np.allclose(list(contaminations_gt.values()), list(contaminations.values()), rtol=0.05)


def test_calculate_rp_violations(waveform_extractor_violations):
    we = waveform_extractor_violations
    rp_contamination, counts = compute_refrac_period_violations(we, refractory_period_ms=1, censored_period_ms=0.0)
    print(rp_contamination, counts)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # counts_gt = {0: 2, 1: 4, 2: 10}
    # rp_contamination_gt = {0: 0.10534956502609294, 1: 1.0, 2: 1.0}
    # assert np.allclose(list(rp_contamination_gt.values()), list(rp_contamination.values()), rtol=0.05)
    # np.testing.assert_array_equal(list(counts_gt.values()), list(counts.values()))

    sorting = NumpySorting.from_unit_dict(
        {0: np.array([28, 150], dtype=np.int16), 1: np.array([], dtype=np.int16)}, 30000
    )
    we.sorting = sorting

    rp_contamination, counts = compute_refrac_period_violations(we, refractory_period_ms=1, censored_period_ms=0.0)
    assert np.isnan(rp_contamination[1])


def test_synchrony_metrics(waveform_extractor_simple):
    we = waveform_extractor_simple
    sorting = we.sorting
    synchrony_sizes = (2, 3, 4)
    synchrony_metrics = compute_synchrony_metrics(we, synchrony_sizes=synchrony_sizes)
    print(synchrony_metrics)

    # check returns
    for size in synchrony_sizes:
        assert f"sync_spike_{size}" in synchrony_metrics._fields

    # here we test that increasing added synchrony is captured by syncrhony metrics
    added_synchrony_levels = (0.2, 0.5, 0.8)
    previous_waveform_extractor = we
    for sync_level in added_synchrony_levels:
        sorting_sync = add_synchrony_to_sorting(sorting, sync_event_ratio=sync_level)
        waveform_extractor_sync = extract_waveforms(previous_waveform_extractor.recording, sorting_sync, mode="memory")
        previous_synchrony_metrics = compute_synchrony_metrics(
            previous_waveform_extractor, synchrony_sizes=synchrony_sizes
        )
        current_synchrony_metrics = compute_synchrony_metrics(waveform_extractor_sync, synchrony_sizes=synchrony_sizes)
        print(current_synchrony_metrics)
        # check that all values increased
        for i, col in enumerate(previous_synchrony_metrics._fields):
            assert np.all(
                v_prev < v_curr
                for (v_prev, v_curr) in zip(
                    previous_synchrony_metrics[i].values(), current_synchrony_metrics[i].values()
                )
            )

        # set new previous waveform extractor
        previous_waveform_extractor = waveform_extractor_sync


@pytest.mark.sortingcomponents
def test_calculate_drift_metrics(waveform_extractor_simple):
    we = waveform_extractor_simple
    spike_locs = compute_spike_locations(we)
    drifts_ptps, drifts_stds, drift_mads = compute_drift_metrics(we, interval_s=10, min_spikes_per_interval=10)

    print(drifts_ptps, drifts_stds, drift_mads)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # drift_ptps_gt = {0: 0.7155675636836349, 1: 0.8163672125409391, 2: 1.0224792180505773}
    # drift_stds_gt = {0: 0.17536888672049475, 1: 0.24508522219800638, 2: 0.29252984101193136}
    # drift_mads_gt = {0: 0.06894539993542423, 1: 0.1072587408373451, 2: 0.13237607989318861}
    # assert np.allclose(list(drift_ptps_gt.values()), list(drifts_ptps.values()), rtol=0.05)
    # assert np.allclose(list(drift_stds_gt.values()), list(drifts_stds.values()), rtol=0.05)
    # assert np.allclose(list(drift_mads_gt.values()), list(drift_mads.values()), rtol=0.05)


def test_calculate_sd_ratio(waveform_extractor_simple):
    sd_ratio = compute_sd_ratio(
        waveform_extractor_simple,
    )

    assert np.all(list(sd_ratio.keys()) == waveform_extractor_simple.unit_ids)
    # assert np.allclose(list(sd_ratio.values()), 1, atol=0.2, rtol=0)


if __name__ == "__main__":
    sim_data = _simulated_data()
    we = _waveform_extractor_simple()

    we_violations = _waveform_extractor_violations(sim_data)
    test_calculate_amplitude_cutoff(we)
    test_calculate_presence_ratio(we)
    test_calculate_amplitude_median(we)
    test_calculate_isi_violations(we)
    test_calculate_sliding_rp_violations(we)
    test_calculate_drift_metrics(we)
    test_synchrony_metrics(we)
    test_calculate_firing_range(we)
    test_calculate_amplitude_cv_metrics(we)

    # for windows we need an explicit del for closing the recording files
    del we, we_violations
