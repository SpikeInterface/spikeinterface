import pytest
from pathlib import Path
import numpy as np
from copy import deepcopy
import csv
from spikeinterface.core import (
    NumpySorting,
    synthetize_spike_train_bad_isi,
    add_synchrony_to_sorting,
    generate_ground_truth_recording,
    create_sorting_analyzer,
    synthesize_random_firings,
)

from spikeinterface.qualitymetrics.utils import create_ground_truth_pc_distributions

from spikeinterface.qualitymetrics.quality_metric_list import (
    _misc_metric_name_to_func,
)

from spikeinterface.qualitymetrics import (
    get_quality_metric_list,
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
    _get_synchrony_counts,
    compute_quality_metrics,
)


from spikeinterface.core.basesorting import minimum_spike_dtype


job_kwargs = dict(n_jobs=2, progress_bar=True, chunk_duration="1s")


def test_compute_new_quality_metrics(small_sorting_analyzer):
    """
    Computes quality metrics then computes a subset of quality metrics, and checks
    that the old quality metrics are not deleted.
    """

    qm_params = {
        "presence_ratio": {"bin_duration_s": 0.1},
        "amplitude_cutoff": {"num_histogram_bins": 3},
        "firing_range": {"bin_size_s": 1},
    }

    small_sorting_analyzer.compute({"quality_metrics": {"metric_names": ["snr"]}})
    qm_extension = small_sorting_analyzer.get_extension("quality_metrics")
    calculated_metrics = list(qm_extension.get_data().keys())

    assert calculated_metrics == ["snr"]

    small_sorting_analyzer.compute(
        {"quality_metrics": {"metric_names": list(qm_params.keys()), "metric_params": qm_params}}
    )
    small_sorting_analyzer.compute({"quality_metrics": {"metric_names": ["snr"]}})

    quality_metric_extension = small_sorting_analyzer.get_extension("quality_metrics")

    # Check old metrics are not deleted and the new one is added to the data and metadata
    assert set(list(quality_metric_extension.get_data().keys())) == set(
        [
            "amplitude_cutoff",
            "firing_range",
            "presence_ratio",
            "snr",
        ]
    )
    assert set(list(quality_metric_extension.params.get("metric_names"))) == set(
        [
            "amplitude_cutoff",
            "firing_range",
            "presence_ratio",
            "snr",
        ]
    )

    # check that, when parameters are changed, the data and metadata are updated
    old_snr_data = deepcopy(quality_metric_extension.get_data()["snr"].values)
    small_sorting_analyzer.compute(
        {"quality_metrics": {"metric_names": ["snr"], "metric_params": {"snr": {"peak_mode": "peak_to_peak"}}}}
    )
    new_quality_metric_extension = small_sorting_analyzer.get_extension("quality_metrics")
    new_snr_data = new_quality_metric_extension.get_data()["snr"].values

    assert np.all(old_snr_data != new_snr_data)
    assert new_quality_metric_extension.params["metric_params"]["snr"]["peak_mode"] == "peak_to_peak"

    # check that all quality metrics are deleted when parents are recomputed, even after
    # recomputation
    extensions_to_compute = {
        "templates": {"operators": ["average", "median"]},
        "spike_amplitudes": {},
        "spike_locations": {},
        "principal_components": {},
    }

    small_sorting_analyzer.compute(extensions_to_compute)

    assert small_sorting_analyzer.get_extension("quality_metrics") is None


def test_metric_names_in_same_order(small_sorting_analyzer):
    """
    Computes sepecified quality metrics and checks order is propagated.
    """
    specified_metric_names = ["firing_range", "snr", "amplitude_cutoff"]
    small_sorting_analyzer.compute("quality_metrics", metric_names=specified_metric_names)
    qm_keys = small_sorting_analyzer.get_extension("quality_metrics").get_data().keys()
    for i in range(3):
        assert specified_metric_names[i] == qm_keys[i]


def test_save_quality_metrics(small_sorting_analyzer, create_cache_folder):
    """
    Computes quality metrics in binary folder format. Then computes subsets of quality
    metrics and checks if they are saved correctly.
    """

    # can't use _misc_metric_name_to_func as some functions compute several qms
    # e.g. isi_violation and synchrony
    quality_metrics = [
        "num_spikes",
        "firing_rate",
        "presence_ratio",
        "snr",
        "isi_violations_ratio",
        "isi_violations_count",
        "rp_contamination",
        "rp_violations",
        "sliding_rp_violation",
        "amplitude_cutoff",
        "amplitude_median",
        "amplitude_cv_median",
        "amplitude_cv_range",
        "sync_spike_2",
        "sync_spike_4",
        "sync_spike_8",
        "firing_range",
        "drift_ptp",
        "drift_std",
        "drift_mad",
        "sd_ratio",
        "isolation_distance",
        "l_ratio",
        "d_prime",
        "silhouette",
        "nn_hit_rate",
        "nn_miss_rate",
    ]

    small_sorting_analyzer.compute("quality_metrics")

    cache_folder = create_cache_folder
    output_folder = cache_folder / "sorting_analyzer"

    folder_analyzer = small_sorting_analyzer.save_as(format="binary_folder", folder=output_folder)
    quality_metrics_filename = output_folder / "extensions" / "quality_metrics" / "metrics.csv"

    with open(quality_metrics_filename) as metrics_file:
        saved_metrics = csv.reader(metrics_file)
        metric_names = next(saved_metrics)

    for metric_name in quality_metrics:
        assert metric_name in metric_names

    folder_analyzer.compute("quality_metrics", metric_names=["snr"], delete_existing_metrics=False)

    with open(quality_metrics_filename) as metrics_file:
        saved_metrics = csv.reader(metrics_file)
        metric_names = next(saved_metrics)

    for metric_name in quality_metrics:
        assert metric_name in metric_names

    folder_analyzer.compute("quality_metrics", metric_names=["snr"], delete_existing_metrics=True)

    with open(quality_metrics_filename) as metrics_file:
        saved_metrics = csv.reader(metrics_file)
        metric_names = next(saved_metrics)

    for metric_name in quality_metrics:
        if metric_name == "snr":
            assert metric_name in metric_names
        else:
            assert metric_name not in metric_names


def test_unit_structure_in_output(small_sorting_analyzer):

    qm_params = {
        "presence_ratio": {"bin_duration_s": 0.1},
        "amplitude_cutoff": {"num_histogram_bins": 3},
        "amplitude_cv": {"average_num_spikes_per_bin": 7, "min_num_bins": 3},
        "firing_range": {"bin_size_s": 1},
        "isi_violation": {"isi_threshold_ms": 10},
        "drift": {"interval_s": 1, "min_spikes_per_interval": 5},
        "sliding_rp_violation": {"max_ref_period_ms": 50, "bin_size_ms": 0.15},
        "rp_violation": {"refractory_period_ms": 10.0, "censored_period_ms": 0.0},
    }

    for metric_name in get_quality_metric_list():

        try:
            qm_param = qm_params[metric_name]
        except:
            qm_param = {}

        result_all = _misc_metric_name_to_func[metric_name](sorting_analyzer=small_sorting_analyzer, **qm_param)
        result_sub = _misc_metric_name_to_func[metric_name](
            sorting_analyzer=small_sorting_analyzer, unit_ids=["#4", "#9"], **qm_param
        )

        if isinstance(result_all, dict):
            assert list(result_all.keys()) == ["#3", "#9", "#4"]
            assert list(result_sub.keys()) == ["#4", "#9"]
            assert result_sub["#9"] == result_all["#9"]
            assert result_sub["#4"] == result_all["#4"]

        else:
            for result_ind, result in enumerate(result_sub):

                assert list(result_all[result_ind].keys()) == ["#3", "#9", "#4"]
                assert result_sub[result_ind].keys() == set(["#4", "#9"])

                assert result_sub[result_ind]["#9"] == result_all[result_ind]["#9"]
                assert result_sub[result_ind]["#4"] == result_all[result_ind]["#4"]


def test_unit_id_order_independence(small_sorting_analyzer):
    """
    Takes two almost-identical sorting_analyzers, whose unit_ids are in different orders and have different labels,
    and checks that their calculated quality metrics are independent of the ordering and labelling.
    """

    recording = small_sorting_analyzer.recording
    sorting = small_sorting_analyzer.sorting.select_units(["#4", "#9", "#3"], [1, 7, 2])

    small_sorting_analyzer_2 = create_sorting_analyzer(recording=recording, sorting=sorting, format="memory")

    extensions_to_compute = {
        "random_spikes": {"seed": 1205},
        "noise_levels": {"seed": 1205},
        "waveforms": {},
        "templates": {},
        "spike_amplitudes": {},
        "spike_locations": {},
        "principal_components": {},
    }

    small_sorting_analyzer_2.compute(extensions_to_compute)

    # need special params to get non-nan results on a short recording
    qm_params = {
        "presence_ratio": {"bin_duration_s": 0.1},
        "amplitude_cutoff": {"num_histogram_bins": 3},
        "amplitude_cv": {"average_num_spikes_per_bin": 7, "min_num_bins": 3},
        "firing_range": {"bin_size_s": 1},
        "isi_violation": {"isi_threshold_ms": 10},
        "drift": {"interval_s": 1, "min_spikes_per_interval": 5},
        "sliding_rp_violation": {"max_ref_period_ms": 50, "bin_size_ms": 0.15},
    }

    quality_metrics_1 = compute_quality_metrics(
        small_sorting_analyzer, metric_names=get_quality_metric_list(), metric_params=qm_params
    )
    quality_metrics_2 = compute_quality_metrics(
        small_sorting_analyzer_2, metric_names=get_quality_metric_list(), metric_params=qm_params
    )

    for metric, metric_2_data in quality_metrics_2.items():
        assert quality_metrics_1[metric]["#3"] == metric_2_data[2]
        assert quality_metrics_1[metric]["#9"] == metric_2_data[7]
        assert quality_metrics_1[metric]["#4"] == metric_2_data[1]


def _sorting_violation():
    max_time = 100.0
    sampling_frequency = 30000
    trains = [
        synthetize_spike_train_bad_isi(max_time, 10, 2),
        synthetize_spike_train_bad_isi(max_time, 5, 4),
        synthetize_spike_train_bad_isi(max_time, 5, 10),
    ]

    labels = [np.ones((len(trains[i]),), dtype="int") * i for i in range(len(trains))]

    spike_times = np.concatenate(trains)
    spike_labels = np.concatenate(labels)

    order = np.argsort(spike_times)
    max_num_samples = np.floor(max_time * sampling_frequency) - 1
    indexes = np.arange(0, max_time + 1, 1 / sampling_frequency)
    spike_times = np.searchsorted(indexes, spike_times[order], side="left")
    spike_labels = spike_labels[order]
    mask = spike_times < max_num_samples
    spike_times = spike_times[mask]
    spike_labels = spike_labels[mask]

    unit_ids = ["a", "b", "c"]
    sorting = NumpySorting.from_samples_and_labels(spike_times, spike_labels, sampling_frequency, unit_ids=unit_ids)

    return sorting


def _sorting_analyzer_violations():

    sorting = _sorting_violation()
    duration = (sorting.to_spike_vector()["sample_index"][-1] + 1) / sorting.sampling_frequency

    recording, sorting = generate_ground_truth_recording(
        durations=[duration],
        sampling_frequency=sorting.sampling_frequency,
        num_channels=6,
        sorting=sorting,
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=True)
    # this used only for ISI metrics so no need to compute heavy extensions
    return sorting_analyzer


@pytest.fixture(scope="module")
def sorting_analyzer_violations():
    return _sorting_analyzer_violations()


def test_synchrony_counts_no_sync():

    spike_times, spike_units = synthesize_random_firings(num_units=1, duration=1, firing_rates=1.0)

    one_spike = np.zeros(len(spike_times), minimum_spike_dtype)
    one_spike["sample_index"] = spike_times
    one_spike["unit_index"] = spike_units

    sync_count = _get_synchrony_counts(one_spike, np.array([2, 4, 8]), [0])

    assert np.all(sync_count[0] == np.array([0]))


def test_synchrony_counts_one_sync():
    # a spike train containing two synchronized spikes
    spike_indices, spike_labels = synthesize_random_firings(
        num_units=2,
        duration=1,
        firing_rates=1.0,
    )

    added_spikes_indices = [100, 100]
    added_spikes_labels = [1, 0]

    two_spikes = np.zeros(len(spike_indices) + 2, minimum_spike_dtype)
    two_spikes["sample_index"] = np.concatenate((spike_indices, added_spikes_indices))
    two_spikes["unit_index"] = np.concatenate((spike_labels, added_spikes_labels))

    sync_count = _get_synchrony_counts(two_spikes, np.array([2, 4, 8]), [0, 1])

    assert np.all(sync_count[0] == np.array([1, 1]))


def test_synchrony_counts_one_quad_sync():
    # a spike train containing four synchronized spikes
    spike_indices, spike_labels = synthesize_random_firings(
        num_units=4,
        duration=1,
        firing_rates=1.0,
    )

    added_spikes_indices = [100, 100, 100, 100]
    added_spikes_labels = [0, 1, 2, 3]

    four_spikes = np.zeros(len(spike_indices) + 4, minimum_spike_dtype)
    four_spikes["sample_index"] = np.concatenate((spike_indices, added_spikes_indices))
    four_spikes["unit_index"] = np.concatenate((spike_labels, added_spikes_labels))

    sync_count = _get_synchrony_counts(four_spikes, np.array([2, 4, 8]), [0, 1, 2, 3])

    assert np.all(sync_count[0] == np.array([1, 1, 1, 1]))
    assert np.all(sync_count[1] == np.array([1, 1, 1, 1]))


def test_synchrony_counts_not_all_units():
    # a spike train containing two synchronized spikes
    spike_indices, spike_labels = synthesize_random_firings(num_units=3, duration=1, firing_rates=1.0)

    added_spikes_indices = [50, 100, 100]
    added_spikes_labels = [0, 1, 2]

    three_spikes = np.zeros(len(spike_indices) + 3, minimum_spike_dtype)
    three_spikes["sample_index"] = np.concatenate((spike_indices, added_spikes_indices))
    three_spikes["unit_index"] = np.concatenate((spike_labels, added_spikes_labels))

    sync_count = _get_synchrony_counts(three_spikes, np.array([2, 4, 8]), [0, 1, 2])

    assert np.all(sync_count[0] == np.array([0, 1, 1]))


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


def test_calculate_firing_rate_num_spikes(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    firing_rates = compute_firing_rates(sorting_analyzer)
    num_spikes = compute_num_spikes(sorting_analyzer)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # firing_rates_gt = {0: 10.01, 1: 5.03, 2: 5.09}
    # num_spikes_gt = {0: 1001, 1: 503, 2: 509}
    # assert np.allclose(list(firing_rates_gt.values()), list(firing_rates.values()), rtol=0.05)
    # np.testing.assert_array_equal(list(num_spikes_gt.values()), list(num_spikes.values()))


def test_calculate_firing_range(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    firing_ranges = compute_firing_ranges(sorting_analyzer)
    print(firing_ranges)

    with pytest.warns(UserWarning) as w:
        firing_ranges_nan = compute_firing_ranges(
            sorting_analyzer, bin_size_s=sorting_analyzer.get_total_duration() + 1
        )
        assert np.all([np.isnan(f) for f in firing_ranges_nan.values()])


def test_calculate_amplitude_cutoff(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    # spike_amps = sorting_analyzer.get_extension("spike_amplitudes").get_data()
    amp_cuts = compute_amplitude_cutoffs(sorting_analyzer, num_histogram_bins=10)
    # print(amp_cuts)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # amp_cuts_gt = {0: 0.33067210050787543, 1: 0.43482247296942045, 2: 0.43482247296942045}
    # assert np.allclose(list(amp_cuts_gt.values()), list(amp_cuts.values()), rtol=0.05)


def test_calculate_amplitude_median(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    # spike_amps = sorting_analyzer.get_extension("spike_amplitudes").get_data()
    amp_medians = compute_amplitude_medians(sorting_analyzer)
    # print(amp_medians)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # amp_medians_gt = {0: 130.77323354628675, 1: 130.7461997791725, 2: 130.7461997791725}
    # assert np.allclose(list(amp_medians_gt.values()), list(amp_medians.values()), rtol=0.05)


def test_calculate_amplitude_cv_metrics(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    amp_cv_median, amp_cv_range = compute_amplitude_cv_metrics(sorting_analyzer, average_num_spikes_per_bin=20)
    print(amp_cv_median)
    print(amp_cv_range)

    # amps_scalings = compute_amplitude_scalings(sorting_analyzer)
    sorting_analyzer.compute("amplitude_scalings", **job_kwargs)
    amp_cv_median_scalings, amp_cv_range_scalings = compute_amplitude_cv_metrics(
        sorting_analyzer,
        average_num_spikes_per_bin=20,
        amplitude_extension="amplitude_scalings",
        min_num_bins=5,
    )
    print(amp_cv_median_scalings)
    print(amp_cv_range_scalings)


def test_calculate_snrs(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    snrs = compute_snrs(sorting_analyzer)
    print(snrs)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # snrs_gt = {0: 12.92, 1: 12.99, 2: 12.99}
    # assert np.allclose(list(snrs_gt.values()), list(snrs.values()), rtol=0.05)


def test_calculate_presence_ratio(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    ratios = compute_presence_ratios(sorting_analyzer, bin_duration_s=10)
    print(ratios)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # ratios_gt = {0: 1.0, 1: 1.0, 2: 1.0}
    # np.testing.assert_array_equal(list(ratios_gt.values()), list(ratios.values()))


def test_calculate_isi_violations(sorting_analyzer_violations):
    sorting_analyzer = sorting_analyzer_violations
    isi_viol, counts = compute_isi_violations(sorting_analyzer, isi_threshold_ms=1, min_isi_ms=0.0)
    print(isi_viol)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # isi_viol_gt = {0: 0.0998002996004994, 1: 0.7904857139469347, 2: 1.929898371551754}
    # counts_gt = {0: 2, 1: 4, 2: 10}
    # assert np.allclose(list(isi_viol_gt.values()), list(isi_viol.values()), rtol=0.05)
    # np.testing.assert_array_equal(list(counts_gt.values()), list(counts.values()))


def test_calculate_sliding_rp_violations(sorting_analyzer_violations):
    sorting_analyzer = sorting_analyzer_violations
    contaminations = compute_sliding_rp_violations(sorting_analyzer, bin_size_ms=0.25, window_size_s=1)
    print(contaminations)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # contaminations_gt = {0: 0.03, 1: 0.185, 2: 0.325}
    # assert np.allclose(list(contaminations_gt.values()), list(contaminations.values()), rtol=0.05)


def test_calculate_rp_violations(sorting_analyzer_violations):
    sorting_analyzer = sorting_analyzer_violations
    rp_contamination, counts = compute_refrac_period_violations(
        sorting_analyzer, refractory_period_ms=1, censored_period_ms=0.0
    )
    print(rp_contamination, counts)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # counts_gt = {0: 2, 1: 4, 2: 10}
    # rp_contamination_gt = {0: 0.10534956502609294, 1: 1.0, 2: 1.0}
    # assert np.allclose(list(rp_contamination_gt.values()), list(rp_contamination.values()), rtol=0.05)
    # np.testing.assert_array_equal(list(counts_gt.values()), list(counts.values()))

    sorting = NumpySorting.from_unit_dict(
        {0: np.array([28, 150], dtype=np.int16), 1: np.array([], dtype=np.int16)}, 30000
    )
    # we.sorting = sorting
    sorting_analyzer2 = create_sorting_analyzer(sorting, sorting_analyzer.recording, format="memory", sparse=False)

    rp_contamination, counts = compute_refrac_period_violations(
        sorting_analyzer2, refractory_period_ms=1, censored_period_ms=0.0
    )
    assert np.isnan(rp_contamination[1])


def test_synchrony_metrics(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    sorting = sorting_analyzer.sorting
    synchrony_metrics = compute_synchrony_metrics(sorting_analyzer)

    synchrony_sizes = np.array([2, 4, 8])

    # check returns
    for size in synchrony_sizes:
        assert f"sync_spike_{size}" in synchrony_metrics._fields

    # here we test that increasing added synchrony is captured by syncrhony metrics
    added_synchrony_levels = (0.2, 0.5, 0.8)
    previous_sorting_analyzer = sorting_analyzer
    for sync_level in added_synchrony_levels:
        sorting_sync = add_synchrony_to_sorting(sorting, sync_event_ratio=sync_level)
        sorting_analyzer_sync = create_sorting_analyzer(sorting_sync, sorting_analyzer.recording, format="memory")

        previous_synchrony_metrics = compute_synchrony_metrics(previous_sorting_analyzer)
        current_synchrony_metrics = compute_synchrony_metrics(sorting_analyzer_sync)
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
        previous_sorting_analyzer = sorting_analyzer_sync


def test_synchrony_metrics_unit_id_subset(sorting_analyzer_simple):

    unit_ids_subset = [3, 7]

    synchrony_metrics = compute_synchrony_metrics(sorting_analyzer_simple, unit_ids=unit_ids_subset)

    assert list(synchrony_metrics.sync_spike_2.keys()) == [3, 7]
    assert list(synchrony_metrics.sync_spike_4.keys()) == [3, 7]
    assert list(synchrony_metrics.sync_spike_8.keys()) == [3, 7]


def test_synchrony_metrics_no_unit_ids(sorting_analyzer_simple):

    synchrony_metrics = compute_synchrony_metrics(sorting_analyzer_simple)
    assert np.all(list(synchrony_metrics.sync_spike_2.keys()) == sorting_analyzer_simple.unit_ids)


@pytest.mark.sortingcomponents
def test_calculate_drift_metrics(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    sorting_analyzer.compute("spike_locations", **job_kwargs)

    drifts_ptps, drifts_stds, drift_mads = compute_drift_metrics(
        sorting_analyzer, interval_s=10, min_spikes_per_interval=10
    )

    # print(drifts_ptps, drifts_stds, drift_mads)

    # testing method accuracy with magic number is not a good pratcice, I remove this.
    # drift_ptps_gt = {0: 0.7155675636836349, 1: 0.8163672125409391, 2: 1.0224792180505773}
    # drift_stds_gt = {0: 0.17536888672049475, 1: 0.24508522219800638, 2: 0.29252984101193136}
    # drift_mads_gt = {0: 0.06894539993542423, 1: 0.1072587408373451, 2: 0.13237607989318861}
    # assert np.allclose(list(drift_ptps_gt.values()), list(drifts_ptps.values()), rtol=0.05)
    # assert np.allclose(list(drift_stds_gt.values()), list(drifts_stds.values()), rtol=0.05)
    # assert np.allclose(list(drift_mads_gt.values()), list(drift_mads.values()), rtol=0.05)


def test_calculate_sd_ratio(sorting_analyzer_simple):
    sd_ratio = compute_sd_ratio(
        sorting_analyzer_simple,
    )

    assert np.all(list(sd_ratio.keys()) == sorting_analyzer_simple.unit_ids)
    # @aurelien can you check this, this is not working anymore
    # assert np.allclose(list(sd_ratio.values()), 1, atol=0.25, rtol=0)


if __name__ == "__main__":

    sorting_analyzer = _sorting_analyzer_simple()
    print(sorting_analyzer)

    test_unit_structure_in_output(_small_sorting_analyzer())

    # test_calculate_firing_rate_num_spikes(sorting_analyzer)

    # test_calculate_snrs(sorting_analyzer)
    # test_calculate_amplitude_cutoff(sorting_analyzer)
    # test_calculate_presence_ratio(sorting_analyzer)
    # test_calculate_amplitude_median(sorting_analyzer)
    # test_calculate_sliding_rp_violations(sorting_analyzer)
    # test_calculate_drift_metrics(sorting_analyzer)
    # test_synchrony_metrics(sorting_analyzer)
    # test_synchrony_metrics_unit_id_subset(sorting_analyzer)
    # test_synchrony_metrics_no_unit_ids(sorting_analyzer)
    # test_calculate_firing_range(sorting_analyzer)
    # test_calculate_amplitude_cv_metrics(sorting_analyzer)
    # test_calculate_sd_ratio(sorting_analyzer)

    # sorting_analyzer_violations = _sorting_analyzer_violations()
    # print(sorting_analyzer_violations)
    # test_calculate_isi_violations(sorting_analyzer_violations)
    # test_calculate_sliding_rp_violations(sorting_analyzer_violations)
    # test_calculate_rp_violations(sorting_analyzer_violations)
