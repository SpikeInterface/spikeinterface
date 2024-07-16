import pytest
from pathlib import Path
import numpy as np


from spikeinterface.core import (
    generate_ground_truth_recording,
    create_sorting_analyzer,
    NumpySorting,
    aggregate_units,
)


from spikeinterface.qualitymetrics import (
    compute_quality_metrics,
)


job_kwargs = dict(n_jobs=2, progress_bar=True, chunk_duration="1s")


def get_sorting_analyzer(seed=2205):
    # we need high firing rate for amplitude_cutoff
    recording, sorting = generate_ground_truth_recording(
        durations=[
            120.0,
        ],
        sampling_frequency=30_000.0,
        num_channels=6,
        num_units=10,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        generate_unit_locations_kwargs=dict(
            margin_um=5.0,
            minimum_z=5.0,
            maximum_z=20.0,
        ),
        generate_templates_kwargs=dict(
            unit_params=dict(
                alpha=(200.0, 500.0),
            )
        ),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=seed,
    )

    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=True)

    sorting_analyzer.compute("random_spikes", max_spikes_per_unit=300, seed=seed)
    sorting_analyzer.compute("noise_levels")
    sorting_analyzer.compute("waveforms", **job_kwargs)
    sorting_analyzer.compute("templates")
    sorting_analyzer.compute("spike_amplitudes", **job_kwargs)

    return sorting_analyzer


@pytest.fixture(scope="module")
def sorting_analyzer_simple():
    sorting_analyzer = get_sorting_analyzer(seed=2205)
    return sorting_analyzer


def test_compute_quality_metrics(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    print(sorting_analyzer)

    # without PCs
    metrics = compute_quality_metrics(
        sorting_analyzer,
        metric_names=["snr"],
        qm_params=dict(isi_violation=dict(isi_threshold_ms=2)),
        skip_pc_metrics=True,
        seed=2205,
    )
    # print(metrics)

    qm = sorting_analyzer.get_extension("quality_metrics")
    assert qm.params["qm_params"]["isi_violation"]["isi_threshold_ms"] == 2
    assert "snr" in metrics.columns
    assert "isolation_distance" not in metrics.columns

    # with PCs
    sorting_analyzer.compute("principal_components")
    metrics = compute_quality_metrics(
        sorting_analyzer,
        metric_names=None,
        qm_params=dict(isi_violation=dict(isi_threshold_ms=2)),
        skip_pc_metrics=False,
        seed=2205,
    )
    print(metrics.columns)
    assert "isolation_distance" in metrics.columns


def test_compute_quality_metrics_recordingless(sorting_analyzer_simple):

    sorting_analyzer = sorting_analyzer_simple
    metrics = compute_quality_metrics(
        sorting_analyzer,
        metric_names=None,
        qm_params=dict(isi_violation=dict(isi_threshold_ms=2)),
        skip_pc_metrics=False,
        seed=2205,
    )

    # make a copy and make it recordingless
    sorting_analyzer_norec = sorting_analyzer.save_as(format="memory")
    sorting_analyzer_norec.delete_extension("quality_metrics")
    sorting_analyzer_norec._recording = None
    assert not sorting_analyzer_norec.has_recording()

    metrics_norec = compute_quality_metrics(
        sorting_analyzer_norec,
        metric_names=None,
        qm_params=dict(isi_violation=dict(isi_threshold_ms=2)),
        skip_pc_metrics=False,
        seed=2205,
    )

    for metric_name in metrics.columns:
        if metric_name == "sd_ratio":
            # this one need recording!!!
            continue
        assert np.allclose(metrics[metric_name].values, metrics_norec[metric_name].values, rtol=1e-02)


def test_empty_units(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple

    empty_spike_train = np.array([], dtype="int64")
    empty_sorting = NumpySorting.from_unit_dict(
        {100: empty_spike_train, 200: empty_spike_train, 300: empty_spike_train},
        sampling_frequency=sorting_analyzer.sampling_frequency,
    )
    sorting_empty = aggregate_units([sorting_analyzer.sorting, empty_sorting])
    assert len(sorting_empty.get_empty_unit_ids()) == 3

    sorting_analyzer_empty = create_sorting_analyzer(sorting_empty, sorting_analyzer.recording, format="memory")
    sorting_analyzer_empty.compute("random_spikes", max_spikes_per_unit=300, seed=2205)
    sorting_analyzer_empty.compute("noise_levels")
    sorting_analyzer_empty.compute("waveforms", **job_kwargs)
    sorting_analyzer_empty.compute("templates")
    sorting_analyzer_empty.compute("spike_amplitudes", **job_kwargs)

    metrics_empty = compute_quality_metrics(
        sorting_analyzer_empty,
        metric_names=None,
        qm_params=dict(isi_violation=dict(isi_threshold_ms=2)),
        skip_pc_metrics=True,
        seed=2205,
    )

    for empty_unit_id in sorting_empty.get_empty_unit_ids():
        from pandas import isnull

        assert np.all(isnull(metrics_empty.loc[empty_unit_id].values))


# TODO @alessio all theses old test should be moved in test_metric_functions.py or test_pca_metrics()

#     def test_amplitude_cutoff(self):
#         we = self.we_short
#         _ = compute_spike_amplitudes(we, peak_sign="neg")

#         # If too few spikes, should raise a warning and set amplitude cutoffs to nans
#         with pytest.warns(UserWarning) as w:
#             metrics = self.extension_class.get_extension_function()(
#                 we, metric_names=["amplitude_cutoff"], peak_sign="neg"
#             )
#         assert all(np.isnan(cutoff) for cutoff in metrics["amplitude_cutoff"].values)

#         # now we decrease the number of bins and check that amplitude cutoffs are correctly computed
#         qm_params = dict(amplitude_cutoff=dict(num_histogram_bins=5))
#         with warnings.catch_warnings():
#             warnings.simplefilter("error")
#             metrics = self.extension_class.get_extension_function()(
#                 we, metric_names=["amplitude_cutoff"], peak_sign="neg", qm_params=qm_params
#             )
#         assert all(not np.isnan(cutoff) for cutoff in metrics["amplitude_cutoff"].values)

#     def test_presence_ratio(self):
#         we = self.we_long

#         total_duration = we.get_total_duration()
#         # If bin_duration_s is larger than total duration, should raise a warning and set presence ratios to nans
#         qm_params = dict(presence_ratio=dict(bin_duration_s=total_duration + 1))
#         with pytest.warns(UserWarning) as w:
#             metrics = self.extension_class.get_extension_function()(
#                 we, metric_names=["presence_ratio"], qm_params=qm_params
#             )
#         assert all(np.isnan(ratio) for ratio in metrics["presence_ratio"].values)

#         # now we decrease the bin_duration_s and check that presence ratios are correctly computed
#         qm_params = dict(presence_ratio=dict(bin_duration_s=total_duration // 10))
#         with warnings.catch_warnings():
#             warnings.simplefilter("error")
#             metrics = self.extension_class.get_extension_function()(
#                 we, metric_names=["presence_ratio"], qm_params=qm_params
#             )
#         assert all(not np.isnan(ratio) for ratio in metrics["presence_ratio"].values)

#     def test_drift_metrics(self):
#         we = self.we_long  # is also multi-segment

#         # if spike_locations is not an extension, raise a warning and set values to NaN
#         with pytest.warns(UserWarning) as w:
#             metrics = self.extension_class.get_extension_function()(we, metric_names=["drift"])
#         assert all(np.isnan(metric) for metric in metrics["drift_ptp"].values)
#         assert all(np.isnan(metric) for metric in metrics["drift_std"].values)
#         assert all(np.isnan(metric) for metric in metrics["drift_mad"].values)

#         # now we compute spike locations, but use an interval_s larger than half the total duration
#         _ = compute_spike_locations(we)
#         total_duration = we.get_total_duration()
#         qm_params = dict(drift=dict(interval_s=total_duration // 2 + 1, min_spikes_per_interval=10, min_num_bins=2))
#         with pytest.warns(UserWarning) as w:
#             metrics = self.extension_class.get_extension_function()(we, metric_names=["drift"], qm_params=qm_params)
#         assert all(np.isnan(metric) for metric in metrics["drift_ptp"].values)
#         assert all(np.isnan(metric) for metric in metrics["drift_std"].values)
#         assert all(np.isnan(metric) for metric in metrics["drift_mad"].values)

#         # finally let's use an interval compatible with segment durations
#         qm_params = dict(drift=dict(interval_s=total_duration // 10, min_spikes_per_interval=10))
#         with warnings.catch_warnings():
#             warnings.simplefilter("error")
#             metrics = self.extension_class.get_extension_function()(we, metric_names=["drift"], qm_params=qm_params)
#         # print(metrics)
#         assert all(not np.isnan(metric) for metric in metrics["drift_ptp"].values)
#         assert all(not np.isnan(metric) for metric in metrics["drift_std"].values)
#         assert all(not np.isnan(metric) for metric in metrics["drift_mad"].values)

#     def test_peak_sign(self):
#         we = self.we_long
#         rec = we.recording
#         sort = we.sorting

#         # invert recording
#         rec_inv = scale(rec, gain=-1.0)

#         we_inv = extract_waveforms(rec_inv, sort, cache_folder / "toy_waveforms_inv", seed=0)

#         # compute amplitudes
#         _ = compute_spike_amplitudes(we, peak_sign="neg")
#         _ = compute_spike_amplitudes(we_inv, peak_sign="pos")

#         # without PC
#         metrics = self.extension_class.get_extension_function()(
#             we, metric_names=["snr", "amplitude_cutoff"], peak_sign="neg"
#         )
#         metrics_inv = self.extension_class.get_extension_function()(
#             we_inv, metric_names=["snr", "amplitude_cutoff"], peak_sign="pos"
#         )
#         # print(metrics)
#         # print(metrics_inv)
#         # for SNR we allow a 5% tollerance because of waveform sub-sampling
#         assert np.allclose(metrics["snr"].values, metrics_inv["snr"].values, rtol=0.05)
#         # for amplitude_cutoff, since spike amplitudes are computed, values should be exactly the same
#         assert np.allclose(metrics["amplitude_cutoff"].values, metrics_inv["amplitude_cutoff"].values, atol=1e-3)

#     def test_nn_metrics(self):
#         we_dense = self.we1
#         we_sparse = self.we_sparse
#         sparsity = self.sparsity1
#         # print(sparsity)

#         metric_names = ["nearest_neighbor", "nn_isolation", "nn_noise_overlap"]

#         # with external sparsity on dense waveforms
#         _ = compute_principal_components(we_dense, n_components=5, mode="by_channel_local")
#         metrics = self.extension_class.get_extension_function()(
#             we_dense, metric_names=metric_names, sparsity=sparsity, seed=0
#         )
#         # print(metrics)

#         # with sparse waveforms
#         _ = compute_principal_components(we_sparse, n_components=5, mode="by_channel_local")
#         metrics = self.extension_class.get_extension_function()(
#             we_sparse, metric_names=metric_names, sparsity=None, seed=0
#         )
#         # print(metrics)

#         # with 2 jobs
#         # with sparse waveforms
#         _ = compute_principal_components(we_sparse, n_components=5, mode="by_channel_local")
#         metrics_par = self.extension_class.get_extension_function()(
#             we_sparse, metric_names=metric_names, sparsity=None, seed=0, n_jobs=2
#         )
#         for metric_name in metrics.columns:
#             # NaNs are skipped
#             assert np.allclose(metrics[metric_name].dropna(), metrics_par[metric_name].dropna())

if __name__ == "__main__":

    sorting_analyzer = get_sorting_analyzer()
    print(sorting_analyzer)

    test_compute_quality_metrics(sorting_analyzer)
    test_compute_quality_metrics_recordingless(sorting_analyzer)
    test_empty_units(sorting_analyzer)
