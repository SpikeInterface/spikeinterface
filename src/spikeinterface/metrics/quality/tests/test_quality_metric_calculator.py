import pytest
import numpy as np

from spikeinterface.core import (
    generate_ground_truth_recording,
    create_sorting_analyzer,
    NumpySorting,
    aggregate_units,
)

from spikeinterface.metrics.quality.misc_metrics import compute_snrs, compute_drift_metrics


from spikeinterface.metrics import (
    compute_quality_metrics,
)

job_kwargs = dict(n_jobs=2, progress_bar=True, chunk_duration="1s")


def test_warnings_errors_when_missing_deps():
    """
    If the user requests to compute a quality metric which depends on an extension
    that has not been computed, this should error. If the user uses the default
    quality metrics (i.e. they do not explicitly request the specific metrics),
    this should report a warning about which metrics could not be computed.
    We check this behavior in this test.
    """

    recording, sorting = generate_ground_truth_recording()
    analyzer = create_sorting_analyzer(sorting=sorting, recording=recording)

    # user tries to use `compute_snrs` without templates. Should error
    with pytest.raises(ValueError):
        compute_snrs(analyzer)

    # user asks for drift metrics without spike_locations. Should error
    with pytest.raises(ValueError):
        compute_drift_metrics(analyzer)

    # user doesn't specify which metrics to compute. Should return a warning
    # about which metrics have not been computed.
    with pytest.warns(Warning):
        analyzer.compute("quality_metrics")


def test_compute_quality_metrics(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple

    # without PCs
    metrics = compute_quality_metrics(
        sorting_analyzer,
        metric_names=["snr"],
        metric_params=dict(isi_violation=dict(isi_threshold_ms=2)),
        skip_pc_metrics=True,
        seed=2205,
    )
    # print(metrics)

    qm = sorting_analyzer.get_extension("quality_metrics")
    assert qm.params["metric_params"]["isi_violation"]["isi_threshold_ms"] == 2
    assert "snr" in metrics.columns
    assert "isolation_distance" not in metrics.columns

    # with PCs
    sorting_analyzer.compute("principal_components")
    metrics = compute_quality_metrics(
        sorting_analyzer,
        metric_names=None,
        metric_params=dict(isi_violation=dict(isi_threshold_ms=2)),
        skip_pc_metrics=False,
        seed=2205,
    )
    print(metrics.columns)
    assert "isolation_distance" in metrics.columns


def test_merging_quality_metrics(sorting_analyzer_simple):

    sorting_analyzer = sorting_analyzer_simple

    metrics = compute_quality_metrics(
        sorting_analyzer,
        metric_names=None,
        metric_params=dict(isi_violation=dict(isi_threshold_ms=2)),
        skip_pc_metrics=False,
        seed=2205,
    )

    # sorting_analyzer_simple has ten units
    new_sorting_analyzer = sorting_analyzer.merge_units([[0, 1]])
    new_metrics = new_sorting_analyzer.get_extension("quality_metrics").get_data()

    # we should copy over the metrics after merge
    for column in metrics.columns:
        assert column in new_metrics.columns
        # should copy dtype too
        assert metrics[column].dtype == new_metrics[column].dtype

    # 10 units vs 9 units
    assert len(metrics.index) > len(new_metrics.index)


def test_compute_quality_metrics_recordingless(sorting_analyzer_simple):

    sorting_analyzer = sorting_analyzer_simple
    metrics = compute_quality_metrics(
        sorting_analyzer,
        metric_names=None,
        metric_params=dict(isi_violation=dict(isi_threshold_ms=2)),
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
        metric_params=dict(isi_violation=dict(isi_threshold_ms=2)),
        skip_pc_metrics=False,
        seed=2205,
    )

    for metric_name in metrics.columns:
        if metric_name == "sd_ratio":
            # this one need recording!!!
            continue
        assert np.allclose(metrics[metric_name].values, metrics_norec[metric_name].values, rtol=1e-02)


def test_empty_units(sorting_analyzer_simple):
    from pandas import isnull

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
        metric_params=dict(isi_violation=dict(isi_threshold_ms=2)),
        skip_pc_metrics=True,
        seed=2205,
    )

    # test that metrics are either NaN or zero for empty units
    empty_unit_ids = sorting_empty.get_empty_unit_ids()

    for col in metrics_empty.columns:
        all_nans = np.all(isnull(metrics_empty.loc[empty_unit_ids, col].values))
        all_zeros = np.all(metrics_empty.loc[empty_unit_ids, col].values == 0)
        all_neg_ones = np.all(metrics_empty.loc[empty_unit_ids, col].values == -1)
        assert all_nans or all_zeros or all_neg_ones, f"Column {col} failed the empty unit test"


def test_quality_metrics_with_periods():
    """
    Test that quality metrics can be computed using valid unit periods.
    """
    from spikeinterface.core.base import unit_period_dtype

    recording, sorting = generate_ground_truth_recording()
    sorting_analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format="memory")

    # compute dependencies
    sorting_analyzer.compute(["random_spikes", "templates", "amplitude_scalings", "valid_unit_periods"], **job_kwargs)
    print(sorting_analyzer)

    # compute quality metrics using valid periods
    metrics = compute_quality_metrics(
        sorting_analyzer,
        metric_names=None,
        skip_pc_metrics=True,
        use_valid_periods=True,
        seed=2205,
    )
    print(metrics)

    # test with external periods: 1 period per segment from 10 to 90% of recording
    num_segments = recording.get_num_segments()
    periods = np.zeros(len(sorting.unit_ids) * num_segments, dtype=unit_period_dtype)
    for i, unit_id in enumerate(sorting.unit_ids):
        unit_index = sorting.id_to_index(unit_id)
        for segment_index in range(num_segments):
            num_samples = recording.get_num_samples(segment_index=segment_index)
            idx = i * num_segments + segment_index
            periods[idx]["unit_index"] = unit_index
            period_start = int(num_samples * 0.1)
            period_end = int(num_samples * 0.9)
            periods[idx]["start_sample_index"] = period_start
            periods[idx]["end_sample_index"] = period_end
            periods[idx]["segment_index"] = segment_index

    metrics_ext_periods = compute_quality_metrics(
        sorting_analyzer,
        metric_names=None,
        skip_pc_metrics=True,
        use_valid_periods=False,
        periods=periods,
        seed=2205,
    )

    # test failure when both periods and use_valid_periods are set
    with pytest.raises(ValueError):
        compute_quality_metrics(
            sorting_analyzer,
            metric_names=None,
            skip_pc_metrics=True,
            use_valid_periods=True,
            periods=periods,
            seed=2205,
        )

    # test failure if use valid_periods is True but valid_unit_periods extension is missing
    sorting_analyzer.delete_extension("valid_unit_periods")
    with pytest.raises(AssertionError):
        compute_quality_metrics(
            sorting_analyzer,
            metric_names=None,
            skip_pc_metrics=True,
            use_valid_periods=True,
            seed=2205,
        )


if __name__ == "__main__":

    sorting_analyzer = get_sorting_analyzer()
    print(sorting_analyzer)

    test_compute_quality_metrics(sorting_analyzer)
    test_compute_quality_metrics_recordingless(sorting_analyzer)
    test_empty_units(sorting_analyzer)
