import numpy as np
import pytest

from spikeinterface import create_sorting_analyzer, generate_ground_truth_recording


@pytest.mark.parametrize("sparse", [False, True])
def test_postprocessing_metrics_with_no_units(sparse):
    recording, sorting = generate_ground_truth_recording(durations=[0.5], num_channels=4, num_units=2, seed=42)
    analyzer = create_sorting_analyzer(sorting.select_units([]), recording, sparse=sparse)
    analyzer.compute(["random_spikes", "templates", "noise_levels"], n_jobs=1, progress_bar=False)
    for name in ("correlograms", "isi_histograms"):
        result, bins = analyzer.compute(name, method="numpy").get_data()
        expected_shape = (0, 0, len(bins) - 1) if name == "correlograms" else (0, len(bins) - 1)
        assert result.shape == expected_shape
        assert np.all(np.diff(bins) > 0)
    similarity = analyzer.compute("template_similarity").get_data()
    assert similarity.shape == (0, 0)
    for name, metrics in (
        ("quality_metrics", ["firing_rate", "snr"]),
        ("template_metrics", ["peak_to_trough_duration"]),
    ):
        result = analyzer.compute(name, metric_names=metrics).get_data()
        assert result.shape == (0, len(metrics))
        assert list(result.columns) == metrics
        assert list(result.index) == []
