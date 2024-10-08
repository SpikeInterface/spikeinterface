import pytest
import numpy as np

from spikeinterface.qualitymetrics import compute_pc_metrics, get_quality_pca_metric_list


def test_calculate_pc_metrics(small_sorting_analyzer):
    import pandas as pd

    sorting_analyzer = small_sorting_analyzer
    res1 = compute_pc_metrics(sorting_analyzer, n_jobs=1, progress_bar=True, seed=1205)
    res1 = pd.DataFrame(res1)

    res2 = compute_pc_metrics(sorting_analyzer, n_jobs=2, progress_bar=True, seed=1205)
    res2 = pd.DataFrame(res2)

    for metric_name in res1.columns:
        if metric_name != "nn_unit_id":
            assert not np.all(np.isnan(res1[metric_name].values))
            assert not np.all(np.isnan(res2[metric_name].values))

        assert np.array_equal(res1[metric_name].values, res2[metric_name].values)


def test_pca_metrics_multi_processing(small_sorting_analyzer):
    sorting_analyzer = small_sorting_analyzer

    metric_names = get_quality_pca_metric_list()
    metric_names.remove("nn_isolation")
    metric_names.remove("nn_noise_overlap")

    print(f"Computing PCA metrics with 1 thread per process")
    res1 = compute_pc_metrics(
        sorting_analyzer, n_jobs=-1, metric_names=metric_names, max_threads_per_process=1, progress_bar=True
    )
    print(f"Computing PCA metrics with 2 thread per process")
    res2 = compute_pc_metrics(
        sorting_analyzer, n_jobs=-1, metric_names=metric_names, max_threads_per_process=2, progress_bar=True
    )
    print("Computing PCA metrics with spawn context")
    res2 = compute_pc_metrics(
        sorting_analyzer, n_jobs=-1, metric_names=metric_names, max_threads_per_process=2, progress_bar=True
    )
