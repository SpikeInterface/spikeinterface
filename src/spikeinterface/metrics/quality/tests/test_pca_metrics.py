import pytest
import warnings
import numpy as np

from spikeinterface.metrics import compute_quality_metrics, get_quality_pca_metric_list


def test_compute_pc_metrics_multi_processing(small_sorting_analyzer, tmp_path):
    import pandas as pd

    sorting_analyzer = small_sorting_analyzer
    metric_names = get_quality_pca_metric_list()
    metric_params = dict(nn_advanced=dict(seed=2308))
    res1 = compute_quality_metrics(
        sorting_analyzer, metric_names=metric_names, n_jobs=1, progress_bar=True, seed=1205, metric_params=metric_params
    )

    # this should raise a warning, since nn_advanced can be parallelized only if not in memory
    with pytest.warns(UserWarning):
        res2 = compute_quality_metrics(
            sorting_analyzer,
            metric_names=metric_names,
            n_jobs=2,
            progress_bar=True,
            seed=1205,
            metric_params=metric_params,
        )

    # now we cache the analyzer and there should be no warning
    sorting_analyzer_saved = sorting_analyzer.save_as(folder=tmp_path / "analyzer", format="binary_folder")
    # assert no warnings this time
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Falling back to n_jobs=1.")
        res2 = compute_quality_metrics(
            sorting_analyzer_saved, metric_names=metric_names, n_jobs=2, progress_bar=True, seed=1205
        )

    for metric_name in res1.columns:
        values1 = res1[metric_name].values
        values2 = res2[metric_name].values

        if values1.dtype.kind == "f":
            np.testing.assert_almost_equal(values1, values2, decimal=4)
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(nrows=2, share=True)
            # ax =a xs[0]
            # ax.plot(res1[metric_name].values)
            # ax.plot(res2[metric_name].values)
            # ax =a xs[1]
            # ax.plot(res2[metric_name].values - res1[metric_name].values)
            # plt.show()
        else:
            assert np.array_equal(values1, values2)


if __name__ == "__main__":
    from spikeinterface.metrics.tests.conftest import make_small_analyzer

    small_sorting_analyzer = make_small_analyzer()
    test_compute_pc_metrics_multi_processing(small_sorting_analyzer)
