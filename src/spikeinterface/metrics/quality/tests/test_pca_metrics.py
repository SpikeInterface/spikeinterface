import pytest
import warnings
import numpy as np

from spikeinterface.metrics import compute_quality_metrics, get_quality_pca_metric_list


def test_compute_pc_metrics_multi_processing(small_sorting_analyzer, tmp_path):

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
            sorting_analyzer_saved,
            metric_names=metric_names,
            n_jobs=2,
            progress_bar=True,
            seed=1205,
            metric_params=metric_params,
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


def test_compute_pc_metrics_lazy_loading(small_sorting_analyzer, tmp_path):
    """
    PCA metrics (including the multi-processed nn_advanced metric) must give the same
    results whether the SortingAnalyzer is loaded eagerly (extension data fully read into
    RAM) or lazily (extension data kept as memmap/zarr handles). This is the cheapest way
    to reduce baseline memory usage for quality metrics on very large recordings.
    """
    sorting_analyzer = small_sorting_analyzer
    metric_names = get_quality_pca_metric_list()
    metric_params = dict(nn_advanced=dict(seed=2308))

    sorting_analyzer_saved = sorting_analyzer.save_as(folder=tmp_path / "analyzer_lazy", format="binary_folder")

    res_eager = compute_quality_metrics(
        sorting_analyzer_saved,
        metric_names=metric_names,
        n_jobs=2,
        seed=1205,
        metric_params=metric_params,
    )

    from spikeinterface.core import load_sorting_analyzer

    sorting_analyzer_lazy = load_sorting_analyzer(tmp_path / "analyzer_lazy", format="auto", lazy=True)
    res_lazy = compute_quality_metrics(
        sorting_analyzer_lazy,
        metric_names=metric_names,
        n_jobs=2,
        seed=1205,
        metric_params=metric_params,
    )

    for metric_name in res_eager.columns:
        values_eager = res_eager[metric_name].values
        values_lazy = res_lazy[metric_name].values

        if values_eager.dtype.kind == "f":
            np.testing.assert_almost_equal(values_eager, values_lazy, decimal=4)
        else:
            assert np.array_equal(values_eager, values_lazy)

    # a lazy (but not read-only) analyzer is allowed to persist to disk by default.
    # the eager computation above already did this (analyzer is not lazy), so reset first.
    sorting_analyzer_saved.delete_extension("quality_metrics")
    sorting_analyzer_lazy2 = load_sorting_analyzer(tmp_path / "analyzer_lazy", format="auto", lazy=True)
    sorting_analyzer_lazy2.compute("quality_metrics", metric_names=metric_names, seed=1205, metric_params=metric_params)
    reloaded = load_sorting_analyzer(tmp_path / "analyzer_lazy", format="auto", lazy=True)
    assert reloaded.has_extension("quality_metrics")

    # a lazy AND read-only analyzer must not persist to disk, even if asked to explicitly
    sorting_analyzer_saved.delete_extension("quality_metrics")
    sorting_analyzer_lazy_ro = load_sorting_analyzer(
        tmp_path / "analyzer_lazy", format="auto", lazy=True, read_only=True
    )
    sorting_analyzer_lazy_ro.compute(
        "quality_metrics", metric_names=metric_names, seed=1205, metric_params=metric_params, save=True
    )
    reloaded = load_sorting_analyzer(tmp_path / "analyzer_lazy", format="auto", lazy=True)
    assert not reloaded.has_extension("quality_metrics")


if __name__ == "__main__":
    from spikeinterface.metrics.conftest import make_small_analyzer

    small_sorting_analyzer = make_small_analyzer()
    test_compute_pc_metrics_multi_processing(small_sorting_analyzer)
