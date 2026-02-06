import pytest

from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.metrics.template import (
    ComputeTemplateMetrics,
    compute_template_metrics,
    get_single_channel_template_metric_names,
)
from spikeinterface.metrics.template.metrics import single_channel_metrics, multi_channel_metrics

template_metrics = get_single_channel_template_metric_names()


def test_different_params_template_metrics(small_sorting_analyzer):
    """
    Computes template metrics using different params, and check that they are
    actually calculated using the different params.
    """
    compute_template_metrics(
        sorting_analyzer=small_sorting_analyzer,
        metric_names=["exp_decay", "spread", "half_width"],
        metric_params={"exp_decay": {"peak_function": "min"}, "spread": {"spread_smooth_um": 15}},
    )

    tm_extension = small_sorting_analyzer.get_extension("template_metrics")
    tm_params = tm_extension.params["metric_params"]

    assert tm_params["exp_decay"]["peak_function"] == "min"
    assert tm_params["spread"]["spread_smooth_um"] == 15


def test_compute_new_template_metrics(small_sorting_analyzer):
    """
    Computes template metrics then computes a subset of template metrics, and checks
    that the old template metrics are not deleted.

    Then computes template metrics with new parameters and checks that old metrics
    are deleted.
    """

    small_sorting_analyzer.delete_extension("template_metrics")

    # calculate just exp_decay
    small_sorting_analyzer.compute({"template_metrics": {"metric_names": ["exp_decay"]}})
    template_metric_extension = small_sorting_analyzer.get_extension("template_metrics")

    assert "exp_decay" in list(template_metric_extension.get_data().keys())
    assert "half_width" not in list(template_metric_extension.get_data().keys())

    # calculate all template metrics
    small_sorting_analyzer.compute("template_metrics")
    # calculate just exp_decay - this should not delete any other metrics
    small_sorting_analyzer.compute({"template_metrics": {"metric_names": ["exp_decay"]}})
    template_metric_extension = small_sorting_analyzer.get_extension("template_metrics")

    set(template_metrics) == set(template_metric_extension.get_data().keys())

    # calculate just exp_decay with delete_existing_metrics
    small_sorting_analyzer.compute(
        {"template_metrics": {"metric_names": ["exp_decay"], "delete_existing_metrics": True}}
    )
    template_metric_extension = small_sorting_analyzer.get_extension("template_metrics")
    computed_metric_names = template_metric_extension.get_data().keys()

    for metric_name in template_metrics:
        if metric_name == "exp_decay":
            assert metric_name in computed_metric_names
        else:
            assert metric_name not in computed_metric_names

    # check that, when parameters are changed, the old metrics are deleted
    small_sorting_analyzer.compute(
        {
            "template_metrics": {
                "metric_names": ["exp_decay"],
                "metric_params": {"recovery_slope": {"recovery_window_ms": 0.6}},
            }
        }
    )


def test_metric_names_in_same_order(small_sorting_analyzer):
    """
    Computes sepecified template metrics and checks order is propagated.
    """
    specified_metric_names = ["peak_trough_ratio", "half_width", "peak_to_valley"]
    small_sorting_analyzer.compute(
        "template_metrics", metric_names=specified_metric_names, delete_existing_metrics=True
    )
    tm_columns = small_sorting_analyzer.get_extension("template_metrics").get_data().columns
    for specified_name, column in zip(specified_metric_names, tm_columns):
        assert specified_name == column


def test_save_template_metrics(small_sorting_analyzer, create_cache_folder):
    """
    Computes template metrics in binary folder format. Then computes subsets of template
    metrics and checks if they are saved correctly.
    """
    import pandas as pd

    column_names = []
    for m in single_channel_metrics:
        column_names.extend(list(m.metric_columns.keys()))
    small_sorting_analyzer.compute("template_metrics")

    cache_folder = create_cache_folder
    output_folder = cache_folder / "sorting_analyzer"

    folder_analyzer = small_sorting_analyzer.save_as(format="binary_folder", folder=output_folder)
    template_metrics_filename = output_folder / "extensions" / "template_metrics" / "metrics.csv"

    saved_metrics = pd.read_csv(template_metrics_filename)
    metric_names = saved_metrics.columns

    for metric_name in column_names:
        assert metric_name in metric_names

    folder_analyzer.compute("template_metrics", metric_names=["half_width"], delete_existing_metrics=False)

    saved_metrics = pd.read_csv(template_metrics_filename)
    metric_names = saved_metrics.columns
    for metric_name in column_names:
        assert metric_name in metric_names

    folder_analyzer.compute("template_metrics", metric_names=["half_width"], delete_existing_metrics=True)

    saved_metrics = pd.read_csv(template_metrics_filename)
    metric_names = saved_metrics.columns
    for metric_name in column_names:
        if metric_name == "half_width":
            assert metric_name in metric_names
        else:
            assert metric_name not in metric_names


class TestTemplateMetrics(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(),
            dict(upsampling_factor=2),
            dict(include_multi_channel_metrics=True),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeTemplateMetrics, params)
