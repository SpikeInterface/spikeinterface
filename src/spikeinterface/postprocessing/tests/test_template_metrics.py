from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeTemplateMetrics
import pytest
import csv

from spikeinterface.postprocessing.template_metrics import _single_channel_metric_name_to_func

template_metrics = list(_single_channel_metric_name_to_func.keys())


def test_compute_new_template_metrics(small_sorting_analyzer):
    """
    Computes template metrics then computes a subset of template metrics, and checks
    that the old template metrics are not deleted.

    Then computes template metrics with new parameters and checks that old metrics
    are deleted.
    """

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
        {"template_metrics": {"metric_names": ["exp_decay"], "metrics_kwargs": {"recovery_window_ms": 0.6}}}
    )


def test_metric_names_in_same_order(small_sorting_analyzer):
    """
    Computes sepecified template metrics and checks order is propogated.
    """
    specified_metric_names = ["peak_trough_ratio", "num_negative_peaks", "half_width"]
    small_sorting_analyzer.compute("template_metrics", metric_names=specified_metric_names)
    tm_keys = small_sorting_analyzer.get_extension("template_metrics").get_data().keys()
    for i in range(3):
        assert specified_metric_names[i] == tm_keys[i]


def test_save_template_metrics(small_sorting_analyzer, create_cache_folder):
    """
    Computes template metrics in binary folder format. Then computes subsets of template
    metrics and checks if they are saved correctly.
    """

    small_sorting_analyzer.compute("template_metrics")

    cache_folder = create_cache_folder
    output_folder = cache_folder / "sorting_analyzer"

    folder_analyzer = small_sorting_analyzer.save_as(format="binary_folder", folder=output_folder)
    template_metrics_filename = output_folder / "extensions" / "template_metrics" / "metrics.csv"

    with open(template_metrics_filename) as metrics_file:
        saved_metrics = csv.reader(metrics_file)
        metric_names = next(saved_metrics)

    for metric_name in template_metrics:
        assert metric_name in metric_names

    folder_analyzer.compute("template_metrics", metric_names=["half_width"], delete_existing_metrics=False)

    with open(template_metrics_filename) as metrics_file:
        saved_metrics = csv.reader(metrics_file)
        metric_names = next(saved_metrics)

    for metric_name in template_metrics:
        assert metric_name in metric_names

    folder_analyzer.compute("template_metrics", metric_names=["half_width"], delete_existing_metrics=True)

    with open(template_metrics_filename) as metrics_file:
        saved_metrics = csv.reader(metrics_file)
        metric_names = next(saved_metrics)

    for metric_name in template_metrics:
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
