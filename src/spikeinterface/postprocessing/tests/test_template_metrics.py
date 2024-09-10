from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeTemplateMetrics
import pytest
import csv


def test_compute_new_template_metrics(small_sorting_analyzer):
    """
    Computes template metrics then computes a subset of template metrics, and checks
    that the old template metrics are not deleted.

    Then computes template metrics with new parameters and checks that old metrics
    are deleted.
    """

    # calculate all template metrics
    small_sorting_analyzer.compute("template_metrics")

    # calculate just exp_decay - this should not delete the previously calculated metrics
    small_sorting_analyzer.compute({"template_metrics": {"metric_names": ["exp_decay"]}})

    template_metric_extension = small_sorting_analyzer.get_extension("template_metrics")

    # Check old metrics are not deleted and the new one is added to the data and metadata
    assert "exp_decay" in list(template_metric_extension.get_data().keys())
    assert "half_width" in list(template_metric_extension.get_data().keys())

    # check that, when parameters are changed, the old metrics are deleted
    small_sorting_analyzer.compute(
        {"template_metrics": {"metric_names": ["exp_decay"], "metrics_kwargs": {"recovery_window_ms": 0.6}}}
    )

    template_metric_extension = small_sorting_analyzer.get_extension("template_metrics")

    assert "half_width" not in list(template_metric_extension.get_data().keys())

    assert small_sorting_analyzer.get_extension("quality_metrics") is None


def test_save_template_metrics(small_sorting_analyzer, create_cache_folder):
    """
    Computes template metrics in binary folder format. Then computes subsets of template
    metrics and checks if they are saved correctly.
    """

    from spikeinterface.postprocessing.template_metrics import _single_channel_metric_name_to_func

    small_sorting_analyzer.compute("template_metrics")

    cache_folder = create_cache_folder
    output_folder = cache_folder / "sorting_analyzer"

    folder_analyzer = small_sorting_analyzer.save_as(format="binary_folder", folder=output_folder)
    template_metrics_filename = output_folder / "extensions" / "template_metrics" / "metrics.csv"

    with open(template_metrics_filename) as metrics_file:
        saved_metrics = csv.reader(metrics_file)
        metric_names = next(saved_metrics)

    for metric_name in list(_single_channel_metric_name_to_func.keys()):
        assert metric_name in metric_names

    folder_analyzer.compute("template_metrics", metric_names=["half_width"], delete_existing_metrics=False)

    with open(template_metrics_filename) as metrics_file:
        saved_metrics = csv.reader(metrics_file)
        metric_names = next(saved_metrics)

    for metric_name in list(_single_channel_metric_name_to_func.keys()):
        assert metric_name in metric_names

    folder_analyzer.compute("template_metrics", metric_names=["half_width"], delete_existing_metrics=True)

    with open(template_metrics_filename) as metrics_file:
        saved_metrics = csv.reader(metrics_file)
        metric_names = next(saved_metrics)

    for metric_name in list(_single_channel_metric_name_to_func.keys()):
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
