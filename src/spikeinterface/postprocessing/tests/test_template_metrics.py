from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeTemplateMetrics
import pytest


def test_compute_new_template_metrics(small_sorting_analyzer):
    """
    Computes template metrics then computes a subset of quality metrics, and checks
    that the old quality metrics are not deleted.

    Then computes template metrics with new parameters and checks that old metrics
    are deleted.
    """

    small_sorting_analyzer.compute("template_metrics")
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
