import unittest


from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeTemplateMetrics


class TemplateMetricsTest(AnalyzerExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeTemplateMetrics
    extension_function_params_list = [
        dict(),
        dict(upsampling_factor=2),
        dict(include_multi_channel_metrics=True),
    ]


if __name__ == "__main__":
    test = TemplateMetricsTest()
    test.setUpClass()
    test.test_extension()
