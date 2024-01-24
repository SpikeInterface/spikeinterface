import unittest


from spikeinterface.postprocessing.tests.common_extension_tests import ResultExtensionCommonTestSuite
from spikeinterface.postprocessing import ComputeTemplateMetrics



class TemplateMetricsTest(ResultExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeTemplateMetrics
    extension_function_kwargs_list = [
        dict(),
        dict(upsampling_factor=2),
        dict(include_multi_channel_metrics=True),
    ]

    # def test_sparse_metrics(self):
    #     tm_sparse = self.extension_class.get_extension_function()(self.we1, sparsity=self.sparsity1)
    #     print(tm_sparse)

    # def test_multi_channel_metrics(self):
    #     tm_multi = self.extension_class.get_extension_function()(self.we1, include_multi_channel_metrics=True)
    #     print(tm_multi)


if __name__ == "__main__":
    test = TemplateMetricsTest()
    test.setUp()
    test.test_extension()
    # test.test_extension()
    # test.test_multi_channel_metrics()
