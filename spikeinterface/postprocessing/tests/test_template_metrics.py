import unittest

from spikeinterface import extract_waveforms, WaveformExtractor
from spikeinterface.extractors import toy_example

from spikeinterface.postprocessing import TemplateMetricsCalculator

from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite


class TemplateMetricsExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = TemplateMetricsCalculator
    extension_data_names = ["metrics"]
    extension_function_kwargs_list = [
        dict(),
        dict(upsampling_factor=2)
    ]

    def test_sparse_metrics(self):
        tm_sparse = self.extension_class.get_extension_function()(self.we1,
                                                                  sparsity=self.sparsity1)
        print(tm_sparse)


if __name__ == '__main__':
    test = TemplateMetricsExtensionTest()
    test.setUp()
    test.test_extension()
    test.test_sparse_metrics()