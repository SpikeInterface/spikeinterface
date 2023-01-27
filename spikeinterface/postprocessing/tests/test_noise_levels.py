import unittest

from spikeinterface.postprocessing import compute_noise_levels, NoiseLevelCalculator
from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite



class NoiseLevelCalculatorExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = NoiseLevelCalculator
    extension_data_names = ["noise_levels"]


if __name__ == '__main__':
    test = NoiseLevelCalculatorExtensionTest()
    test.setUp()
    test.test_extension()
