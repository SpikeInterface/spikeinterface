import unittest

from spikeinterface.postprocessing import compute_noise_levels, NoiseLevelsCalculator
from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite


class NoiseLevelsCalculatorExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = NoiseLevelsCalculator
    extension_data_names = ["noise_levels"]

    exact_same_content = False


if __name__ == "__main__":
    test = NoiseLevelsCalculatorExtensionTest()
    test.setUp()
    test.test_extension()
