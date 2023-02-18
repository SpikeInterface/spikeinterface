import unittest

from spikeinterface.postprocessing import NoiseLevelsCalculator, compute_noise_levels
from spikeinterface.postprocessing.tests.common_extension_tests import (
    WaveformExtensionCommonTestSuite,
)


class NoiseLevelsCalculatorExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = NoiseLevelsCalculator
    extension_data_names = ["noise_levels"]


if __name__ == "__main__":
    test = NoiseLevelsCalculatorExtensionTest()
    test.setUp()
    test.test_extension()
