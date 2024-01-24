import unittest

from spikeinterface.postprocessing.tests.common_extension_tests import ResultExtensionCommonTestSuite
from spikeinterface.postprocessing import compute_noise_levels, ComputeNoiseLevels


class ComputeNoiseLevelsTest(ResultExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeNoiseLevels
    extension_function_kwargs_list = [
        dict(),
    ]


if __name__ == "__main__":
    test = ComputeNoiseLevelsTest()
    test.setUp()
    test.test_extension()
