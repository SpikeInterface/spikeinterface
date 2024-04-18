import unittest
import numpy as np

from spikeinterface.postprocessing import ComputeSpikeAmplitudes
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite


class ComputeSpikeAmplitudesTest(AnalyzerExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeSpikeAmplitudes
    extension_function_params_list = [
        dict(),
    ]


if __name__ == "__main__":
    test = ComputeSpikeAmplitudesTest()
    test.setUpClass()
    test.test_extension()

    # for k, sorting_analyzer in test.sorting_analyzers.items():
    #     print(sorting_analyzer)
    #     print(sorting_analyzer.get_extension("spike_amplitudes").data["amplitudes"].shape)
