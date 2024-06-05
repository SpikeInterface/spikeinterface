import unittest
import numpy as np

from spikeinterface.postprocessing import ComputeSpikeAmplitudes
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite


class TestComputeSpikeAmplitudes(AnalyzerExtensionCommonTestSuite):
    extension_class = ComputeSpikeAmplitudes
    extension_function_params_list = [
        dict(),
    ]
