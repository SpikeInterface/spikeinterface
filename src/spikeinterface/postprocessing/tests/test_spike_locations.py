import unittest
import numpy as np

from spikeinterface.postprocessing import ComputeSpikeLocations
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite


class SpikeLocationsExtensionTest(AnalyzerExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeSpikeLocations
    extension_function_params_list = [
        dict(
            method="center_of_mass", spike_retriver_kwargs=dict(channel_from_template=True)
        ),  # chunk_size=10000, n_jobs=1,
        dict(method="center_of_mass", spike_retriver_kwargs=dict(channel_from_template=False)),
        dict(
            method="center_of_mass",
        ),
        dict(method="monopolar_triangulation"),  # , chunk_size=10000, n_jobs=1
        dict(method="grid_convolution"),  # , chunk_size=10000, n_jobs=1
    ]


if __name__ == "__main__":
    test = SpikeLocationsExtensionTest()
    test.setUpClass()
    test.test_extension()
