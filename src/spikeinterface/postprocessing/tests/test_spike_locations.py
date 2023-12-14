import unittest
import numpy as np

from spikeinterface.postprocessing import SpikeLocationsCalculator

from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite


class SpikeLocationsExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = SpikeLocationsCalculator
    extension_data_names = ["spike_locations"]
    extension_function_kwargs_list = [
        dict(
            method="center_of_mass", chunk_size=10000, n_jobs=1, spike_retriver_kwargs=dict(channel_from_template=True)
        ),
        dict(
            method="center_of_mass", chunk_size=10000, n_jobs=1, spike_retriver_kwargs=dict(channel_from_template=False)
        ),
        dict(method="center_of_mass", chunk_size=10000, n_jobs=1, outputs="by_unit"),
        dict(method="monopolar_triangulation", chunk_size=10000, n_jobs=1, outputs="by_unit"),
        dict(method="monopolar_triangulation", chunk_size=10000, n_jobs=1, outputs="by_unit"),
    ]

    def test_parallel(self):
        locs_mono1 = self.extension_class.get_extension_function()(
            self.we1, method="monopolar_triangulation", chunk_size=10000, n_jobs=1
        )
        locs_mono2 = self.extension_class.get_extension_function()(
            self.we1, method="monopolar_triangulation", chunk_size=10000, n_jobs=2
        )

        assert np.array_equal(locs_mono1[0], locs_mono2[0])


if __name__ == "__main__":
    test = SpikeLocationsExtensionTest()
    test.setUp()
    test.test_extension()
    test.test_parallel()
