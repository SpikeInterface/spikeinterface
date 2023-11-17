import unittest
import numpy as np

from spikeinterface import compute_sparsity
from spikeinterface.postprocessing import AmplitudeScalingsCalculator

from spikeinterface.postprocessing.tests.common_extension_tests import (
    WaveformExtensionCommonTestSuite,
)


class AmplitudeScalingsExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = AmplitudeScalingsCalculator
    extension_data_names = ["amplitude_scalings"]
    extension_function_kwargs_list = [
        dict(outputs="concatenated", chunk_size=10000, n_jobs=1),
        dict(outputs="concatenated", chunk_size=10000, n_jobs=1, ms_before=0.5, ms_after=0.5),
        dict(outputs="by_unit", chunk_size=10000, n_jobs=1),
        dict(outputs="concatenated", chunk_size=10000, n_jobs=-1),
        dict(outputs="concatenated", chunk_size=10000, n_jobs=2, ms_before=0.5, ms_after=0.5),
    ]

    def test_scaling_parallel(self):
        scalings1 = self.extension_class.get_extension_function()(
            self.we1,
            outputs="concatenated",
            chunk_size=10000,
            n_jobs=1,
        )
        scalings2 = self.extension_class.get_extension_function()(
            self.we1,
            outputs="concatenated",
            chunk_size=10000,
            n_jobs=2,
        )
        np.testing.assert_array_equal(scalings1, scalings2)

    def test_scaling_values(self):
        scalings1 = self.extension_class.get_extension_function()(
            self.we1,
            outputs="by_unit",
            chunk_size=10000,
            n_jobs=1,
        )
        # since this is GT spikes, the rounded median must be 1
        for u, scalings in scalings1[0].items():
            median_scaling = np.median(scalings)
            print(u, median_scaling)
            np.testing.assert_array_equal(np.round(median_scaling), 1)


if __name__ == "__main__":
    test = AmplitudeScalingsExtensionTest()
    test.setUp()
    test.test_extension()
    # test.test_scaling_values()
    # test.test_scaling_parallel()
