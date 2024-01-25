import unittest
import numpy as np


from spikeinterface.postprocessing.tests.common_extension_tests import ResultExtensionCommonTestSuite

from spikeinterface.postprocessing import ComputeAmplitudeScalings



class AmplitudeScalingsExtensionTest(ResultExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeAmplitudeScalings
    extension_function_kwargs_list = [
        dict(),
        dict(ms_before=0.5, ms_after=0.5),
    ]

    def test_scaling_values(self):
        key0 = next(iter(self.sorting_results.keys()))
        sorting_result = self.sorting_results[key0]

        spikes = sorting_result.sorting.to_spike_vector()

        ext = sorting_result.get_extension("amplitude_scalings")
        ext.data["amplitude_scalings"]
        for unit_index, unit_id in enumerate(sorting_result.unit_ids):
            mask = spikes["unit_index"] == unit_index
            scalings = ext.data["amplitude_scalings"][mask]
            median_scaling = np.median(scalings)
            print(unit_index, median_scaling)
            np.testing.assert_array_equal(np.round(median_scaling), 1)


if __name__ == "__main__":
    test = AmplitudeScalingsExtensionTest()
    test.setUp()
    test.test_extension()
    test.test_scaling_values()
