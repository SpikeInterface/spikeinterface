import unittest
import numpy as np

from spikeinterface.postprocessing import SpikeAmplitudesCalculator

from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite


class SpikeAmplitudesExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = SpikeAmplitudesCalculator
    extension_data_names = ["amplitude_segment_0"]
    extension_function_kwargs_list = [
        dict(peak_sign='neg', outputs='concatenated', chunk_size=10000, n_jobs=1),
        dict(peak_sign='neg', outputs='by_unit', chunk_size=10000, n_jobs=1),
    ]

    def test_scaled(self):
        amplitudes_scaled = self.extension_class.get_extension_function()(
            self.we1, peak_sign='neg', outputs='concatenated', chunk_size=10000, n_jobs=1,
            return_scaled=True)
        amplitudes_unscaled = self.extension_class.get_extension_function()(
            self.we1, peak_sign='neg', outputs='concatenated', chunk_size=10000, n_jobs=1,
            return_scaled=False)
        gain = self.we1.recording.get_channel_gains()[0]

        assert np.allclose(amplitudes_scaled[0], amplitudes_unscaled[0] * gain)

    def test_parallel(self):
        amplitudes1 = self.extension_class.get_extension_function()(
            self.we1, peak_sign='neg', load_if_exists=False, outputs='concatenated', chunk_size=10000, n_jobs=1)
        # TODO : fix multi processing for spike amplitudes!!!!!!!
        amplitudes2 = self.extension_class.get_extension_function()(
            self.we1, peak_sign='neg', load_if_exists=False, outputs='concatenated', chunk_size=10000, n_jobs=2)

        assert np.array_equal(amplitudes1[0], amplitudes2[0])


if __name__ == '__main__':
    test = SpikeAmplitudesExtensionTest()
    test.setUp()
    test.test_extension()
    # test.test_scaled()
    # test.test_parallel()
