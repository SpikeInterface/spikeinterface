import unittest

from spikeinterface.postprocessing import (check_equal_template_with_distribution_overlap,
                                           TemplateSimilarityCalculator)

from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite


class SimilarityExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = TemplateSimilarityCalculator
    extension_data_names = ["similarity"]

    # extend common test 
    def test_check_equal_template_with_distribution_overlap(self):
        we = self.we1
        for unit_id0 in we.unit_ids:
            waveforms0 = we.get_waveforms(unit_id0)
            for unit_id1 in we.unit_ids:
                if unit_id0 == unit_id1:
                    continue
                waveforms1 = we.get_waveforms(unit_id1)
                check_equal_template_with_distribution_overlap(waveforms0, waveforms1)


if __name__ == '__main__':
    test = SimilarityExtensionTest()
    test.setUp()
    test.test_extension()
    test.test_check_equal_template_with_distribution_overlap()
