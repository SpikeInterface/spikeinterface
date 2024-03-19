import unittest

from spikeinterface.postprocessing import check_equal_template_with_distribution_overlap, TemplateSimilarityCalculator

from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite

from spikeinterface.core import extract_waveforms
from spikeinterface.extractors import toy_example
from spikeinterface.comparison import compare_templates

import numpy as np


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


def test_compare_multiple_templates_different_units():

    duration = 5
    num_channels = 4

    num_units_1 = 5
    num_units_2 = 10

    rec1, sort1 = toy_example(duration=duration, num_segments=1, num_channels=num_channels, num_units=num_units_1)

    rec2, sort2 = toy_example(duration=duration, num_segments=1, num_channels=num_channels, num_units=num_units_2)

    # compute waveforms
    we1 = extract_waveforms(rec1, sort1, n_jobs=1, mode="memory")
    we2 = extract_waveforms(rec2, sort2, n_jobs=1, mode="memory")

    # paired comparison
    temp_cmp = compare_templates(we1, we2)

    assert np.shape(temp_cmp.agreement_scores) == (num_units_1, num_units_2)


if __name__ == "__main__":
    test = SimilarityExtensionTest()
    test.setUp()
    test.test_extension()
    test.test_check_equal_template_with_distribution_overlap()
    test_compare_multiple_templates_different_units()
