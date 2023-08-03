import pytest
import numpy as np
import unittest

from spikeinterface.extractors import CompressedBinaryIblExtractor, read_cbin_ibl


from spikeinterface.extractors.tests.common_tests import RecordingCommonTestSuite, SortingCommonTestSuite


class CompressedBinaryIblExtractorTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = CompressedBinaryIblExtractor
    downloads = []
    entities = []


# ~ def test_read_cbin_ibl():
# ~ base_folder = '/media/samuel/dataspikesorting/DataSpikeSorting/olivier_destripe/'
# ~ data_folder = base_folder + '4c04120d-523a-4795-ba8f-49dbb8d9f63a'
# ~ rec = read_cbin_ibl(data_folder)

# ~ import matplotlib.pyplot as plt
# ~ import spikeinterface.widgets as sw
# ~ from probeinterface.plotting import plot_probe
# ~ sw.plot_traces(rec)
# ~ plot_probe(rec.get_probe())
# ~ plt.show()


if __name__ == "__main__":
    # ~ test_read_cbin_ibl()

    test = CompressedBinaryIblExtractorTest()
    test.setUp()
    test.test_open()
