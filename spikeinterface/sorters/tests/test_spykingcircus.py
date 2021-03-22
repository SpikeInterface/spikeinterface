import unittest
import pytest

from spikeinterface.sorters import SpykingcircusSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# # This run several tests
@pytest.mark.skipif(not SpykingcircusSorter.is_installed(), reason='spykingcircus not installed')
class SpykingcircusCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = SpykingcircusSorter


if __name__ == '__main__':
    #~ SpykingcircusCommonTestSuite().test_on_toy()
    SpykingcircusCommonTestSuite().test_with_BinDatRecordingExtractor()
    #~ SpykingcircusCommonTestSuite().test_get_version()
