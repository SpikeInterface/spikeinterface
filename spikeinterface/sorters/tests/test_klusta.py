import unittest
import pytest


from spikeinterface.sorters import KlustaSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite

# This run several tests
@pytest.mark.skipif(not KlustaSorter.is_installed(), reason='klusta not installed')
class KlustaCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = KlustaSorter



if __name__ == '__main__':
    #~ KlustaCommonTestSuite().test_on_toy()
    KlustaCommonTestSuite().test_with_BinDatRecordingExtractor()
    #~ KlustaCommonTestSuite().test_get_version()
