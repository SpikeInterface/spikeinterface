import unittest
import pytest

from spikeinterface.sorters import Spykingcircus2Sorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# # This run several tests
@pytest.mark.skipif(not Spykingcircus2Sorter.is_installed(), reason='spyking circus 2 is not installed')
class SpykingCircus2orterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Spykingcircus2Sorter


if __name__ == '__main__':
    test = SpykingCircus2orterCommonTestSuite()
    test.setUp()
    test.test_with_class()
    test.test_with_run()
