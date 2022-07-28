import unittest
import pytest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite

from spikeinterface.sorters import Spykingcircus2Sorter, Tridesclous2Sorter



# @pytest.mark.skipif(not Spykingcircus2Sorter.is_installed(), reason='spyking circus 2 is not installed')
class SpykingCircus2SorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Spykingcircus2Sorter


class Tridesclous2SorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Tridesclous2Sorter


if __name__ == '__main__':
    test = SpykingCircus2SorterCommonTestSuite()
    #~ test = Tridesclous2SorterCommonTestSuite()
    test.setUp()
    test.test_with_class()
    test.test_with_run()
