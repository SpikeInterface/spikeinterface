
import unittest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite

from spikeinterface.sorters import Tridesclous2Sorter



class Tridesclous2SorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Tridesclous2Sorter


if __name__ == '__main__':
    test = Tridesclous2SorterCommonTestSuite()
    test.setUp()
    test.test_with_run()
