import unittest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite

from spikeinterface.sorters import SimpleSorter


class SimpleSorterSorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = SimpleSorter


if __name__ == "__main__":
    test = SimpleSorterSorterCommonTestSuite()
    test.setUp()
    test.test_with_run()
