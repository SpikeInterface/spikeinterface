import unittest
import pytest

from spikeinterface.sorters import RTSortSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


@pytest.mark.skipif(not RTSortSorter.is_installed(), reason="rt-sort not installed")
class RTSortSorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = RTSortSorter


if __name__ == "__main__":
    test = RTSortSorterCommonTestSuite()
    test.setUp()
    test.test_with_run()
