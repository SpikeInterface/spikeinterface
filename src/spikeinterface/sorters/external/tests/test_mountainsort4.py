import unittest
import pytest

from spikeinterface.sorters import Mountainsort4Sorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
# @pytest.mark.skipif(True, reason='travis bug not fixed yet')
@pytest.mark.skipif(not Mountainsort4Sorter.is_installed(), reason='moutainsort4 not installed')
class Mountainsort4CommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Mountainsort4Sorter


if __name__ == '__main__':
    test = Mountainsort4CommonTestSuite()
    test.setUp()
    test.test_with_run()
