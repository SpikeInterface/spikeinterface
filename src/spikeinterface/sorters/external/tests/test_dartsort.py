import unittest
import pytest

from spikeinterface.sorters import DartsortSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


@pytest.mark.skipif(not DartsortSorter.is_installed(), reason="dartsort not installed")
class DartsortCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Mountainsort4Sorter


if __name__ == "__main__":
    test = DartsortCommonTestSuite()
    test.setUp()
    test.test_with_run()
