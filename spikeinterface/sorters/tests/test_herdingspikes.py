import unittest
import pytest

from spikeinterface.sorters import HerdingspikesSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(True, reason='test')
@pytest.mark.skipif(not HerdingspikesSorter.is_installed(), reason='herdingspikes not installed')
class HerdingspikesSorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = HerdingspikesSorter


if __name__ == '__main__':
    test = HerdingspikesSorterCommonTestSuite()
    test.setUp()
    test.test_with_class()
    test.test_with_run()
