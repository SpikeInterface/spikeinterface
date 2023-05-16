import unittest
import pytest

from spikeinterface.sorters import YassSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not YassSorter.is_installed(), reason='yass not installed')
class YassCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = YassSorter


if __name__ == '__main__':
    test = YassCommonTestSuite()
    test.setUp()
    test.test_with_run()
