import unittest

import pytest
from spikeinterface.extractors import toy_example
from spikeinterface.sorters import TridesclousSorter, run_tridesclous
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not TridesclousSorter.is_installed(), reason='tridesclous not installed')
class TridesclousCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = TridesclousSorter


if __name__ == '__main__':
    test = TridesclousCommonTestSuite()
    test.setUp()
    test.test_with_run()
