import unittest
import pytest

from spikeinterface.sorters import SpikeInterfaceSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# # This run several tests
@pytest.mark.skipif(not SpikeInterfaceSorter.is_installed(), reason='spikeinterface not installed')
class SpikeInterfaceSorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = SpikeInterfaceSorter


if __name__ == '__main__':
    test = SpikeInterfaceSorterCommonTestSuite()
    test.setUp()
    test.test_with_class()
    test.test_with_run()
