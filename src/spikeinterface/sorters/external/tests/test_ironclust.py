import os, getpass

if getpass.getuser() == 'samuel':
    ironclust_path = '/home/samuel/Documents/SpikeInterface/ironclust/'
    os.environ["IRONCLUST_PATH"] = ironclust_path

import unittest
import pytest

from spikeinterface.sorters import IronClustSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not IronClustSorter.is_installed(), reason='ironclust not installed')
class IronclustCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = IronClustSorter


if __name__ == '__main__':
    test = IronclustCommonTestSuite()
    test.setUp()
    test.test_with_run()
