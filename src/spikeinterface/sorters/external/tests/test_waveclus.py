import os, getpass

if getpass.getuser() == 'samuel':
    waveclus_path = '/home/samuel/Documents/SpikeInterface/wave_clus/'
    os.environ["WAVECLUS_PATH"] = waveclus_path

import unittest
import pytest

from spikeinterface.sorters import WaveClusSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not WaveClusSorter.is_installed(), reason='waveclus not installed')
class WaveClusCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = WaveClusSorter


if __name__ == '__main__':
    test = WaveClusCommonTestSuite()
    test.setUp()
    test.test_with_run()
