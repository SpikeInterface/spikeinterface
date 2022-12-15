from spikeinterface.sorters import Kilosort2_5Sorter

import os, getpass

if getpass.getuser() == 'samuel':
    kilosort2_5_path = '/home/samuel/Documents/SpikeInterface/Kilosort2.5'
    Kilosort2_5Sorter.set_kilosort2_5_path(kilosort2_5_path)

import unittest
import pytest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not Kilosort2_5Sorter.is_installed(), reason='kilosort2.5 not installed')
class Kilosort2_5CommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Kilosort2_5Sorter


if __name__ == '__main__':
    test = Kilosort2_5CommonTestSuite()
    test.setUp()
    test.test_with_run()
