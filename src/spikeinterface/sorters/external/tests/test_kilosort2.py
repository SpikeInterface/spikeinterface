from spikeinterface.sorters import Kilosort2Sorter

import os, getpass

if getpass.getuser() == 'samuel':
    kilosort2_path = '/home/samuel/Documents/SpikeInterface/Kilosort2'
    Kilosort2Sorter.set_kilosort2_path(kilosort2_path)

import unittest
import pytest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not Kilosort2Sorter.is_installed(), reason='kilosort2 not installed')
class Kilosort2CommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Kilosort2Sorter


if __name__ == '__main__':
    test = Kilosort2CommonTestSuite()
    test.setUp()
    test.test_with_run()
