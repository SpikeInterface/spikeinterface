from spikeinterface.sorters import Kilosort3Sorter

import os, getpass

if getpass.getuser() == 'samuel':
    kilosort3_path = '/home/samuel/Documents/SpikeInterface/Kilosort3'
    Kilosort3Sorter.set_kilosort3_path(kilosort3_path)

import unittest
import pytest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not Kilosort3Sorter.is_installed(), reason='kilosort3 not installed')
class Kilosort3SorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Kilosort3Sorter


if __name__ == '__main__':
    test = Kilosort3SorterCommonTestSuite()
    test.setUp()
    test.test_with_run()
