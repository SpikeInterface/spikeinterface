from spikeinterface.sorters import KilosortSorter

import os, getpass

if getpass.getuser() == 'samuel':
    # kilosort_path = '/home/samuel/Documents/SpikeInterface/Kilosort1/'
    kilosort_path = '/home/samuel/Documents/SpikeInterface/KiloSort1_cortex_lab/'
    KilosortSorter.set_kilosort_path(kilosort_path)

import unittest
import pytest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not KilosortSorter.is_installed(), reason='kilosort not installed')
class KilosortCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = KilosortSorter


if __name__ == '__main__':
    test = KilosortSorter()
    test.setUp()
    test.test_with_run()
