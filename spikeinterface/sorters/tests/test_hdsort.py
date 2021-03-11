from spikeinterface.sorters import HDSortSorter

import os, getpass
if getpass.getuser() == 'samuel':
    hdsort_path = '/home/samuel/Documents/SpikeInterface/HDsort/'
    HDSortSorter.set_hdsort_path(hdsort_path)

import unittest
import pytest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not HDSortSorter.is_installed(), reason='hdsort not installed')
class HDSortSorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = HDSortSorter


if __name__ == '__main__':
    HDSortSorterCommonTestSuite().test_on_toy()
    #~ HDSortSorterCommonTestSuite().test_with_BinDatRecordingExtractor()
    #~ HDSortSorterCommonTestSuite().test_get_version()
