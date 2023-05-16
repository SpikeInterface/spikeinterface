from spikeinterface.sorters import CombinatoSorter

import os, getpass

if getpass.getuser() == 'samuel':
    combinato_path = '/home/samuel/Documents/SpikeInterface/combinato/'
    CombinatoSorter.set_combinato_path(combinato_path)

import unittest
import pytest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not CombinatoSorter.is_installed(), reason='combinato not installed')
class CombinatoSorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = CombinatoSorter


if __name__ == '__main__':
    test = CombinatoSorterCommonTestSuite()
    test.setUp()
    test.test_with_run()
