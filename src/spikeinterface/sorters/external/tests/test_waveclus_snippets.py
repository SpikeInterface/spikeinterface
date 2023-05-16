import os
import getpass

if getpass.getuser() == 'localadmin1':
    waveclus_path = 'F:/GitHub/wave_clus_original/'
    os.environ["WAVECLUS_PATH"] = waveclus_path

import unittest
import pytest

from spikeinterface.sorters import WaveClusSnippetsSorter
from spikeinterface.sorters.tests.common_tests import SnippetsSorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not WaveClusSnippetsSorter.is_installed(), reason='waveclus not installed')
class WaveClusSnippetsCommonTestSuite(SnippetsSorterCommonTestSuite, unittest.TestCase):
    SorterClass = WaveClusSnippetsSorter


if __name__ == '__main__':
    test = WaveClusSnippetsCommonTestSuite()
    test.setUp()
    test.test_with_run()
