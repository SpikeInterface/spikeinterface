import unittest

import pytest
from spikeinterface.sorters import PyKilosortSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not PyKilosortSorter.is_installed(), reason="pykilosort not installed")
class PyKilosortCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = PyKilosortSorter


if __name__ == "__main__":
    test = PyKilosortCommonTestSuite()
    test.setUp()
    test.test_with_run()
