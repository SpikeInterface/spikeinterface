import unittest

import pytest

from spikeinterface.sorters import VanillaSortSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


@pytest.mark.skipif(not VanillaSortSorter.is_installed(), reason="vanillasort not installed")
class VanillaSortCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = VanillaSortSorter
