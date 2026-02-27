import unittest
import pytest

from spikeinterface.sorters import DartsortSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


@pytest.mark.skipif(not DartsortSorter.is_installed(), reason="dartsort not installed")
class DartsortCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = DartsortSorter


if __name__ == "__main__":
    from pathlib import Path
    test = DartsortCommonTestSuite()
    test.cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "sorters"
    test.setUp()
    test.test_with_run()
