import unittest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite

from spikeinterface.sorters import Tridesclous2Sorter

from pathlib import Path


class Tridesclous2SorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Tridesclous2Sorter


if __name__ == "__main__":
    test = Tridesclous2SorterCommonTestSuite()
    test.cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "sorters"
    test.setUp()
    test.test_with_run()
