import unittest

import pytest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite

from spikeinterface.sorters import LupinSorter, run_sorter

from pathlib import Path


class LupinSorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = LupinSorter

    # TODO: remove once writing to zarr is supported with zarr>=3 (save_array writes the templates to zarr)
    @pytest.mark.requires_zarr_write
    def test_with_run(self):
        super().test_with_run()


if __name__ == "__main__":
    test = LupinSorterCommonTestSuite()
    test.cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "sorters"
    test.setUp()
    test.test_with_run()
