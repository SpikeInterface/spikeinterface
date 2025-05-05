import unittest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite

from spikeinterface.sorters import Spykingcircus2Sorter

from pathlib import Path


class SpykingCircus2SorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Spykingcircus2Sorter


if __name__ == "__main__":
    from spikeinterface import set_global_job_kwargs

    set_global_job_kwargs(n_jobs=1, progress_bar=False)
    test = SpykingCircus2SorterCommonTestSuite()
    test.cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "sorters"
    test.setUp()
    test.test_with_run()
