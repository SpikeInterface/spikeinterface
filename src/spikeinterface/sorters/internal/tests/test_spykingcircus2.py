import unittest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite
from spikeinterface.sorters import Spykingcircus2Sorter, run_sorter

from pathlib import Path


class SpykingCircus2SorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Spykingcircus2Sorter

    @unittest.skip("performance reason")
    def test_with_numpy_gather(self):
        recording = self.recording
        sorter_name = self.SorterClass.sorter_name
        output_folder = self.cache_folder / sorter_name
        sorter_params = self.SorterClass.default_params()

        sorter_params["matching"]["gather_mode"] = "npy"

        sorting = run_sorter(
            sorter_name,
            recording,
            folder=output_folder,
            remove_existing_folder=True,
            delete_output_folder=False,
            verbose=False,
            raise_error=True,
            **sorter_params,
        )
        assert (output_folder / "sorter_output" / "matching").is_dir()
        assert (output_folder / "sorter_output" / "matching" / "spikes.npy").is_file()


if __name__ == "__main__":
    from spikeinterface import set_global_job_kwargs

    set_global_job_kwargs(n_jobs=1, progress_bar=False)
    test = SpykingCircus2SorterCommonTestSuite()
    test.cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "sorters"
    test.setUp()
    test.test_with_run()
