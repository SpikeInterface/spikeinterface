import unittest
import pytest
from pathlib import Path

from spikeinterface import load_extractor, generate_ground_truth_recording
from spikeinterface.sorters import Kilosort4Sorter, run_sorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not Kilosort4Sorter.is_installed(), reason="kilosort4 not installed")
class Kilosort4SorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Kilosort4Sorter

    # 4 channels is to few for KS4
    def setUp(self):
        if (self.cache_folder / "rec").is_dir():
            recording = load_extractor(self.cache_folder / "rec")
        else:
            recording, _ = generate_ground_truth_recording(num_channels=32, durations=[60], seed=0)
            recording = recording.save(folder=self.cache_folder / "rec", verbose=False, format="binary")
        self.recording = recording
        print(self.recording)

    def test_with_run_skip_correction(self):
        recording = self.recording

        sorter_name = self.SorterClass.sorter_name

        output_folder = self.cache_folder / sorter_name

        sorter_params = self.SorterClass.default_params()
        sorter_params["do_correction"] = False

        sorting = run_sorter(
            sorter_name,
            recording,
            output_folder=output_folder,
            remove_existing_folder=True,
            delete_output_folder=True,
            verbose=False,
            raise_error=True,
            **sorter_params,
        )
        assert sorting.sorting_info is not None
        assert "recording" in sorting.sorting_info.keys()
        assert "params" in sorting.sorting_info.keys()
        assert "log" in sorting.sorting_info.keys()

        del sorting
        # test correct deletion of sorter folder, but not run metadata
        assert not (output_folder / "sorter_output").is_dir()
        assert (output_folder / "spikeinterface_recording.json").is_file()
        assert (output_folder / "spikeinterface_params.json").is_file()
        assert (output_folder / "spikeinterface_log.json").is_file()

    def test_with_run_skip_preprocessing(self):
        from spikeinterface.preprocessing import whiten

        recording = self.recording

        sorter_name = self.SorterClass.sorter_name

        output_folder = self.cache_folder / sorter_name

        sorter_params = self.SorterClass.default_params()
        sorter_params["skip_kilosort_preprocessing"] = True
        recording = whiten(recording)

        sorting = run_sorter(
            sorter_name,
            recording,
            output_folder=output_folder,
            remove_existing_folder=True,
            delete_output_folder=True,
            verbose=False,
            raise_error=True,
            **sorter_params,
        )
        assert sorting.sorting_info is not None
        assert "recording" in sorting.sorting_info.keys()
        assert "params" in sorting.sorting_info.keys()
        assert "log" in sorting.sorting_info.keys()

        del sorting
        # test correct deletion of sorter folder, but not run metadata
        assert not (output_folder / "sorter_output").is_dir()
        assert (output_folder / "spikeinterface_recording.json").is_file()
        assert (output_folder / "spikeinterface_params.json").is_file()
        assert (output_folder / "spikeinterface_log.json").is_file()

    def test_with_run_skip_preprocessing_and_correction(self):
        from spikeinterface.preprocessing import whiten

        recording = self.recording

        sorter_name = self.SorterClass.sorter_name

        output_folder = self.cache_folder / sorter_name

        sorter_params = self.SorterClass.default_params()
        sorter_params["skip_kilosort_preprocessing"] = True
        sorter_params["do_correction"] = False
        recording = whiten(recording)

        sorting = run_sorter(
            sorter_name,
            recording,
            output_folder=output_folder,
            remove_existing_folder=True,
            delete_output_folder=True,
            verbose=False,
            raise_error=True,
            **sorter_params,
        )
        assert sorting.sorting_info is not None
        assert "recording" in sorting.sorting_info.keys()
        assert "params" in sorting.sorting_info.keys()
        assert "log" in sorting.sorting_info.keys()

        del sorting
        # test correct deletion of sorter folder, but not run metadata
        assert not (output_folder / "sorter_output").is_dir()
        assert (output_folder / "spikeinterface_recording.json").is_file()
        assert (output_folder / "spikeinterface_params.json").is_file()
        assert (output_folder / "spikeinterface_log.json").is_file()


if __name__ == "__main__":
    test = Kilosort4SorterCommonTestSuite()
    test.setUp()
    test.test_with_run_skip_preprocessing_and_correction()
