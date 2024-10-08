from __future__ import annotations

import pytest
import shutil

from spikeinterface import generate_ground_truth_recording
from spikeinterface.sorters import run_sorter
from spikeinterface.core.snippets_tools import snippets_from_sorting


class SorterCommonTestSuite:
    """
    This class run some basic for a sorter class.
    This is the minimal test suite for each sorter class:
      * run once
    """

    SorterClass = None

    @pytest.fixture(autouse=True)
    def create_cache_folder(self, tmp_path_factory):
        self.cache_folder = tmp_path_factory.mktemp("cache_folder")

    def setUp(self):
        recording, sorting_gt = generate_ground_truth_recording(num_channels=4, durations=[60], seed=0)
        rec_folder = self.cache_folder / "rec"
        if rec_folder.is_dir():
            shutil.rmtree(rec_folder)
        self.recording = recording.save(folder=self.cache_folder / "rec", verbose=False, format="binary")
        print(self.recording)

    def test_with_run(self):
        # some sorter (TDC, KS, KS2, ...) work by default with the raw binary
        # format as input to avoid copy when the recording is already this format

        recording = self.recording

        sorter_name = self.SorterClass.sorter_name

        output_folder = self.cache_folder / sorter_name

        sorter_params = self.SorterClass.default_params()

        sorting = run_sorter(
            sorter_name,
            recording,
            folder=output_folder,
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

    def test_get_version(self):
        version = self.SorterClass.get_sorter_version()
        print("test_get_versions", self.SorterClass.sorter_name, version)


class SnippetsSorterCommonTestSuite:
    """
    This class run some basic for a sorter class.
    This is the minimal test suite for each sorter class:
      * run once
    """

    @pytest.fixture(autouse=True)
    def create_cache_folder(self, tmp_path_factory):
        self.cache_folder = tmp_path_factory.mktemp("cache_folder")

    SorterClass = None

    def setUp(self):
        recording, sorting_gt = generate_ground_truth_recording(num_channels=4, durations=[60], seed=0)
        snippets_folder = self.cache_folder / "snippets"
        if snippets_folder.is_dir():
            shutil.rmtree(snippets_folder)

        nse = snippets_from_sorting(recording=recording, sorting=sorting_gt, nbefore=20, nafter=44)

        self.snippets = nse.save(folder=snippets_folder, verbose=False, format="npy")
        print(self.snippets)

    def test_with_run(self):
        # some sorter (TDC, KS, KS2, ...) work by default with the raw binary
        # format as input to avoid copy when the recording is already this format

        snippets = self.snippets

        sorter_name = self.SorterClass.sorter_name

        output_folder = self.cache_folder / sorter_name

        sorter_params = self.SorterClass.default_params()

        sorting = run_sorter(
            sorter_name,
            snippets,
            folder=output_folder,
            remove_existing_folder=True,
            delete_output_folder=False,
            verbose=False,
            raise_error=True,
            **sorter_params,
        )

        del sorting

    def test_get_version(self):
        version = self.SorterClass.get_sorter_version()
        print("test_get_versions", self.SorterClass.sorter_name, version)
