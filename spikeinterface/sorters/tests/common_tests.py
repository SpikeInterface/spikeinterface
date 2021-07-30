import unittest

from spikeinterface.extractors import toy_example, BinaryRecordingExtractor
from probeinterface import write_probeinterface, read_probeinterface

from spikeinterface.sorters import run_sorter


class SorterCommonTestSuite:
    """
    This class run some basic for a sorter class.
    This is the minimal test suite for each sorter class:
      * run once
    """
    SorterClass = None

    def setUp(self):
        recording, sorting_gt = toy_example(num_channels=4, duration=60, seed=0, num_segments=1)
        self.recording = recording.save(verbose=False, format='binary')
        print(self.recording)

    def test_with_class(self):
        # test the classmethod approach

        SorterClass = self.SorterClass
        recording = self.recording

        sorter_params = SorterClass.default_params()

        output_folder = None
        verbose = False
        remove_existing_folder = True
        raise_error = True

        output_folder = SorterClass.initialize_folder(recording, output_folder, verbose, remove_existing_folder)
        SorterClass.set_params_to_folder(recording, output_folder, sorter_params, verbose)
        SorterClass.setup_recording(recording, output_folder, verbose)
        SorterClass.run_from_folder(output_folder, raise_error, verbose)
        sorting = SorterClass.get_result_from_folder(output_folder)

        # for unit_id in sorting.get_unit_ids():
        # print('unit #', unit_id, 'nb', len(sorting.get_unit_spike_train(unit_id)))

        del sorting

    def test_with_run(self):
        # some sorter (TDC, KS, KS2, ...) work by default with the raw binary
        # format as input to avoid copy when the recording is already this format

        recording = self.recording

        sorter_name = self.SorterClass.sorter_name

        sorter_params = self.SorterClass.default_params()

        sorting = run_sorter(sorter_name, recording, output_folder=None,
                             remove_existing_folder=True, delete_output_folder=False,
                             verbose=False, raise_error=True, **sorter_params)

        del sorting

    def test_get_version(self):
        version = self.SorterClass.get_sorter_version()
        print('test_get_version:s', self.SorterClass.sorter_name, version)
