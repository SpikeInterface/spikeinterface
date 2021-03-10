import unittest

from spikeinterface.extractors import toy_example, BinaryRecordingExtractor
from probeinterface import write_probeinterface, read_probeinterface



class SorterCommonTestSuite:
    """
    This class run some basic for a sorter class.
    This is the minimal test suite for each sorter class:
      * run once
    """
    SorterClass = None

    #~ def setUp(self):
        #~ self.recording, self.sorting_gt = toy_example(num_channels=4, duration=10, seed=0, num_segments=1)

    def test_on_toy(self):
        recording, sorting_gt = toy_example(num_channels=4, duration=60, seed=0, num_segments=1)

        params = self.SorterClass.default_params()

        sorter = self.SorterClass(recording=recording, output_folder=None, verbose=False)
        sorter.set_params(**params)
        sorter.run()
        sorting = sorter.get_result()

        #~ for unit_id in sorting.get_unit_ids():
            #~ print('unit #', unit_id, 'nb', len(sorting.get_unit_spike_train(unit_id)))

        del sorting

    def test_with_BinDatRecordingExtractor(self):
        # some sorter (TDC, KS, KS2, ...) work by default with the raw binary
        # format as input to avoid copy when the recording is already this format
        
        recording, sorting_gt = toy_example(num_channels=4, duration=10, seed=0, num_segments=1)

        # create a raw dat file and probeinterface file
        raw_filename = 'raw_file.dat'
        BinaryRecordingExtractor.write_recording(recording,
                                files_path=[raw_filename], time_axis=0, dtype='float32')
        probe_filename = 'file_probe.json'
        write_probeinterface(probe_filename, recording.get_probegroup())
        
        samplerate = recording.get_sampling_frequency()
        num_chan = recording.get_num_channels()
        
        # load back
        recording = BinaryRecordingExtractor(raw_filename, samplerate, num_chan, 'float32')
        probegroup = read_probeinterface(probe_filename)
        recording = recording.set_probes(probegroup)

        params = self.SorterClass.default_params()
        sorter = self.SorterClass(recording=recording, output_folder=None)
        sorter.set_params(**params)
        sorter.run()
        sorting = sorter.get_result()

        #~ for unit_id in sorting.get_unit_ids():
            #~ print('unit #', unit_id, 'nb', len(sorting.get_unit_spike_train(unit_id)))
        del sorting

    def test_get_version(self):
        version = self.SorterClass.get_sorter_version()
        print('test_get_version:s', self.SorterClass.sorter_name, version)
