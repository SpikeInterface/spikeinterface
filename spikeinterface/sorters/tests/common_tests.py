import unittest

from spikeinterface.extractors import toy_example, BinaryRecordingExtractor



class SorterCommonTestSuite:
    """
    This class run some basic for a sorter class.
    This is the minimal test suite for each sorter class:
      * run once
      * run with several groups
      * run with several groups in thread
    """
    SorterClass = None

    def test_on_toy(self):

        recording, sorting_gt = se.example_datasets.toy_example(num_channels=4, duration=60, seed=0)

        params = self.SorterClass.default_params()

        sorter = self.SorterClass(recording=recording, output_folder=None,
                                  grouping_property=None, verbose=False)
        sorter.set_params(**params)
        sorter.run(parallel=False)
        sorting = sorter.get_result()

        for unit_id in sorting.get_unit_ids():
            print('unit #', unit_id, 'nb', len(sorting.get_unit_spike_train(unit_id)))
        del sorting

    #~ def test_several_groups(self):

        #~ # run sorter with several groups in paralel or not
        #~ recording, sorting_gt = se.example_datasets.toy_example(num_channels=8, duration=30, seed=1, dumpable=True,
                                                                #~ dump_folder='test_groups')

        #~ # make 2 artificial groups
        #~ for ch_id in range(0, 4):
            #~ recording.set_channel_property(ch_id, 'group', 0)
        #~ for ch_id in range(4, 8):
            #~ recording.set_channel_property(ch_id, 'group', 1)

        #~ params = self.SorterClass.default_params()
        
        #~ parallel_cases = [False, True]
        #~ joblib_backends = ['loky', 'multiprocessing', 'threading']
        
        #~ for parallel in parallel_cases:
            #~ for backend in joblib_backends:
                #~ sorter = self.SorterClass(recording=recording, output_folder=None,
                                          #~ grouping_property='group', verbose=False)
                #~ if sorter.compatible_with_parallel[backend]:
                    #~ sorter.set_params(**params)
                    #~ sorter.run(parallel=parallel, joblib_backend=backend)
                    #~ sorting = sorter.get_result()
                    #~ for unit_id in sorting.get_unit_ids():
                        #~ print('unit #', unit_id, 'nb', len(sorting.get_unit_spike_train(unit_id)))
                    #~ del sorting


    def test_with_BinDatRecordingExtractor(self):
        # some sorter (TDC, KS, KS2, ...) work by default with the raw binary
        # format as input to avoid copy when the recording is already this format
        
        recording, sorting_gt = se.example_datasets.toy_example(num_channels=2, duration=10, seed=0,
                            num_segments=1)

        # create a raw dat file and prb file
        raw_filename = 'raw_file.dat'
        prb_filename = 'raw_file.prb'

        samplerate = recording.get_sampling_frequency()
        traces = recording.get_traces().astype('float32')
        with open(raw_filename, mode='wb') as f:
            f.write(traces.T.tobytes())

        recording.save_to_probe_file(prb_filename)
        
        recording = BinaryRecordingExtractor(raw_filename, samplerate, 2, 'float32', time_axis=0, offset=0)
        recording = recording.load_probe_file(prb_filename)

        params = self.SorterClass.default_params()
        sorter = self.SorterClass(recording=recording, output_folder=None)
        sorter.set_params(**params)
        sorter.run()
        sorting = sorter.get_result()

        for unit_id in sorting.get_unit_ids():
            print('unit #', unit_id, 'nb', len(sorting.get_unit_spike_train(unit_id)))
        del sorting

    def test_get_version(self):
        self.SorterClass.get_sorter_version()
