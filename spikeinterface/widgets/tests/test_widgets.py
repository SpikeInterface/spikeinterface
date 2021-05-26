import unittest
import sys

if __name__ != '__main__':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


from spikeinterface import extract_waveforms, download_dataset
import spikeinterface.extractors as se
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
import spikeinterface.toolkit as st

if sys.platform == "win32":
    memmaps = [False]
else:
    memmaps = [False, True]


class TestWidgets(unittest.TestCase):
    def setUp(self):
        #~ self._rec, self._sorting = se.toy_example(num_channels=10, duration=10, num_segments=1)
        #~ self._rec = self._rec.save()
        #~ self._sorting = self._sorting.save()
        local_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
        self._rec = se.MEArecRecordingExtractor(local_path)
        
        self._sorting = se.MEArecSortingExtractor(local_path)
        
        self.num_units = len(self._sorting.get_unit_ids())
        # self._we = extract_waveforms(self._rec, self._sorting, './toy_example', load_if_exists=True)
        self._we = extract_waveforms(self._rec, self._sorting, './mearec_test', load_if_exists=True)
        
        self._amplitudes = st.get_unit_amplitudes(self._we,  peak_sign='neg', outputs='by_units')
        self._gt_comp = sc.compare_sorter_to_ground_truth(self._sorting, self._sorting)


    def tearDown(self):
        pass

    def test_timeseries(self):
        sw.plot_timeseries(self._rec)

    def test_rasters(self):
        sw.plot_rasters(self._sorting)
    
    def test_plot_probe_map(self):
        sw.plot_probe_map(self._rec)

    # TODO
    # def test_spectrum(self):
        # sw.plot_spectrum(self._rec)

    # TODO
    # def test_spectrogram(self):
        # sw.plot_spectrogram(self._rec, channel=0)

    def test_unitwaveforms(self):
        sw.plot_unit_waveforms(self._we)

    def test_unittemplates(self):
        sw.plot_unit_templates(self._we)
    
    def test_plot_unit_map(self):
        sw.plot_unit_map(self._we)
        sw.plot_unit_map(self._we, animated=True)

    def test_amplitudes_timeseries(self):
        sw.plot_amplitudes_timeseries(self._we)
        sw.plot_amplitudes_timeseries(self._we, amplitudes=self._amplitudes)

    def test_amplitudes_distribution(self):
        sw.plot_amplitudes_distribution(self._we)
        sw.plot_amplitudes_distribution(self._we, amplitudes=self._amplitudes)
        
    def test_principal_component(self):
        sw.plot_principal_component(self._we)
        
    def test_plot_unit_localization(self):
        sw.plot_unit_localization(self._we)

    def test_autocorrelograms(self):
        unit_ids = self._sorting.unit_ids[:4]
        sw.plot_autocorrelograms(self._sorting, unit_ids=unit_ids, window_ms=500.0, bin_ms=20.0, symmetrize=True)
        
    def test_crosscorrelogram(self):
        unit_ids = self._sorting.unit_ids[:4]
        sw.plot_crosscorrelograms(self._sorting, unit_ids=unit_ids, window_ms=500.0, bin_ms=20.0, symmetrize=True)


    def test_isi_distribution(self):
        sw.plot_isi_distribution(self._sorting, bins=10, window=1)
        fig, axes = plt.subplots(self.num_units, 1)
        sw.plot_isi_distribution(self._sorting, axes=axes)
    
    def test_plot_drift_over_time(self):
        from spikeinterface.sortingcomponents import detect_peaks
        peaks = detect_peaks(self._rec, method='locally_exclusive')
        sw.plot_drift_over_time(self._rec, peaks=peaks, bin_duration_s=1.,
            weight_with_amplitudes=True, mode='heatmap')
        sw.plot_drift_over_time(self._rec, peaks=peaks, bin_duration_s=1.,
                weight_with_amplitudes=False, mode='heatmap')
        sw.plot_drift_over_time(self._rec, peaks=peaks, weight_with_amplitudes=False, mode='scatter')
    
    def test_plot_peak_activity_map(self):
        sw.plot_peak_activity_map(self._rec)
        sw.plot_peak_activity_map(self._rec, bin_duration_s=1.)

    def test_confusion(self):
        
        sw.plot_confusion_matrix(self._gt_comp, count_text=True)

    def test_agreement(self):
        
        sw.plot_agreement_matrix(self._gt_comp, count_text=True)
        
        
    def test_multicomp_graph(self):
        msc = sc.compare_multiple_sorters([self._sorting, self._sorting, self._sorting])
        sw.plot_multicomp_graph(msc, edge_cmap='viridis', node_cmap='rainbow', draw_labels=False)
        sw.plot_multicomp_agreement(msc)
        sw.plot_multicomp_agreement_by_sorter(msc)
        fig, axes = plt.subplots(len(msc.sorting_list), 1)
        sw.plot_multicomp_agreement_by_sorter(msc, axes=axes)
        
    def test_sorting_performance(self):
        metrics = st.compute_quality_metrics(self._we, metric_names=['snr'])
        sw.plot_sorting_performance(self._gt_comp, metrics, performance_name='accuracy', metric_name='snr')
    
    


if __name__ == '__main__':
    # unittest.main()
    
    mytest = TestWidgets()
    mytest.setUp()

    # mytest.test_timeseries()
    # mytest.test_rasters()
    # mytest.test_plot_probe_map()
    # mytest.test_unitwaveforms()
    # mytest.test_unittemplates()
    # mytest.test_plot_unit_map()
    # mytest.test_amplitudes_timeseries()
    # mytest.test_amplitudes_distribution()
    # mytest.test_principal_component()
    # mytest.test_plot_unit_localization()
    
    # mytest.test_autocorrelograms()
    # mytest.test_crosscorrelogram()
    # mytest.test_isi_distribution()
    
    
    mytest.test_plot_drift_over_time()
    # mytest.test_plot_peak_activity_map()
    
    
    
    # mytest.test_confusion()
    # mytest.test_agreement()
    #~ mytest.test_multicomp_graph()
    # mytest.test_sorting_performance()
        
    plt.show()
