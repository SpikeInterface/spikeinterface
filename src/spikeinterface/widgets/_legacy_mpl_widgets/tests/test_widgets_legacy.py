import unittest
import pytest
import sys
from pathlib import Path

if __name__ != '__main__':
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spikeinterface import extract_waveforms, load_waveforms, download_dataset
import spikeinterface.extractors as se
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
from spikeinterface.postprocessing import compute_spike_amplitudes
from spikeinterface.qualitymetrics import compute_quality_metrics


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "widgets"
else:
    cache_folder = Path("cache_folder") / "widgets"


class TestWidgets(unittest.TestCase):
    def setUp(self):
        local_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
        self._rec = se.MEArecRecordingExtractor(local_path)

        self._sorting = se.MEArecSortingExtractor(local_path)

        self.num_units = len(self._sorting.get_unit_ids())
        #  self._we = extract_waveforms(self._rec, self._sorting, './toy_example', load_if_exists=True)
        if (cache_folder / 'mearec_test').is_dir():
            self._we = load_waveforms(cache_folder / 'mearec_test')
        else:
            self._we = extract_waveforms(self._rec, self._sorting, cache_folder / 'mearec_test')

        self._amplitudes = compute_spike_amplitudes(self._we, peak_sign='neg', outputs='by_unit')
        self._gt_comp = sc.compare_sorter_to_ground_truth(self._sorting, self._sorting)

    def tearDown(self):
        pass

    # def test_timeseries(self):
    #     sw.plot_timeseries(self._rec, mode='auto')
    #     sw.plot_timeseries(self._rec, mode='line', show_channel_ids=True)
    #     sw.plot_timeseries(self._rec, mode='map', show_channel_ids=True)
    #     sw.plot_timeseries(self._rec, mode='map', show_channel_ids=True, order_channel_by_depth=True)

    def test_rasters(self):
        sw.plot_rasters(self._sorting)

    def test_plot_probe_map(self):
        
        sw.plot_probe_map(self._rec)
        sw.plot_probe_map(self._rec, with_channel_ids=True)

    # TODO
    # def test_spectrum(self):
    # sw.plot_spectrum(self._rec)

    # TODO
    # def test_spectrogram(self):
    # sw.plot_spectrogram(self._rec, channel=0)

    # def test_unitwaveforms(self):
    #     w = sw.plot_unit_waveforms(self._we)
    #     unit_ids = self._sorting.unit_ids[:6]
    #     sw.plot_unit_waveforms(self._we, max_channels=5, unit_ids=unit_ids)
    #     sw.plot_unit_waveforms(self._we, radius_um=60, unit_ids=unit_ids)

    # def test_plot_unit_waveform_density_map(self):
    #    unit_ids = self._sorting.unit_ids[:3]
    #    sw.plot_unit_waveform_density_map(self._we, unit_ids=unit_ids, max_channels=4)
    #    sw.plot_unit_waveform_density_map(self._we, unit_ids=unit_ids, radius_um=50)
    #
    #    sw.plot_unit_waveform_density_map(self._we, unit_ids=unit_ids, radius_um=25, same_axis=True)
    #    sw.plot_unit_waveform_density_map(self._we, unit_ids=unit_ids, max_channels=2, same_axis=True)

    # def test_unittemplates(self):
    #     sw.plot_unit_templates(self._we)

    def test_plot_unit_probe_map(self):
        sw.plot_unit_probe_map(self._we, with_channel_ids=True)
        sw.plot_unit_probe_map(self._we, animated=True)

    # def test_plot_units_depth_vs_amplitude(self):
    #     sw.plot_units_depth_vs_amplitude(self._we)

    # def test_amplitudes_timeseries(self):
    #     sw.plot_amplitudes_timeseries(self._we)
    #     unit_ids = self._sorting.unit_ids[:4]
    #     sw.plot_amplitudes_timeseries(self._we, unit_ids=unit_ids)

    # def test_amplitudes_distribution(self):
    #     sw.plot_amplitudes_distribution(self._we)

    def test_principal_component(self):
        sw.plot_principal_component(self._we)

    # def test_plot_unit_localization(self):
    #     sw.plot_unit_localization(self._we, with_channel_ids=True)
    #     sw.plot_unit_localization(self._we, method='monopolar_triangulation')

    # def test_autocorrelograms(self):
    #     unit_ids = self._sorting.unit_ids[:4]
    #     sw.plot_autocorrelograms(self._sorting, unit_ids=unit_ids, window_ms=500.0, bin_ms=20.0)

    # def test_crosscorrelogram(self):
    #     unit_ids = self._sorting.unit_ids[:4]
    #     sw.plot_crosscorrelograms(self._sorting, unit_ids=unit_ids, window_ms=500.0, bin_ms=20.0)

    def test_isi_distribution(self):
        sw.plot_isi_distribution(self._sorting, bin_ms=5., window_ms=500.)
        fig, axes = plt.subplots(self.num_units, 1)
        sw.plot_isi_distribution(self._sorting, axes=axes)

    def test_plot_drift_over_time(self):
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        peaks = detect_peaks(self._rec, method='locally_exclusive')
        sw.plot_drift_over_time(self._rec, peaks=peaks, bin_duration_s=1.,
                                weight_with_amplitudes=True, mode='heatmap')
        sw.plot_drift_over_time(self._rec, peaks=peaks, bin_duration_s=1.,
                                weight_with_amplitudes=False, mode='heatmap')
        sw.plot_drift_over_time(self._rec, peaks=peaks, weight_with_amplitudes=False, mode='scatter',
                                scatter_plot_kwargs={'color': 'r'})

    def test_plot_peak_activity_map(self):
        sw.plot_peak_activity_map(self._rec, with_channel_ids=True)
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
        fig, axes = plt.subplots(len(msc.object_list), 1)
        sw.plot_multicomp_agreement_by_sorter(msc, axes=axes)

    def test_sorting_performance(self):
        metrics = compute_quality_metrics(self._we, metric_names=['snr'])
        sw.plot_sorting_performance(self._gt_comp, metrics, performance_name='accuracy', metric_name='snr')

    #~ def test_plot_unit_summary(self):
        #~ unit_id = self._sorting.unit_ids[4]
        #~ sw.plot_unit_summary(self._we, unit_id)


if __name__ == '__main__':
    # unittest.main()

    mytest = TestWidgets()
    mytest.setUp()

    #~ mytest.test_timeseries()
    #~ mytest.test_rasters()
    mytest.test_plot_probe_map()
    #~ mytest.test_unitwaveforms()
    #~ mytest.test_plot_unit_waveform_density_map()
    # mytest.test_unittemplates()
    #~ mytest.test_plot_unit_probe_map()
    #  mytest.test_plot_units_depth_vs_amplitude()
    #~ mytest.test_amplitudes_timeseries()
    #~ mytest.test_amplitudes_distribution()
    #~ mytest.test_principal_component()
    #~ mytest.test_plot_unit_localization()

    #~ mytest.test_autocorrelograms()
    #~ mytest.test_crosscorrelogram()
    #~ mytest.test_isi_distribution()

    #~ mytest.test_plot_drift_over_time()
    #~ mytest.test_plot_peak_activity_map()

    # mytest.test_confusion()
    # mytest.test_agreement()
    #~ mytest.test_multicomp_graph()
    #  mytest.test_sorting_performance()

    #~ mytest.test_plot_unit_summary()

    plt.show()
