import unittest
import sys

import matplotlib.pyplot as plt

import spikeinterface.extractors as se
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc


if sys.platform == "win32":
    memmaps = [False]
else:
    memmaps = [False, True]


class TestWidgets(unittest.TestCase):
    def setUp(self):
        self._rec, self._sorting = se.toy_example(num_channels=10, duration=10, num_segments=1)
        self.num_units = len(self._sorting.get_unit_ids())

    def tearDown(self):
        pass

    def test_timeseries(self):
        sw.plot_timeseries(self._rec)

    def test_rasters(self):
        sw.plot_rasters(self._sorting)
    
    def test_plot_probe_map(self):
        sw.plot_probe_map(self._rec)
        #~ plt.show()

    #~ def test_spectrum(self):
        #~ sw.plot_spectrum(self._rec)

    #~ def test_spectrogram(self):
        #~ sw.plot_spectrogram(self._rec, channel=0)

    #~ def test_geometry(self):
        #~ sw.plot_electrode_geometry(self._rec)

    #~ def test_activitymap(self):
        #~ sw.plot_activity_map(self._rec, activity='rate')
        #~ sw.plot_activity_map(self._rec, activity='amplitude')

    #~ def test_unitwaveforms(self):
        #~ for m in memmaps:
            #~ sw.plot_unit_waveforms(self._rec, self._sorting, memmap=m)
            #~ fig, axes = plt.subplots(self.num_units, 1)
            #~ sw.plot_unit_waveforms(self._rec, self._sorting, axes=axes, memmap=m)

    #~ def test_unittemplates(self):
        #~ for m in memmaps:
            #~ sw.plot_unit_templates(self._rec, self._sorting, memmap=m)
            #~ fig, axes = plt.subplots(self.num_units, 1)
            #~ sw.plot_unit_templates(self._rec, self._sorting, axes=axes, memmap=m)

    #~ def test_unittemplatemaps(self):
        #~ for m in memmaps:
            #~ sw.plot_unit_template_maps(self._rec, self._sorting, memmap=m)

    #~ def test_ampdist(self):
        #~ sw.plot_amplitudes_distribution(self._rec, self._sorting)
        #~ fig, axes = plt.subplots(self.num_units, 1)
        #~ sw.plot_amplitudes_distribution(self._rec, self._sorting, axes=axes)

    #~ def test_amptime(self):
        #~ sw.plot_amplitudes_timeseries(self._rec, self._sorting)
        #~ fig, axes = plt.subplots(self.num_units, 1)
        #~ sw.plot_amplitudes_timeseries(self._rec, self._sorting, axes=axes)

    #~ def test_features(self):
        #~ for m in memmaps:
            #~ sw.plot_pca_features(self._rec, self._sorting, memap=m)
            #~ fig, axes = plt.subplots(self.num_units, 1)
            #~ sw.plot_pca_features(self._rec, self._sorting, axes=axes, memap=m)

    #~ def test_ach(self):
        #~ sw.plot_autocorrelograms(self._sorting, bin_size=1, window=10)
        #~ fig, axes = plt.subplots(self.num_units, 1)
        #~ sw.plot_autocorrelograms(self._sorting, axes=axes)

    #~ def test_cch(self):
        #~ sw.plot_crosscorrelograms(self._sorting, bin_size=1, window=10)
        #~ fig, axes = plt.subplots(self.num_units, self.num_units)  # for cch need square matrix
        #~ sw.plot_crosscorrelograms(self._sorting, axes=axes)

    #~ def test_isi(self):
        #~ sw.plot_isi_distribution(self._sorting, bins=10, window=1)
        #~ fig, axes = plt.subplots(self.num_units, 1)
        #~ sw.plot_isi_distribution(self._sorting, axes=axes)


    def test_confusion(self):
        gt_comp = sc.compare_sorter_to_ground_truth(self._sorting, self._sorting)
        sw.plot_confusion_matrix(gt_comp, count_text=True)

    def test_agreement(self):
        comp = sc.compare_sorter_to_ground_truth(self._sorting, self._sorting)
        sw.plot_agreement_matrix(comp, count_text=True)
        
        
    def test_multicomp_graph(self):
        msc = sc.compare_multiple_sorters([self._sorting, self._sorting, self._sorting])
        sw.plot_multicomp_graph(msc, edge_cmap='viridis', node_cmap='rainbow', draw_labels=False)
        sw.plot_multicomp_agreement(msc)
        sw.plot_multicomp_agreement_by_sorter(msc)
        fig, axes = plt.subplots(len(msc.sorting_list), 1)
        sw.plot_multicomp_agreement_by_sorter(msc, axes=axes)


if __name__ == '__main__':
    #~ unittest.main()
    
    mytest = TestWidgets()
    mytest.setUp()
    mytest.test_plot_probe_map()
    
    plt.show()
