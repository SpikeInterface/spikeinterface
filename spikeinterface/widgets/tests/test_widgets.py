import unittest
import pytest
import sys
from pathlib import Path

if __name__ != '__main__':
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spikeinterface import extract_waveforms, download_dataset

from spikeinterface.widgets import HAVE_MPL, HAVE_FIGURL

import spikeinterface.extractors as se
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
from spikeinterface.preprocessing import scale
from spikeinterface.postprocessing import compute_spike_amplitudes
from spikeinterface.qualitymetrics import compute_quality_metrics


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "widgets"
else:
    cache_folder = Path("cache_folder") / "widgets"


class TestWidgets(unittest.TestCase):
    def setUp(self):
        local_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
        self.recording = se.MEArecRecordingExtractor(local_path)

        self.sorting = se.MEArecSortingExtractor(local_path)

        self.num_units = len(self.sorting.get_unit_ids())
        #self.we = extract_waveforms(self.recording, self.sorting, './toy_example', load_if_exists=True)
        self.we = extract_waveforms(self.recording, self.sorting, cache_folder / 'mearec_test', load_if_exists=True)

        # @jeremy : for testing sorting view we can find something here
        # at the moment only mpl will be tested on github actions
        sw.set_default_plotter_backend('matplotlib')
        
        self.amplitudes = compute_spike_amplitudes(self.we, peak_sign='neg', outputs='by_unit')
        self.gt_comp = sc.compare_sorter_to_ground_truth(self.sorting, self.sorting)

    def tearDown(self):
        pass
    
    def test_plot_timeseries(self):
        sw.plot_timeseries(self.recording, mode='auto')
        sw.plot_timeseries(self.recording, mode='line', show_channel_ids=True)
        sw.plot_timeseries(self.recording, mode='map', show_channel_ids=True)
        sw.plot_timeseries(self.recording, mode='map', show_channel_ids=True, order_channel_by_depth=True)
        
        # multi layer
        sw.plot_timeseries({'rec0' : self.recording, 'rec1' : scale(self.recording, gain=0.8, offset=0)},
                    color='r', mode='line', show_channel_ids=True)

    def test_plot_unit_waveforms(self):
        w = sw.plot_unit_waveforms(self.we)
        unit_ids = self.sorting.unit_ids[:6]
        sw.plot_unit_waveforms(self.we, max_channels=5, unit_ids=unit_ids)
        sw.plot_unit_waveforms(self.we, radius_um=60, unit_ids=unit_ids)

    def test_plot_unit_templates(self):
        w = sw.plot_unit_templates(self.we)
        unit_ids = self.sorting.unit_ids[:6]
        sw.plot_unit_templates(self.we, max_channels=5, unit_ids=unit_ids)

    def test_plot_unit_waveforms_density_map(self):
        unit_ids = self.sorting.unit_ids[:2]
        sw.plot_unit_waveforms_density_map(self.we, max_channels=5, unit_ids=unit_ids)
        sw.plot_unit_waveforms_density_map(self.we, max_channels=5, same_axis=True, unit_ids=unit_ids)



if __name__ == '__main__':
    # unittest.main()

    mytest = TestWidgets()
    mytest.setUp()
    
    mytest.test_plot_timeseries()
    
    # mytest.test_plot_unit_waveforms()
    # mytest.test_plot_unit_templates()
    # mytest.test_plot_unit_waveforms_density_map()

    plt.show()
