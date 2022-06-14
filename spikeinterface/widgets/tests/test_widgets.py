import unittest
import pytest
import sys
from pathlib import Path

if __name__ != '__main__':
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spikeinterface import extract_waveforms, download_dataset

from spikeinterface.widgets import HAVE_MPL, HAVE_SV

import spikeinterface.extractors as se
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
import spikeinterface.toolkit as st


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
        self._we = extract_waveforms(self._rec, self._sorting, cache_folder / 'mearec_test', load_if_exists=True)

        self._amplitudes = st.compute_spike_amplitudes(self._we, peak_sign='neg', outputs='by_unit')
        self._gt_comp = sc.compare_sorter_to_ground_truth(self._sorting, self._sorting)

        # @jeremy : for testing sorting view we can find something here
        # at the moment only mpl will be tested on github actions
        sw.set_default_plotter_backend('matplotlib')

    def tearDown(self):
        pass

    def test_plot_unit_waveforms(self):
        w = sw.plot_unit_waveforms(self._we)
        unit_ids = self._sorting.unit_ids[:6]
        sw.plot_unit_waveforms(self._we, max_channels=5, unit_ids=unit_ids)
        sw.plot_unit_waveforms(self._we, radius_um=60, unit_ids=unit_ids)

    def test_plot_unit_templates(self):
        w = sw.plot_unit_templates(self._we)
        unit_ids = self._sorting.unit_ids[:6]
        sw.plot_unit_templates(self._we, max_channels=5, unit_ids=unit_ids)

    def test_plot_unit_waveforms_density_map(self):
        unit_ids = self._sorting.unit_ids[:2]
        sw.plot_unit_waveforms_density_map(self._we, max_channels=5, unit_ids=unit_ids)
        sw.plot_unit_waveforms_density_map(self._we, max_channels=5, same_axis=True, unit_ids=unit_ids)



if __name__ == '__main__':
    # unittest.main()

    mytest = TestWidgets()
    mytest.setUp()

    # mytest.test_plot_unit_waveforms()
    # mytest.test_plot_unit_templates()
    mytest.test_plot_unit_waveforms_density_map()

    plt.show()
