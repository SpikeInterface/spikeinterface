import unittest
import pytest
import os
from pathlib import Path

if __name__ != '__main__':
    import matplotlib

    matplotlib.use('Agg')

import matplotlib.pyplot as plt


from spikeinterface import extract_waveforms, download_dataset, compute_sparsity

from spikeinterface.widgets import HAVE_MPL, HAVE_SV

import spikeinterface.extractors as se
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
from spikeinterface.preprocessing import scale
from spikeinterface.postprocessing import (compute_correlograms, compute_spike_amplitudes, 
                                           compute_spike_locations, compute_unit_locations,
                                           compute_template_metrics, compute_template_similarity)
from spikeinterface.qualitymetrics import compute_quality_metrics


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "widgets"
else:
    cache_folder = Path("cache_folder") / "widgets"


ON_GITHUB = bool(os.getenv('GITHUB_ACTIONS'))
KACHERY_CLOUD_SET = bool(os.getenv('KACHERY_CLOUD_CLIENT_ID')) and bool(os.getenv('KACHERY_CLOUD_PRIVATE_KEY'))


class TestWidgets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        local_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
        cls.recording = se.MEArecRecordingExtractor(local_path)

        cls.sorting = se.MEArecSortingExtractor(local_path)

        cls.num_units = len(cls.sorting.get_unit_ids())
        cls.we = extract_waveforms(cls.recording, cls.sorting, cache_folder / 'mearec_test', load_if_exists=True)
        
        sw.set_default_plotter_backend('matplotlib')

        metric_names = ["snr", "isi_violations", "num_spikes"]
        _ = compute_spike_amplitudes(cls.we)
        _ = compute_unit_locations(cls.we)
        _ = compute_spike_locations(cls.we)
        _ = compute_quality_metrics(cls.we, metric_names=metric_names)
        _ = compute_template_metrics(cls.we)
        _ = compute_correlograms(cls.we)
        _ = compute_template_similarity(cls.we)

        # make sparse waveforms
        cls.sparsity_radius =  compute_sparsity(cls.we, method="radius", radius_um=50)
        cls.sparsity_best =  compute_sparsity(cls.we, method="best_channels", num_channels=5)
        cls.we_sparse = cls.we.save(folder=cache_folder / 'mearec_test_sparse', sparsity=cls.sparsity_radius)

        cls.skip_backends = ["ipywidgets"]

        if ON_GITHUB and not KACHERY_CLOUD_SET:
            cls.skip_backends.append("sortingview")

        print(f"Widgets tests: skipping backends - {cls.skip_backends}")

        cls.backend_kwargs = {
            'matplotlib': {},
            'sortingview': {},
            'ipywidgets': {}
        }

        cls.gt_comp = sc.compare_sorter_to_ground_truth(cls.sorting, cls.sorting)
    
    def test_plot_timeseries(self):
        possible_backends = list(sw.TimeseriesWidget.possible_backends.keys())
        for backend in possible_backends:
            if ON_GITHUB and backend == "sortingview":
                continue
            if backend not in self.skip_backends:
                sw.plot_timeseries(self.recording, mode='map', show_channel_ids=True,
                                   backend=backend, **self.backend_kwargs[backend])
                sw.plot_timeseries(self.recording, mode='map', show_channel_ids=True, 
                                   order_channel_by_depth=True, backend=backend,
                                   **self.backend_kwargs[backend])

                if backend != "sortingview":
                    sw.plot_timeseries(self.recording, mode='auto', backend=backend, **self.backend_kwargs[backend])
                    sw.plot_timeseries(self.recording, mode='line', show_channel_ids=True, backend=backend,
                                    **self.backend_kwargs[backend])
                    # multi layer
                    sw.plot_timeseries({'rec0' : self.recording, 'rec1' : scale(self.recording, gain=0.8, offset=0)},
                                        color='r', mode='line', show_channel_ids=True, backend=backend,
                                        **self.backend_kwargs[backend])

    def test_plot_unit_waveforms(self):
        possible_backends = list(sw.UnitWaveformsWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_waveforms(self.we, backend=backend, **self.backend_kwargs[backend])
                unit_ids = self.sorting.unit_ids[:6]
                sw.plot_unit_waveforms(self.we, sparsity=self.sparsity_radius, unit_ids=unit_ids, backend=backend,
                                       **self.backend_kwargs[backend])
                sw.plot_unit_waveforms(self.we, sparsity=self.sparsity_best, unit_ids=unit_ids, backend=backend,
                                       **self.backend_kwargs[backend])
                sw.plot_unit_waveforms(self.we_sparse, unit_ids=unit_ids, backend=backend,
                                       **self.backend_kwargs[backend])

    def test_plot_unit_templates(self):
        possible_backends = list(sw.UnitWaveformsWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_templates(self.we, backend=backend, **self.backend_kwargs[backend])
                unit_ids = self.sorting.unit_ids[:6]
                sw.plot_unit_templates(self.we, sparsity=self.sparsity_radius, unit_ids=unit_ids, backend=backend,
                                       **self.backend_kwargs[backend])
                sw.plot_unit_templates(self.we_sparse, sparsity=self.sparsity_best, unit_ids=unit_ids, backend=backend,
                                       **self.backend_kwargs[backend])

    def test_plot_unit_waveforms_density_map(self):
        possible_backends = list(sw.UnitWaveformDensityMapWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.sorting.unit_ids[:2]
                sw.plot_unit_waveforms_density_map(self.we, unit_ids=unit_ids, backend=backend,
                                                   **self.backend_kwargs[backend])
                sw.plot_unit_waveforms_density_map(self.we, sparsity=self.sparsity_radius, same_axis=False,
                                                   unit_ids=unit_ids, backend=backend,
                                                   **self.backend_kwargs[backend])
                sw.plot_unit_waveforms_density_map(self.we_sparse, sparsity=None, same_axis=True,
                                                   unit_ids=unit_ids, backend=backend,
                                                   **self.backend_kwargs[backend])

    def test_autocorrelograms(self):
        possible_backends = list(sw.AutoCorrelogramsWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.sorting.unit_ids[:4]
                sw.plot_autocorrelograms(self.sorting, unit_ids=unit_ids, window_ms=500.0, bin_ms=20.0, 
                                         backend=backend, **self.backend_kwargs[backend])

    def test_crosscorrelogram(self):
        possible_backends = list(sw.CrossCorrelogramsWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.sorting.unit_ids[:4]
                sw.plot_crosscorrelograms(self.sorting, unit_ids=unit_ids, window_ms=500.0, bin_ms=20.0, 
                                          backend=backend, **self.backend_kwargs[backend])
        
    def test_amplitudes(self):
        possible_backends = list(sw.AmplitudesWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_amplitudes(self.we, backend=backend, **self.backend_kwargs[backend])
                unit_ids = self.we.unit_ids[:4]
                sw.plot_amplitudes(self.we, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend])
                sw.plot_amplitudes(self.we, unit_ids=unit_ids, plot_histograms=True,
                                   backend=backend, **self.backend_kwargs[backend])
                sw.plot_amplitudes(self.we_sparse, unit_ids=unit_ids, plot_histograms=True,
                                   backend=backend, **self.backend_kwargs[backend])

    def test_plot_all_amplitudes_distributions(self):
        possible_backends = list(sw.AllAmplitudesDistributionsWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.we.unit_ids[:4]
                sw.plot_all_amplitudes_distributions(self.we, unit_ids=unit_ids, backend=backend,
                                                     **self.backend_kwargs[backend])
                sw.plot_all_amplitudes_distributions(self.we_sparse, unit_ids=unit_ids, backend=backend,
                                                     **self.backend_kwargs[backend])

    def test_unit_locations(self):
        possible_backends = list(sw.UnitLocationsWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_locations(self.we, with_channel_ids=True, backend=backend,
                                       **self.backend_kwargs[backend])
                sw.plot_unit_locations(self.we_sparse, with_channel_ids=True, backend=backend,
                                       **self.backend_kwargs[backend])

    def test_spike_locations(self):
        possible_backends = list(sw.SpikeLocationsWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_spike_locations(self.we, with_channel_ids=True, backend=backend,
                                        **self.backend_kwargs[backend])
                sw.plot_spike_locations(self.we_sparse, with_channel_ids=True, backend=backend,
                                        **self.backend_kwargs[backend])

    def test_similarity(self):
        possible_backends = list(sw.TemplateSimilarityWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_template_similarity(self.we, backend=backend,
                                            **self.backend_kwargs[backend])
                sw.plot_template_similarity(self.we_sparse, backend=backend,
                                            **self.backend_kwargs[backend])

    def test_quality_metrics(self):
        possible_backends = list(sw.QualityMetricsWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_quality_metrics(self.we, backend=backend, **self.backend_kwargs[backend])
                sw.plot_quality_metrics(self.we_sparse, backend=backend, **self.backend_kwargs[backend])

    def test_template_metrics(self):
        possible_backends = list(sw.TemplateMetricsWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_template_metrics(self.we, backend=backend, **self.backend_kwargs[backend])
                sw.plot_template_metrics(self.we_sparse, backend=backend, **self.backend_kwargs[backend])
    
    def test_plot_unit_depths(self):
        possible_backends = list(sw.UnitDepthsWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_depths(self.we, backend=backend, **self.backend_kwargs[backend])
                sw.plot_unit_depths(self.we_sparse, backend=backend, **self.backend_kwargs[backend])

    def test_plot_unit_summary(self):
        possible_backends = list(sw.UnitSummaryWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_summary(self.we, self.we.sorting.unit_ids[0], backend=backend,
                                     **self.backend_kwargs[backend])
                sw.plot_unit_summary(self.we_sparse, self.we.sorting.unit_ids[0], backend=backend,
                                     **self.backend_kwargs[backend])

    def test_sorting_summary(self):
        possible_backends = list(sw.SortingSummaryWidget.possible_backends.keys())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_sorting_summary(self.we, backend=backend, **self.backend_kwargs[backend])
                sw.plot_sorting_summary(self.we_sparse, backend=backend, **self.backend_kwargs[backend])

if __name__ == '__main__':
    # unittest.main()

    mytest = TestWidgets()
    mytest.setUpClass()

    mytest.test_plot_unit_waveforms_density_map()
    # mytest.test_plot_unit_summary()
    # mytest.test_plot_all_amplitudes_distributions()
    # mytest.test_plot_timeseries()
    # mytest.test_plot_unit_waveforms()
    # mytest.test_plot_unit_templates()
    # mytest.test_plot_unit_templates()
    # mytest.test_plot_unit_depths()
    # mytest.test_plot_unit_templates()
    # mytest.test_plot_unit_summary()
    mytest.test_quality_metrics()
    mytest.test_sorting_summary()

    # plt.ion()
    plt.show()
