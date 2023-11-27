import unittest
import pytest
import os
from pathlib import Path
import shutil

if __name__ != "__main__":
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt


from spikeinterface import (
    load_extractor,
    extract_waveforms,
    load_waveforms,
    download_dataset,
    compute_sparsity,
    generate_ground_truth_recording,
)

import spikeinterface.extractors as se
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
from spikeinterface.preprocessing import scale
from spikeinterface.postprocessing import (
    compute_correlograms,
    compute_spike_amplitudes,
    compute_spike_locations,
    compute_unit_locations,
    compute_template_metrics,
    compute_template_similarity,
)
from spikeinterface.qualitymetrics import compute_quality_metrics


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "widgets"
else:
    cache_folder = Path("cache_folder") / "widgets"


ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))
KACHERY_CLOUD_SET = bool(os.getenv("KACHERY_CLOUD_CLIENT_ID")) and bool(os.getenv("KACHERY_CLOUD_PRIVATE_KEY"))


class TestWidgets(unittest.TestCase):
    @classmethod
    def _delete_widget_folders(cls):
        for name in (
            "recording",
            "sorting",
            "we_dense",
            "we_sparse",
        ):
            if (cache_folder / name).is_dir():
                shutil.rmtree(cache_folder / name)

    @classmethod
    def setUpClass(cls):
        cls._delete_widget_folders()

        if (cache_folder / "recording").is_dir() and (cache_folder / "sorting").is_dir():
            cls.recording = load_extractor(cache_folder / "recording")
            cls.sorting = load_extractor(cache_folder / "sorting")
        else:
            recording, sorting = generate_ground_truth_recording(
                durations=[30.0],
                sampling_frequency=28000.0,
                num_channels=32,
                num_units=10,
                generate_probe_kwargs=dict(
                    num_columns=2,
                    xpitch=20,
                    ypitch=20,
                    contact_shapes="circle",
                    contact_shape_params={"radius": 6},
                ),
                generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
                noise_kwargs=dict(noise_level=5.0, strategy="on_the_fly"),
                seed=2205,
            )
            # cls.recording = recording.save(folder=cache_folder / "recording")
            # cls.sorting = sorting.save(folder=cache_folder / "sorting")
            cls.recording = recording
            cls.sorting = sorting

        cls.num_units = len(cls.sorting.get_unit_ids())

        if (cache_folder / "we_dense").is_dir():
            cls.we_dense = load_waveforms(cache_folder / "we_dense")
        else:
            cls.we_dense = extract_waveforms(
                recording=cls.recording, sorting=cls.sorting, folder=None, mode="memory", sparse=False
            )
            metric_names = ["snr", "isi_violation", "num_spikes"]
            _ = compute_spike_amplitudes(cls.we_dense)
            _ = compute_unit_locations(cls.we_dense)
            _ = compute_spike_locations(cls.we_dense)
            _ = compute_quality_metrics(cls.we_dense, metric_names=metric_names)
            _ = compute_template_metrics(cls.we_dense)
            _ = compute_correlograms(cls.we_dense)
            _ = compute_template_similarity(cls.we_dense)

        sw.set_default_plotter_backend("matplotlib")

        # make sparse waveforms
        cls.sparsity_radius = compute_sparsity(cls.we_dense, method="radius", radius_um=50)
        cls.sparsity_best = compute_sparsity(cls.we_dense, method="best_channels", num_channels=5)
        if (cache_folder / "we_sparse").is_dir():
            cls.we_sparse = load_waveforms(cache_folder / "we_sparse")
        else:
            cls.we_sparse = cls.we_dense.save(folder=cache_folder / "we_sparse", sparsity=cls.sparsity_radius)

        cls.skip_backends = ["ipywidgets", "ephyviewer"]

        if ON_GITHUB and not KACHERY_CLOUD_SET:
            cls.skip_backends.append("sortingview")

        print(f"Widgets tests: skipping backends - {cls.skip_backends}")

        cls.backend_kwargs = {"matplotlib": {}, "sortingview": {}, "ipywidgets": {"display": False}}

        cls.gt_comp = sc.compare_sorter_to_ground_truth(cls.sorting, cls.sorting)

        from spikeinterface.sortingcomponents.peak_detection import detect_peaks

        cls.peaks = detect_peaks(cls.recording, method="locally_exclusive")

    @classmethod
    def tearDownClass(cls):
        del cls.recording, cls.sorting, cls.peaks, cls.gt_comp, cls.we_sparse, cls.we_dense
        # cls._delete_widget_folders()

    def test_plot_traces(self):
        possible_backends = list(sw.TracesWidget.get_possible_backends())
        for backend in possible_backends:
            if ON_GITHUB and backend == "sortingview":
                continue
            if backend not in self.skip_backends:
                sw.plot_traces(
                    self.recording, mode="map", show_channel_ids=True, backend=backend, **self.backend_kwargs[backend]
                )
                sw.plot_traces(
                    self.recording,
                    mode="map",
                    show_channel_ids=True,
                    order_channel_by_depth=True,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

                if backend != "sortingview":
                    sw.plot_traces(self.recording, mode="auto", backend=backend, **self.backend_kwargs[backend])
                    sw.plot_traces(
                        self.recording,
                        mode="line",
                        show_channel_ids=True,
                        backend=backend,
                        **self.backend_kwargs[backend],
                    )
                    # multi layer
                    sw.plot_traces(
                        {"rec0": self.recording, "rec1": scale(self.recording, gain=0.8, offset=0)},
                        color="r",
                        mode="line",
                        show_channel_ids=True,
                        backend=backend,
                        **self.backend_kwargs[backend],
                    )

    def test_plot_unit_waveforms(self):
        possible_backends = list(sw.UnitWaveformsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_waveforms(self.we_dense, backend=backend, **self.backend_kwargs[backend])
                unit_ids = self.sorting.unit_ids[:6]
                sw.plot_unit_waveforms(
                    self.we_dense,
                    sparsity=self.sparsity_radius,
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_unit_waveforms(
                    self.we_dense,
                    sparsity=self.sparsity_best,
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_unit_waveforms(
                    self.we_sparse, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend]
                )

    def test_plot_unit_templates(self):
        possible_backends = list(sw.UnitTemplatesWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                print(f"Testing backend {backend}")
                sw.plot_unit_templates(
                    self.we_dense, backend=backend, templates_percentile_shading=None, **self.backend_kwargs[backend]
                )
                unit_ids = self.sorting.unit_ids[:6]
                sw.plot_unit_templates(
                    self.we_dense,
                    sparsity=self.sparsity_radius,
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                # test different shadings
                sw.plot_unit_templates(
                    self.we_sparse,
                    sparsity=self.sparsity_best,
                    unit_ids=unit_ids,
                    templates_percentile_shading=None,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_unit_templates(
                    self.we_sparse,
                    sparsity=self.sparsity_best,
                    unit_ids=unit_ids,
                    templates_percentile_shading=None,
                    scale=10,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                # test different shadings
                sw.plot_unit_templates(
                    self.we_sparse,
                    sparsity=self.sparsity_best,
                    unit_ids=unit_ids,
                    backend=backend,
                    templates_percentile_shading=None,
                    shade_templates=False,
                    **self.backend_kwargs[backend],
                )
                sw.plot_unit_templates(
                    self.we_sparse,
                    sparsity=self.sparsity_best,
                    unit_ids=unit_ids,
                    templates_percentile_shading=5,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_unit_templates(
                    self.we_sparse,
                    sparsity=self.sparsity_best,
                    unit_ids=unit_ids,
                    templates_percentile_shading=[10, 90],
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                if backend != "sortingview":
                    sw.plot_unit_templates(
                        self.we_sparse,
                        sparsity=self.sparsity_best,
                        unit_ids=unit_ids,
                        templates_percentile_shading=[1, 5, 25, 75, 95, 99],
                        backend=backend,
                        **self.backend_kwargs[backend],
                    )
                else:
                    # sortingview doesn't support more than 2 shadings
                    with self.assertRaises(AssertionError):
                        sw.plot_unit_templates(
                            self.we_sparse,
                            sparsity=self.sparsity_best,
                            unit_ids=unit_ids,
                            templates_percentile_shading=[1, 5, 25, 75, 95, 99],
                            backend=backend,
                            **self.backend_kwargs[backend],
                        )

    def test_plot_unit_waveforms_density_map(self):
        possible_backends = list(sw.UnitWaveformDensityMapWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.sorting.unit_ids[:2]
                sw.plot_unit_waveforms_density_map(
                    self.we_dense, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend]
                )

    def test_plot_unit_waveforms_density_map_sparsity_radius(self):
        possible_backends = list(sw.UnitWaveformDensityMapWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.sorting.unit_ids[:2]
                sw.plot_unit_waveforms_density_map(
                    self.we_dense,
                    sparsity=self.sparsity_radius,
                    same_axis=False,
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

    def test_plot_unit_waveforms_density_map_sparsity_None_same_axis(self):
        possible_backends = list(sw.UnitWaveformDensityMapWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.sorting.unit_ids[:2]
                sw.plot_unit_waveforms_density_map(
                    self.we_sparse,
                    sparsity=None,
                    same_axis=True,
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

    def test_plot_autocorrelograms(self):
        possible_backends = list(sw.AutoCorrelogramsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.sorting.unit_ids[:4]
                sw.plot_autocorrelograms(
                    self.sorting,
                    unit_ids=unit_ids,
                    window_ms=500.0,
                    bin_ms=20.0,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

    def test_plot_crosscorrelogram(self):
        possible_backends = list(sw.CrossCorrelogramsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.sorting.unit_ids[:4]
                sw.plot_crosscorrelograms(
                    self.sorting,
                    unit_ids=unit_ids,
                    window_ms=500.0,
                    bin_ms=20.0,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

    def test_plot_isi_distribution(self):
        possible_backends = list(sw.ISIDistributionWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.sorting.unit_ids[:4]
                sw.plot_isi_distribution(
                    self.sorting,
                    unit_ids=unit_ids,
                    window_ms=25.0,
                    bin_ms=2.0,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

    def test_plot_amplitudes(self):
        possible_backends = list(sw.AmplitudesWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_amplitudes(self.we_dense, backend=backend, **self.backend_kwargs[backend])
                unit_ids = self.we_dense.unit_ids[:4]
                sw.plot_amplitudes(self.we_dense, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend])
                sw.plot_amplitudes(
                    self.we_dense,
                    unit_ids=unit_ids,
                    plot_histograms=True,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_amplitudes(
                    self.we_sparse,
                    unit_ids=unit_ids,
                    plot_histograms=True,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

    def test_plot_all_amplitudes_distributions(self):
        possible_backends = list(sw.AllAmplitudesDistributionsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.we_dense.unit_ids[:4]
                sw.plot_all_amplitudes_distributions(
                    self.we_dense, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend]
                )
                sw.plot_all_amplitudes_distributions(
                    self.we_sparse, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend]
                )

    def test_plot_unit_locations(self):
        possible_backends = list(sw.UnitLocationsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_locations(
                    self.we_dense, with_channel_ids=True, backend=backend, **self.backend_kwargs[backend]
                )
                sw.plot_unit_locations(
                    self.we_sparse, with_channel_ids=True, backend=backend, **self.backend_kwargs[backend]
                )

    def test_plot_spike_locations(self):
        possible_backends = list(sw.SpikeLocationsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_spike_locations(
                    self.we_dense, with_channel_ids=True, backend=backend, **self.backend_kwargs[backend]
                )
                sw.plot_spike_locations(
                    self.we_sparse, with_channel_ids=True, backend=backend, **self.backend_kwargs[backend]
                )

    def test_plot_similarity(self):
        possible_backends = list(sw.TemplateSimilarityWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_template_similarity(self.we_dense, backend=backend, **self.backend_kwargs[backend])
                sw.plot_template_similarity(self.we_sparse, backend=backend, **self.backend_kwargs[backend])

    def test_plot_quality_metrics(self):
        possible_backends = list(sw.QualityMetricsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_quality_metrics(self.we_dense, backend=backend, **self.backend_kwargs[backend])
                sw.plot_quality_metrics(self.we_sparse, backend=backend, **self.backend_kwargs[backend])

    def test_plot_template_metrics(self):
        possible_backends = list(sw.TemplateMetricsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_template_metrics(self.we_dense, backend=backend, **self.backend_kwargs[backend])
                sw.plot_template_metrics(self.we_sparse, backend=backend, **self.backend_kwargs[backend])

    def test_plot_unit_depths(self):
        possible_backends = list(sw.UnitDepthsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_depths(self.we_dense, backend=backend, **self.backend_kwargs[backend])
                sw.plot_unit_depths(self.we_sparse, backend=backend, **self.backend_kwargs[backend])

    def test_plot_unit_summary(self):
        possible_backends = list(sw.UnitSummaryWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_summary(
                    self.we_dense, self.we_dense.sorting.unit_ids[0], backend=backend, **self.backend_kwargs[backend]
                )
                sw.plot_unit_summary(
                    self.we_sparse, self.we_sparse.sorting.unit_ids[0], backend=backend, **self.backend_kwargs[backend]
                )

    def test_plot_sorting_summary(self):
        possible_backends = list(sw.SortingSummaryWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_sorting_summary(self.we_dense, backend=backend, **self.backend_kwargs[backend])
                sw.plot_sorting_summary(self.we_sparse, backend=backend, **self.backend_kwargs[backend])

    def test_plot_agreement_matrix(self):
        possible_backends = list(sw.AgreementMatrixWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_agreement_matrix(self.gt_comp)

    def test_plot_confusion_matrix(self):
        possible_backends = list(sw.AgreementMatrixWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_confusion_matrix(self.gt_comp)

    def test_plot_probe_map(self):
        possible_backends = list(sw.ProbeMapWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_probe_map(self.recording, with_channel_ids=True, with_contact_id=True)

    def test_plot_rasters(self):
        possible_backends = list(sw.RasterWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_rasters(self.sorting)

    def test_plot_unit_probe_map(self):
        possible_backends = list(sw.UnitProbeMapWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_probe_map(self.we_dense)

    def test_plot_unit_presence(self):
        possible_backends = list(sw.UnitPresenceWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_presence(self.sorting)

    def test_plot_peak_activity(self):
        possible_backends = list(sw.PeakActivityMapWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_peak_activity(self.recording, self.peaks)

    def test_plot_multicomparison(self):
        mcmp = sc.compare_multiple_sorters([self.sorting, self.sorting, self.sorting])
        possible_backends_graph = list(sw.MultiCompGraphWidget.get_possible_backends())
        for backend in possible_backends_graph:
            sw.plot_multicomparison_graph(
                mcmp, edge_cmap="viridis", node_cmap="rainbow", draw_labels=False, backend=backend
            )
        possible_backends_glob = list(sw.MultiCompGlobalAgreementWidget.get_possible_backends())
        for backend in possible_backends_glob:
            sw.plot_multicomparison_agreement(mcmp, backend=backend)
        possible_backends_by_sorter = list(sw.MultiCompAgreementBySorterWidget.get_possible_backends())
        for backend in possible_backends_by_sorter:
            sw.plot_multicomparison_agreement_by_sorter(mcmp)
            if backend == "matplotlib":
                _, axes = plt.subplots(len(mcmp.object_list), 1)
                sw.plot_multicomparison_agreement_by_sorter(mcmp, axes=axes)


if __name__ == "__main__":
    # unittest.main()

    TestWidgets.setUpClass()
    mytest = TestWidgets()

    # mytest.test_plot_unit_waveforms_density_map()
    # mytest.test_plot_unit_summary()
    # mytest.test_plot_all_amplitudes_distributions()
    # mytest.test_plot_traces()
    # mytest.test_plot_unit_waveforms()
    # mytest.test_plot_unit_templates()
    mytest.test_plot_unit_templates()
    # mytest.test_plot_unit_depths()
    # mytest.test_plot_unit_templates()
    # mytest.test_plot_unit_summary()
    # mytest.test_crosscorrelogram()
    # mytest.test_isi_distribution()
    # mytest.test_unit_locations()
    # mytest.test_quality_metrics()
    # mytest.test_template_metrics()
    # mytest.test_amplitudes()
    # mytest.test_plot_agreement_matrix()
    # mytest.test_plot_confusion_matrix()
    # mytest.test_plot_probe_map()
    # mytest.test_plot_rasters()
    # mytest.test_plot_unit_probe_map()
    # mytest.test_plot_unit_presence()
    # mytest.test_plot_multicomparison()

    TestWidgets.tearDownClass()

    plt.show()
