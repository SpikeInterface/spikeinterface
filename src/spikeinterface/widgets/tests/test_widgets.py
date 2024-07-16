import unittest
import pytest
import os

import numpy as np

if __name__ != "__main__":
    try:
        import matplotlib

        matplotlib.use("Agg")
    except:
        pass


from spikeinterface import (
    compute_sparsity,
    generate_ground_truth_recording,
    create_sorting_analyzer,
)


import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
from spikeinterface.preprocessing import scale, correct_motion


ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))
KACHERY_CLOUD_SET = bool(os.getenv("KACHERY_CLOUD_CLIENT_ID")) and bool(os.getenv("KACHERY_CLOUD_PRIVATE_KEY"))
SKIP_SORTINGVIEW = bool(os.getenv("SKIP_SORTINGVIEW"))


class TestWidgets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

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
            noise_kwargs=dict(noise_levels=5.0, strategy="on_the_fly"),
            seed=2205,
        )
        # cls.recording = recording.save(folder=cache_folder / "recording")
        # cls.sorting = sorting.save(folder=cache_folder / "sorting")
        cls.recording = recording
        cls.sorting = sorting

        # estimate motion for motion widgets
        _, cls.motion_info = correct_motion(recording, preset="kilosort_like", output_motion_info=True)

        cls.num_units = len(cls.sorting.get_unit_ids())

        extensions_to_compute = dict(
            waveforms=dict(),
            templates=dict(),
            noise_levels=dict(),
            spike_amplitudes=dict(),
            unit_locations=dict(),
            spike_locations=dict(),
            quality_metrics=dict(metric_names=["snr", "isi_violation", "num_spikes"]),
            template_metrics=dict(),
            correlograms=dict(),
            template_similarity=dict(),
        )
        job_kwargs = dict(n_jobs=-1)

        # create dense
        cls.sorting_analyzer_dense = create_sorting_analyzer(cls.sorting, cls.recording, format="memory", sparse=False)
        cls.sorting_analyzer_dense.compute("random_spikes")
        cls.sorting_analyzer_dense.compute(extensions_to_compute, **job_kwargs)

        sw.set_default_plotter_backend("matplotlib")

        # make sparse waveforms
        cls.sparsity_radius = compute_sparsity(cls.sorting_analyzer_dense, method="radius", radius_um=50)
        cls.sparsity_strict = compute_sparsity(cls.sorting_analyzer_dense, method="radius", radius_um=20)
        cls.sparsity_large = compute_sparsity(cls.sorting_analyzer_dense, method="radius", radius_um=80)
        cls.sparsity_best = compute_sparsity(cls.sorting_analyzer_dense, method="best_channels", num_channels=5)

        # create sparse
        cls.sorting_analyzer_sparse = create_sorting_analyzer(
            cls.sorting, cls.recording, format="memory", sparsity=cls.sparsity_radius
        )
        cls.sorting_analyzer_sparse.compute("random_spikes")
        cls.sorting_analyzer_sparse.compute(extensions_to_compute, **job_kwargs)

        cls.skip_backends = ["ipywidgets", "ephyviewer", "spikeinterface_gui"]
        # cls.skip_backends = ["ipywidgets", "ephyviewer", "sortingview"]

        if (ON_GITHUB and not KACHERY_CLOUD_SET) or (SKIP_SORTINGVIEW):
            cls.skip_backends.append("sortingview")

        print(f"Widgets tests: skipping backends - {cls.skip_backends}")

        cls.backend_kwargs = {
            "matplotlib": {},
            "sortingview": {},
            "ipywidgets": {"display": False},
            "spikeinterface_gui": {},
        }

        cls.gt_comp = sc.compare_sorter_to_ground_truth(cls.sorting, cls.sorting)

        from spikeinterface.sortingcomponents.peak_detection import detect_peaks

        cls.peaks = detect_peaks(cls.recording, method="locally_exclusive", **job_kwargs)

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

    def test_plot_spikes_on_traces(self):
        possible_backends = list(sw.SpikesOnTracesWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_spikes_on_traces(self.sorting_analyzer_dense, backend=backend, **self.backend_kwargs[backend])

    def test_plot_unit_waveforms(self):
        possible_backends = list(sw.UnitWaveformsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_waveforms(self.sorting_analyzer_dense, backend=backend, **self.backend_kwargs[backend])
                unit_ids = self.sorting.unit_ids[:6]
                sw.plot_unit_waveforms(
                    self.sorting_analyzer_dense,
                    sparsity=self.sparsity_radius,
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_unit_waveforms(
                    self.sorting_analyzer_dense,
                    sparsity=self.sparsity_best,
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_unit_waveforms(
                    self.sorting_analyzer_sparse, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend]
                )
                # extra sparsity
                sw.plot_unit_waveforms(
                    self.sorting_analyzer_sparse,
                    sparsity=self.sparsity_strict,
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                # channel ids
                sw.plot_unit_waveforms(
                    self.sorting_analyzer_sparse,
                    channel_ids=self.sorting_analyzer_sparse.channel_ids[::3],
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                # test warning with "larger" sparsity
                with self.assertWarns(UserWarning):
                    sw.plot_unit_waveforms(
                        self.sorting_analyzer_sparse,
                        sparsity=self.sparsity_large,
                        unit_ids=unit_ids,
                        backend=backend,
                        **self.backend_kwargs[backend],
                    )

    def test_plot_unit_templates(self):
        possible_backends = list(sw.UnitTemplatesWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                print(f"Testing backend {backend}")
                # dense
                sw.plot_unit_templates(self.sorting_analyzer_dense, backend=backend, **self.backend_kwargs[backend])
                unit_ids = self.sorting.unit_ids[:6]
                # dense + radius
                sw.plot_unit_templates(
                    self.sorting_analyzer_dense,
                    sparsity=self.sparsity_radius,
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                # dense + best
                sw.plot_unit_templates(
                    self.sorting_analyzer_dense,
                    sparsity=self.sparsity_best,
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                # test different shadings
                sw.plot_unit_templates(
                    self.sorting_analyzer_sparse,
                    unit_ids=unit_ids,
                    templates_percentile_shading=None,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_unit_templates(
                    self.sorting_analyzer_sparse,
                    unit_ids=unit_ids,
                    # templates_percentile_shading=None,
                    scale=10,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_unit_templates(
                    self.sorting_analyzer_sparse,
                    unit_ids=unit_ids,
                    backend=backend,
                    templates_percentile_shading=None,
                    shade_templates=False,
                    **self.backend_kwargs[backend],
                )
                sw.plot_unit_templates(
                    self.sorting_analyzer_sparse,
                    unit_ids=unit_ids,
                    templates_percentile_shading=0.1,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                # extra sparsity
                sw.plot_unit_templates(
                    self.sorting_analyzer_sparse,
                    sparsity=self.sparsity_strict,
                    unit_ids=unit_ids,
                    templates_percentile_shading=[1, 10, 90, 99],
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                # channel ids
                sw.plot_unit_templates(
                    self.sorting_analyzer_sparse,
                    channel_ids=self.sorting_analyzer_sparse.channel_ids[::3],
                    unit_ids=unit_ids,
                    templates_percentile_shading=[1, 10, 90, 99],
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

                # test "larger" sparsity
                with self.assertWarns(UserWarning):
                    sw.plot_unit_templates(
                        self.sorting_analyzer_sparse,
                        sparsity=self.sparsity_large,
                        unit_ids=unit_ids,
                        backend=backend,
                        **self.backend_kwargs[backend],
                    )
                if backend != "sortingview":
                    sw.plot_unit_templates(
                        self.sorting_analyzer_sparse,
                        unit_ids=unit_ids,
                        templates_percentile_shading=[1, 5, 25, 75, 95, 99],
                        backend=backend,
                        **self.backend_kwargs[backend],
                    )
                    # test with templates
                    templates_ext = self.sorting_analyzer_dense.get_extension("templates")
                    templates = templates_ext.get_data(outputs="Templates")
                    sw.plot_unit_templates(
                        templates,
                        sparsity=self.sparsity_strict,
                        unit_ids=unit_ids,
                        backend=backend,
                        **self.backend_kwargs[backend],
                    )
                else:
                    # sortingview doesn't support more than 2 shadings
                    with self.assertRaises(AssertionError):
                        sw.plot_unit_templates(
                            self.sorting_analyzer_sparse,
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

                # on dense
                sw.plot_unit_waveforms_density_map(
                    self.sorting_analyzer_dense, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend]
                )
                # on sparse
                sw.plot_unit_waveforms_density_map(
                    self.sorting_analyzer_sparse, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend]
                )

                # externals parsity
                sw.plot_unit_waveforms_density_map(
                    self.sorting_analyzer_dense,
                    sparsity=self.sparsity_radius,
                    same_axis=False,
                    unit_ids=unit_ids,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

                # on sparse with same_axis
                sw.plot_unit_waveforms_density_map(
                    self.sorting_analyzer_sparse,
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

    def test_plot_crosscorrelograms(self):
        possible_backends = list(sw.CrossCorrelogramsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_crosscorrelograms(
                    self.sorting,
                    window_ms=500.0,
                    bin_ms=20.0,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                unit_ids = self.sorting.unit_ids[:4]
                sw.plot_crosscorrelograms(
                    self.sorting,
                    unit_ids=unit_ids,
                    window_ms=500.0,
                    bin_ms=20.0,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_crosscorrelograms(
                    self.sorting_analyzer_sparse,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_crosscorrelograms(
                    self.sorting_analyzer_sparse,
                    min_similarity_for_correlograms=0.6,
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
                sw.plot_amplitudes(self.sorting_analyzer_dense, backend=backend, **self.backend_kwargs[backend])
                unit_ids = self.sorting_analyzer_dense.unit_ids[:4]
                sw.plot_amplitudes(
                    self.sorting_analyzer_dense, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend]
                )
                sw.plot_amplitudes(
                    self.sorting_analyzer_dense,
                    unit_ids=unit_ids,
                    plot_histograms=True,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_amplitudes(
                    self.sorting_analyzer_sparse,
                    unit_ids=unit_ids,
                    plot_histograms=True,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

    def test_plot_all_amplitudes_distributions(self):
        possible_backends = list(sw.AllAmplitudesDistributionsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                unit_ids = self.sorting_analyzer_dense.unit_ids[:4]
                sw.plot_all_amplitudes_distributions(
                    self.sorting_analyzer_dense, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend]
                )
                sw.plot_all_amplitudes_distributions(
                    self.sorting_analyzer_sparse, unit_ids=unit_ids, backend=backend, **self.backend_kwargs[backend]
                )

    def test_plot_unit_locations(self):
        possible_backends = list(sw.UnitLocationsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_locations(
                    self.sorting_analyzer_dense, with_channel_ids=True, backend=backend, **self.backend_kwargs[backend]
                )
                sw.plot_unit_locations(
                    self.sorting_analyzer_sparse, with_channel_ids=True, backend=backend, **self.backend_kwargs[backend]
                )

    def test_plot_spike_locations(self):
        possible_backends = list(sw.SpikeLocationsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_spike_locations(
                    self.sorting_analyzer_dense, with_channel_ids=True, backend=backend, **self.backend_kwargs[backend]
                )
                sw.plot_spike_locations(
                    self.sorting_analyzer_sparse, with_channel_ids=True, backend=backend, **self.backend_kwargs[backend]
                )

    def test_plot_similarity(self):
        possible_backends = list(sw.TemplateSimilarityWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_template_similarity(
                    self.sorting_analyzer_dense, backend=backend, **self.backend_kwargs[backend]
                )
                sw.plot_template_similarity(
                    self.sorting_analyzer_sparse, backend=backend, **self.backend_kwargs[backend]
                )

    def test_plot_quality_metrics(self):
        possible_backends = list(sw.QualityMetricsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_quality_metrics(self.sorting_analyzer_dense, backend=backend, **self.backend_kwargs[backend])
                sw.plot_quality_metrics(self.sorting_analyzer_sparse, backend=backend, **self.backend_kwargs[backend])

    def test_plot_template_metrics(self):
        possible_backends = list(sw.TemplateMetricsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_template_metrics(self.sorting_analyzer_dense, backend=backend, **self.backend_kwargs[backend])
                sw.plot_template_metrics(self.sorting_analyzer_sparse, backend=backend, **self.backend_kwargs[backend])

    def test_plot_unit_depths(self):
        possible_backends = list(sw.UnitDepthsWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_depths(self.sorting_analyzer_dense, backend=backend, **self.backend_kwargs[backend])
                sw.plot_unit_depths(self.sorting_analyzer_sparse, backend=backend, **self.backend_kwargs[backend])

    def test_plot_unit_summary(self):
        possible_backends = list(sw.UnitSummaryWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_unit_summary(
                    self.sorting_analyzer_dense,
                    self.sorting_analyzer_dense.sorting.unit_ids[0],
                    backend=backend,
                    **self.backend_kwargs[backend],
                )
                sw.plot_unit_summary(
                    self.sorting_analyzer_sparse,
                    self.sorting_analyzer_sparse.sorting.unit_ids[0],
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

    def test_plot_sorting_summary(self):
        possible_backends = list(sw.SortingSummaryWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_sorting_summary(self.sorting_analyzer_dense, backend=backend, **self.backend_kwargs[backend])
                sw.plot_sorting_summary(self.sorting_analyzer_sparse, backend=backend, **self.backend_kwargs[backend])
                sw.plot_sorting_summary(
                    self.sorting_analyzer_sparse,
                    sparsity=self.sparsity_strict,
                    backend=backend,
                    **self.backend_kwargs[backend],
                )

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
                sw.plot_unit_probe_map(self.sorting_analyzer_dense)

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
                import matplotlib.pyplot as plt

                _, axes = plt.subplots(len(mcmp.object_list), 1)
                sw.plot_multicomparison_agreement_by_sorter(mcmp, axes=axes)

    def test_plot_motion(self):
        motion = self.motion_info["motion"]

        possible_backends = list(sw.MotionWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_motion(motion, backend=backend, mode="line")
                sw.plot_motion(motion, backend=backend, mode="map")

    def test_drift_raster_map(self):
        peaks = self.motion_info["peaks"]
        recording = self.recording
        peak_locations = self.motion_info["peak_locations"]
        analyzer = self.sorting_analyzer_sparse

        possible_backends = list(sw.MotionWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                # with recording
                sw.plot_drift_raster_map(
                    peaks=peaks, peak_locations=peak_locations, recording=recording, color_amplitude=True
                )
                # without recording
                sw.plot_drift_raster_map(
                    peaks=peaks,
                    peak_locations=peak_locations,
                    sampling_frequency=recording.sampling_frequency,
                    color_amplitude=False,
                )
                # with analyzer
                sw.plot_drift_raster_map(sorting_analyzer=analyzer, color_amplitude=True, scatter_decimate=2)

    def test_plot_motion_info(self):
        motion_info = self.motion_info
        possible_backends = list(sw.MotionWidget.get_possible_backends())
        for backend in possible_backends:
            if backend not in self.skip_backends:
                sw.plot_motion_info(motion_info, recording=self.recording, backend=backend)


if __name__ == "__main__":
    # unittest.main()
    import matplotlib.pyplot as plt

    TestWidgets.setUpClass()
    mytest = TestWidgets()

    # mytest.test_plot_unit_waveforms_density_map()
    # mytest.test_plot_unit_summary()
    # mytest.test_plot_all_amplitudes_distributions()
    # mytest.test_plot_traces()
    # mytest.test_plot_spikes_on_traces()
    # mytest.test_plot_unit_waveforms()
    # mytest.test_plot_spikes_on_traces()
    # mytest.test_plot_unit_depths()
    # mytest.test_plot_autocorrelograms()
    # mytest.test_plot_crosscorrelograms()
    # mytest.test_plot_isi_distribution()
    # mytest.test_plot_unit_locations()
    # mytest.test_plot_spike_locations()
    # mytest.test_plot_similarity()
    # mytest.test_plot_quality_metrics()
    # mytest.test_plot_template_metrics()
    # mytest.test_plot_amplitudes()
    # mytest.test_plot_agreement_matrix()
    # mytest.test_plot_confusion_matrix()
    # mytest.test_plot_probe_map()
    # mytest.test_plot_rasters()
    # mytest.test_plot_unit_probe_map()
    # mytest.test_plot_unit_presence()
    # mytest.test_plot_peak_activity()
    # mytest.test_plot_multicomparison()
    # mytest.test_plot_sorting_summary()
    # mytest.test_plot_motion()
    mytest.test_plot_motion_info()
    plt.show()

    # TestWidgets.tearDownClass()
