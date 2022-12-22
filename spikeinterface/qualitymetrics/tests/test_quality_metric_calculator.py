import unittest
import pytest
from pathlib import Path
import numpy as np

from spikeinterface import WaveformExtractor, ChannelSparsity, load_extractor, extract_waveforms
from spikeinterface.extractors import toy_example

from spikeinterface.postprocessing import WaveformPrincipalComponent
from spikeinterface.preprocessing import scale
from spikeinterface.qualitymetrics import QualityMetricCalculator
from spikeinterface.postprocessing import get_template_channel_sparsity

from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "qualitymetrics"
else:
    cache_folder = Path("cache_folder") / "qualitymetrics"


class QualityMetricsExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = QualityMetricCalculator
    extension_data_names = ["metrics"]
    extension_function_kwargs_list = [
        dict(),
        dict(n_jobs=2),
        dict(metric_names=["snr", "firing_rate"])
    ]

    def setUp(self):
        super().setUp()
        self.cache_folder = cache_folder
        recording, sorting = toy_example(num_segments=2, num_units=10, duration=300)
        if (cache_folder / 'toy_rec_long').is_dir():
            recording = load_extractor(self.cache_folder / 'toy_rec_long')
        else:
            recording = recording.save(folder=self.cache_folder / 'toy_rec_long')
        if (cache_folder / 'toy_sorting_long').is_dir():
            sorting = load_extractor(self.cache_folder / 'toy_sorting_long')
        else:
            sorting = sorting.save(folder=self.cache_folder / 'toy_sorting_long')
        we_long = extract_waveforms(recording, sorting,
                                    self.cache_folder / 'toy_waveforms_long',
                                    max_spikes_per_unit=None,
                                    overwrite=True)
        self.sparsity_long = ChannelSparsity.from_radius(we_long, radius_um=50)
        self.we_long = we_long

    def test_metrics(self):
        we = self.we_long

        # without PC
        metrics = self.extension_class.get_extension_function()(we, metric_names=['snr'])
        assert 'snr' in metrics.columns
        assert 'isolation_distance' not in metrics.columns
        # print(metrics)

        # with PCs
        pca = WaveformPrincipalComponent(we)
        pca.set_params(n_components=5, mode='by_channel_local')
        pca.run()
        metrics = self.extension_class.get_extension_function()(we)
        assert 'isolation_distance' in metrics.columns

        # with PC - parallel
        metrics_par = self.extension_class.get_extension_function()(
            we, n_jobs=2, verbose=True, progress_bar=True)
        # print(metrics)
        # print(metrics_par)
        for metric_name in metrics.columns:
            assert np.allclose(metrics[metric_name], metrics_par[metric_name])
        # print(metrics)

        # with sparsity
        metrics_sparse = self.extension_class.get_extension_function()(
            we, sparsity=self.sparsity_long, n_jobs=1)
        assert 'isolation_distance' in metrics_sparse.columns
        # for metric_name in metrics.columns:
        #     assert np.allclose(metrics[metric_name], metrics_par[metric_name])
        # print(metrics_sparse)

    def test_peak_sign(self):
        we = self.we_long
        rec = we.recording
        sort = we.sorting

        # invert recording
        rec_inv = scale(rec, gain=-1.)

        we_inv = WaveformExtractor.create(
            rec_inv, sort, self.cache_folder / 'toy_waveforms_inv')
        we_inv.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=None)
        we_inv.run_extract_waveforms(n_jobs=1, chunk_size=30000)
        print(we_inv)

        # without PC
        metrics = self.extension_class.get_extension_function()(
            we, metric_names=['snr', 'amplitude_cutoff'], peak_sign="neg")
        metrics_inv = self.extension_class.get_extension_function()(
            we_inv, metric_names=['snr', 'amplitude_cutoff'], peak_sign="pos")
        assert np.allclose(metrics["snr"].values,
                           metrics_inv["snr"].values, atol=1e-4)
        assert np.allclose(metrics["amplitude_cutoff"].values,
                           metrics_inv["amplitude_cutoff"].values, atol=1e-4)

if __name__ == '__main__':
    test = QualityMetricsExtensionTest
    test.setUp()
    test.test_extension()
    test.test_metrics()
    test.test_peak_sign()
