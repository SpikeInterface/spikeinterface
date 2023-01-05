import unittest
import pytest
import warnings
from pathlib import Path
import numpy as np

from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms, split_recording, select_segment_sorting
from spikeinterface.extractors import toy_example
from spikeinterface.core import get_template_channel_sparsity

from spikeinterface.postprocessing import compute_principal_components, compute_spike_amplitudes
from spikeinterface.preprocessing import scale
from spikeinterface.qualitymetrics import QualityMetricCalculator, get_default_qm_params

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
        recording, sorting = toy_example(num_segments=2, num_units=10, duration=120)
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
                                    max_spikes_per_unit=500,
                                    overwrite=True,
                                    seed=0)
        # make a short we for testing amp cutoff
        recording_one = split_recording(recording)[0]
        sorting_one = select_segment_sorting(sorting, [0])

        nsec_short = 30
        recording_short = recording_one.frame_slice(start_frame=0,
                                                    end_frame=int(nsec_short * recording.sampling_frequency))
        sorting_short = sorting_one.frame_slice(start_frame=0,
                                                end_frame=int(nsec_short * recording.sampling_frequency))
        we_short = extract_waveforms(recording_short, sorting_short,
                                     self.cache_folder / 'toy_waveforms_short',
                                     max_spikes_per_unit=500,
                                     overwrite=True,
                                     seed=0)
        self.sparsity_long = get_template_channel_sparsity(we_long, method="radius",
                                                           radius_um=50)
        self.we_long = we_long
        self.we_short = we_short

    def test_metrics(self):
        we = self.we_long

        # without PC
        metrics = self.extension_class.get_extension_function()(we, metric_names=['snr'])
        assert 'snr' in metrics.columns
        assert 'isolation_distance' not in metrics.columns
        metrics = self.extension_class.get_extension_function()(we, metric_names=['snr'],
                                                                qm_params=dict(isi_violations=dict(isi_threshold_ms=2)))
        # check that parameters are correctly set
        qm = we.load_extension("quality_metrics")
        assert qm._params["qm_params"]["isi_violations"]["isi_threshold_ms"] == 2
        assert 'snr' in metrics.columns
        assert 'isolation_distance' not in metrics.columns
        # print(metrics)

        # with PCs
        _ = compute_principal_components(we, n_components=5, mode='by_channel_local')
        metrics = self.extension_class.get_extension_function()(we, seed=0)
        assert 'isolation_distance' in metrics.columns

        # with PC - parallel
        metrics_par = self.extension_class.get_extension_function()(
            we, n_jobs=2, verbose=True, progress_bar=True, seed=0)
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

    def test_amplitude_cutoff(self):
        we = self.we_short
        _ = compute_spike_amplitudes(we, peak_sign="neg")

        # If too few spikes, should raise a warning and set amplitude cutoffs to nans
        with pytest.warns(UserWarning) as w:
            metrics = self.extension_class.get_extension_function()(
                we, metric_names=['amplitude_cutoff'], peak_sign="neg")
        assert all(np.isnan(cutoff) for cutoff in metrics["amplitude_cutoff"].values)

        # now we decrease the number of bins and check that amplitude cutoffs are correctly computed
        qm_params=dict(amplitude_cutoff=dict(num_histogram_bins=5))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            metrics = self.extension_class.get_extension_function()(
                we, metric_names=['amplitude_cutoff'], peak_sign="neg", qm_params=qm_params)
        assert all(not np.isnan(cutoff) for cutoff in metrics["amplitude_cutoff"].values)


    def test_presence_ratio(self):
        we = self.we_long

        total_duration = we.recording.get_total_duration()
        # If bin_duration_s is larger than total duration, should raise a warning and set presence ratios to nans
        qm_params=dict(presence_ratio=dict(bin_duration_s=total_duration+1))
        with pytest.warns(UserWarning) as w:
            metrics = self.extension_class.get_extension_function()(
                we, metric_names=['presence_ratio'], qm_params=qm_params)
        assert all(np.isnan(ratio) for ratio in metrics["presence_ratio"].values)

        # now we decrease the bin_duration_s and check that presenc ratios are correctly computed
        qm_params=dict(presence_ratio=dict(bin_duration_s=total_duration // 10))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            metrics = self.extension_class.get_extension_function()(
                we, metric_names=['presence_ratio'], qm_params=qm_params)
        assert all(not np.isnan(ratio) for ratio in metrics["presence_ratio"].values)

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

        # compute amplitudes
        _ = compute_spike_amplitudes(we, peak_sign="neg")
        _ = compute_spike_amplitudes(we_inv, peak_sign="pos")


        # without PC
        metrics = self.extension_class.get_extension_function()(
            we, metric_names=['snr', 'amplitude_cutoff'], peak_sign="neg")
        metrics_inv = self.extension_class.get_extension_function()(
            we_inv, metric_names=['snr', 'amplitude_cutoff'], peak_sign="pos")
        # print(metrics)
        # print(metrics_inv)
        # for SNR we allow a 5% tollerance because of waveform sub-sampling
        assert np.allclose(metrics["snr"].values,
                           metrics_inv["snr"].values, rtol=0.05)
        # for amplitude_cutoff, since spike amplitudes are computed, values should be exactly the same
        assert np.allclose(metrics["amplitude_cutoff"].values,
                           metrics_inv["amplitude_cutoff"].values, atol=1e-5)


if __name__ == '__main__':
    test = QualityMetricsExtensionTest()
    test.setUp()
    # test.test_extension()
    test.test_presence_ratio()
    # test.test_peak_sign()
