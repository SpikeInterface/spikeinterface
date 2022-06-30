import pytest
import shutil
from pathlib import Path
import numpy as np

from spikeinterface import WaveformExtractor, load_extractor
from spikeinterface.extractors import toy_example

from spikeinterface.postprocessing import WaveformPrincipalComponent
from spikeinterface.preprocessing import scale
from spikeinterface.qualitymetrics import compute_quality_metrics, QualityMetricCalculator
from spikeinterface.postprocessing import get_template_channel_sparsity


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "qualitymetrics"
else:
    cache_folder = Path("cache_folder") / "qualitymetrics"


def setup_module():
    for folder_name in ('toy_rec', 'toy_sorting', 'toy_waveforms', 'toy_waveforms_filt',
                        'toy_waveforms_inv'):
        if (cache_folder / folder_name).is_dir():
            shutil.rmtree(cache_folder / folder_name)

    recording, sorting = toy_example(num_segments=2, num_units=10, duration=300)
    recording = recording.save(folder=cache_folder / 'toy_rec')
    sorting = sorting.save(folder=cache_folder / 'toy_sorting')

    we = WaveformExtractor.create(
        recording, sorting, cache_folder / 'toy_waveforms')
    we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=None)
    we.run_extract_waveforms(n_jobs=1, chunk_size=30000)


def test_compute_quality_metrics():
    we = WaveformExtractor.load_from_folder(cache_folder / 'toy_waveforms')
    print(we)

    # without PC
    metrics = compute_quality_metrics(we, metric_names=['snr'])
    assert 'snr' in metrics.columns
    assert 'isolation_distance' not in metrics.columns
    print(metrics)

    # with PCs
    pca = WaveformPrincipalComponent(we)
    pca.set_params(n_components=5, mode='by_channel_local')
    pca.run()
    metrics = compute_quality_metrics(we)
    assert 'isolation_distance' in metrics.columns
    
    # with PC - parallel
    metrics_par = compute_quality_metrics(we, n_jobs=2, verbose=True, progress_bar=True)
    for metric_name in metrics.columns:
        assert np.allclose(metrics[metric_name], metrics_par[metric_name])
    print(metrics)
    
    # with sparsity
    sparsity = get_template_channel_sparsity(we, method="radius", radius_um=20)
    print(sparsity)
    # test parallel
    metrics_sparse = compute_quality_metrics(we, sparsity=sparsity, n_jobs=1)
    assert 'isolation_distance' in metrics_sparse.columns
    # for metric_name in metrics.columns:
    #     assert np.allclose(metrics[metric_name], metrics_par[metric_name])
    print(metrics_sparse)

    # reload as an extension from we
    assert QualityMetricCalculator in we.get_available_extensions()
    assert we.is_extension('quality_metrics')
    qmc = we.load_extension('quality_metrics')
    assert isinstance(qmc, QualityMetricCalculator)
    assert qmc._metrics is not None
    # print(qmc._metrics)
    qmc = QualityMetricCalculator.load_from_folder(
        cache_folder / 'toy_waveforms')
    assert qmc._metrics is not None
    # print(qmc._metrics)


def test_compute_quality_metrics_peak_sign():
    rec = load_extractor(cache_folder / 'toy_rec')
    sort = load_extractor(cache_folder / 'toy_sorting')

    # invert recording
    rec_inv = scale(rec, gain=-1.)

    we = WaveformExtractor.load_from_folder(cache_folder / 'toy_waveforms')
    print(we)

    we_inv = WaveformExtractor.create(
        rec_inv, sort, cache_folder / 'toy_waveforms_inv')
    we_inv.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=None)
    we_inv.run_extract_waveforms(n_jobs=1, chunk_size=30000)
    print(we_inv)

    # without PC
    metrics = compute_quality_metrics(
        we, metric_names=['snr', 'amplitude_cutoff'], peak_sign="neg")
    metrics_inv = compute_quality_metrics(
        we_inv, metric_names=['snr', 'amplitude_cutoff'], peak_sign="pos")

    assert np.allclose(metrics["snr"].values, metrics_inv["snr"].values, atol=1e-4)
    assert np.allclose(metrics["amplitude_cutoff"].values,
                       metrics_inv["amplitude_cutoff"].values, atol=1e-4)


def test_select_units():
    we = WaveformExtractor.load_from_folder(cache_folder / 'toy_waveforms')
    qm = compute_quality_metrics(we, load_if_exists=True)

    keep_units = we.sorting.get_unit_ids()[::2]
    we_filt = we.select_units(keep_units, cache_folder / 'toy_waveforms_filt')
    assert "quality_metrics" in we_filt.get_available_extension_names()


if __name__ == '__main__':
    setup_module()
    test_compute_quality_metrics()
    # test_compute_quality_metrics_peak_sign()
