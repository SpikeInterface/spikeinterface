import unittest
import shutil
from pathlib import Path
import numpy as np

import pytest

from spikeinterface import WaveformExtractor, load_extractor
from spikeinterface.extractors import toy_example

from spikeinterface.toolkit.postprocessing import WaveformPrincipalComponent
from spikeinterface.toolkit.preprocessing import scale
from spikeinterface.toolkit.qualitymetrics import compute_quality_metrics, QualityMetricCalculator


def setup_module():
    for folder in ('toy_rec', 'toy_sorting', 'toy_waveforms', 'toy_waveforms_filt',
                   'toy_waveforms_inv'):
        if Path(folder).is_dir():
            shutil.rmtree(folder)

    recording, sorting = toy_example(num_segments=2, num_units=10)
    recording = recording.save(folder='toy_rec')
    sorting = sorting.save(folder='toy_sorting')

    we = WaveformExtractor.create(recording, sorting, 'toy_waveforms')
    we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)
    we.run_extract_waveforms(n_jobs=1, chunk_size=30000)


def test_compute_quality_metrics():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
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
    print(metrics)

    # reload as an extension from we
    assert QualityMetricCalculator in we.get_available_extensions()
    assert we.is_extension('quality_metrics')
    qmc = we.load_extension('quality_metrics')
    assert isinstance(qmc, QualityMetricCalculator)
    assert qmc._metrics is not None
    # print(qmc._metrics)
    qmc = QualityMetricCalculator.load_from_folder('toy_waveforms')
    assert qmc._metrics is not None
    # print(qmc._metrics)
    

def test_compute_quality_metrics_peak_sign():
    rec = load_extractor('toy_rec')
    sort = load_extractor('toy_sorting')
    
    # invert recording
    rec_inv = scale(rec, gain=-1.)
    
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    print(we)
    
    we_inv = WaveformExtractor.create(rec_inv, sort, 'toy_waveforms_inv')
    we_inv.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)
    we_inv.run_extract_waveforms(n_jobs=1, chunk_size=30000)
    print(we_inv)

    # without PC
    metrics = compute_quality_metrics(we, metric_names=['snr', 'amplitude_cutoff'], peak_sign="neg")
    metrics_inv = compute_quality_metrics(we_inv, metric_names=['snr', 'amplitude_cutoff'], peak_sign="pos")
    
    assert np.allclose(metrics["snr"].values, metrics_inv["snr"].values)
    assert np.allclose(metrics["amplitude_cutoff"].values, metrics_inv["amplitude_cutoff"].values)


def test_select_units():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    qm = compute_quality_metrics(we, load_if_exists=True)

    keep_units = we.sorting.get_unit_ids()[::2]
    we_filt = we.select_units(keep_units, 'toy_waveforms_filt')
    assert "quality_metrics" in we_filt.get_available_extension_names()

if __name__ == '__main__':
    setup_module()
    test_compute_quality_metrics_peak_sign()
