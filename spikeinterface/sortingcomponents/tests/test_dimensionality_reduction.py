
import pytest

from spikeinterface.sortingcomponents.dimensionality_reduction import TemporalPCA

import spikeinterface as si
import spikeinterface.extractors as se

from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline
from spikeinterface.sortingcomponents.peak_detection import detect_peaks


def test_dimensionality_reduction(tmp_path):
    local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting = se.read_mearec(local_path)

    local_radius_um = 1

    model_path = tmp_path / "buffer_pca.pkl"
    temporal_pca = TemporalPCA(recording, model_path=model_path, local_radius_um=local_radius_um)

    n_components = 3
    job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)
    detect_peaks_params = dict(method='by_channel', peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1)
    temporal_pca.fit(recording, n_components, detect_peaks_params, job_kwargs)

    steps = [temporal_pca]
    peaks, projected_waveforms = detect_peaks(recording, pipeline_steps=steps)
    extracted_n_peaks, extracted_n_components, extracted_n_channels =  projected_waveforms.shape
    
    n_peaks = peaks.shape[0]
    assert extracted_n_peaks == n_peaks
    assert extracted_n_components == n_components
    assert extracted_n_channels == recording.get_num_channels()

if __name__ == '__main__':
    test_dimensionality_reduction()