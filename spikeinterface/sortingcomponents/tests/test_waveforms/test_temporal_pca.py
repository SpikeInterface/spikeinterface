import pytest

import numpy as np

import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection
from spikeinterface.sortingcomponents.peak_pipeline import ExtractDenseWaveforms, ExtractSparseWaveforms
from spikeinterface.sortingcomponents.peak_detection import detect_peaks


def test_pca_projection(tmp_path):
    local_path = si.download_dataset(remote_path="mearec/mearec_test_10s.h5")
    recording, sorting = se.read_mearec(local_path)

    # Parameters
    ms_before = 1.0
    ms_after = 1.0
    job_kwargs = dict(n_jobs=4, chunk_size=10000, progress_bar=True)
    model_folder_path = tmp_path
    # Fit the model
    n_components = 3
    n_peaks = recording.get_num_channels() * 1e3  # Heuristic for extracting around 1k waveforms per channel
    peak_selection_params = dict(method="uniform", select_per_channel=True, n_peaks=n_peaks)
    detect_peaks_params = dict(method="by_channel", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1)
    TemporalPCAProjection.fit(
        recording=recording,
        model_folder_path=model_folder_path,
        n_components=n_components,
        ms_before=ms_before,
        ms_after=ms_after,
        detect_peaks_params=detect_peaks_params,
        peak_selection_params=peak_selection_params,
        job_kwargs=job_kwargs,
    )

    # Node initialization
    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, ms_before=ms_before, ms_after=ms_after, return_ouput=False
    )
    temporal_pca = TemporalPCAProjection(
        recording=recording, model_folder_path=model_folder_path, parents=[extract_waveforms]
    )
    pipeline_nodes = [extract_waveforms, temporal_pca]

    # Extract features and compare
    peaks, projected_waveforms = detect_peaks(recording, pipeline_nodes=pipeline_nodes, **job_kwargs)
    extracted_n_peaks, extracted_n_components, extracted_n_channels = projected_waveforms.shape
    n_peaks = peaks.shape[0]
    assert extracted_n_peaks == n_peaks
    assert extracted_n_components == n_components
    assert extracted_n_channels == recording.get_num_channels()
    assert extracted_n_channels == recording.get_num_channels()

def test_pca_projection_waveform_extract_and_model_mismatch(tmp_path):
    local_path = si.download_dataset(remote_path="mearec/mearec_test_10s.h5")
    recording, sorting = se.read_mearec(local_path)
    
    # Parameters
    ms_before = 1.0
    ms_after = 1.0
    job_kwargs = dict(n_jobs=4, chunk_size=10000, progress_bar=True)
    model_folder_path = tmp_path
    # Fit the model
    n_components = 3
    n_peaks = recording.get_num_channels() * 1e3  # Heuristic for extracting around 1k waveforms per channel
    peak_selection_params = dict(method="uniform", select_per_channel=True, n_peaks=n_peaks)
    detect_peaks_params = dict(method="by_channel", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1)
    TemporalPCAProjection.fit(
        recording=recording,
        model_folder_path=model_folder_path,
        n_components=n_components,
        ms_before=ms_before,
        ms_after=ms_after,
        detect_peaks_params=detect_peaks_params,
        peak_selection_params=peak_selection_params,
        job_kwargs=job_kwargs,
    )
    
    # Node initialization
    ms_before = 0.5
    ms_after = 1.5
    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, ms_before=ms_before, ms_after=ms_after, return_ouput=False
    )
    # Pytest raise assertion
    with pytest.raises(AttributeError):
        TemporalPCAProjection(
            recording=recording, model_folder_path=model_folder_path, parents=[extract_waveforms]
        )

    
def test_pca_projection_sparsity(tmp_path):
    local_path = si.download_dataset(remote_path="mearec/mearec_test_10s.h5")
    recording, sorting = se.read_mearec(local_path)

    # Parameters
    local_radius_um = 40
    ms_before = 1.0
    ms_after = 1.0
    job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)
    model_folder_path = tmp_path
    
    # Fit the model
    n_components = 8
    n_peaks = 100
    peak_selection_params = dict(method="uniform", select_per_channel=True, n_peaks=n_peaks)
    detect_peaks_params = dict(method="by_channel", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1)
    TemporalPCAProjection.fit(
        recording=recording,
        model_folder_path=model_folder_path,
        n_components=n_components,
        ms_before=ms_before,
        ms_after=ms_after,
        detect_peaks_params=detect_peaks_params,
        peak_selection_params=peak_selection_params,
        job_kwargs=job_kwargs,
    )

    # Node initialization
    extract_waveforms = ExtractSparseWaveforms(
        recording=recording, ms_before=ms_before, ms_after=ms_after, local_radius_um=local_radius_um, return_ouput=True
    )
    temporal_pca = TemporalPCAProjection(
        recording=recording, model_folder_path=model_folder_path, parents=[extract_waveforms]
    )
    pipeline_nodes = [extract_waveforms, temporal_pca]


    # Extract features and compare
    peaks, sparse_waveforms, projected_waveforms = detect_peaks(recording, pipeline_nodes=pipeline_nodes, **job_kwargs)
    extracted_n_peaks, extracted_n_components, extracted_n_channels = projected_waveforms.shape
    n_peaks = peaks.shape[0]
    max_n_channels = sparse_waveforms.shape[2]

    assert extracted_n_peaks == n_peaks
    assert extracted_n_components == n_components
    assert extracted_n_channels == max_n_channels
    
