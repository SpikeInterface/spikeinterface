import pytest

import numpy as np

import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection, TemporalPCADenoising
from spikeinterface.sortingcomponents.peak_pipeline import (
    ExtractDenseWaveforms,
    ExtractSparseWaveforms,
    WaveformExtractorNode,
    PipelineNode,
    run_peak_pipeline,
)
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

job_kwargs = dict(n_jobs=-1, chunk_size=10000, progress_bar=False)

@pytest.fixture
def mearec_recording():
    local_path = si.download_dataset(remote_path="mearec/mearec_test_10s.h5")
    recording, sorting = se.read_mearec(local_path)
    return recording


@pytest.fixture
def detected_peaks(mearec_recording):
    recording = mearec_recording
    peaks = detect_peaks(recording=recording, **job_kwargs)
    return peaks


@pytest.fixture
def model_path_of_trained_pca(tmp_path, mearec_recording):
    recording = mearec_recording

    # Parameters
    ms_before = 1.0
    ms_after = 1.0
    job_kwargs = dict(n_jobs=4, chunk_size=10000, progress_bar=True)
    model_folder_path = tmp_path / "temporal_pca_model"
    model_folder_path.mkdir()
    # model_folder_path.mkdir(parents=True, exist_ok=True)
    # Fit the model
    n_components = 3
    n_peaks = 100  # Heuristic for extracting around 1k waveforms per channel
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

    return model_folder_path

def test_pca_denoising(mearec_recording, detected_peaks, model_path_of_trained_pca):
    
    recording = mearec_recording
    model_folder_path = model_path_of_trained_pca
    peaks = detected_peaks

    # Parameters
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, ms_before=ms_before, ms_after=ms_after, return_output=True
    )
    pca_denoising = TemporalPCADenoising(
        recording=recording, model_folder_path=model_folder_path, parents=[extract_waveforms]
    )
    pipeline_nodes = [extract_waveforms, pca_denoising]  

    # Extract projected waveforms and compare
    waveforms, denoised_waveforms = run_peak_pipeline(recording, peaks, nodes=pipeline_nodes, job_kwargs=job_kwargs)
    assert waveforms.shape == denoised_waveforms.shape
    
def test_pca_denoising_sparse(mearec_recording, detected_peaks, model_path_of_trained_pca):
    
    recording = mearec_recording
    model_folder_path = model_path_of_trained_pca
    peaks = detected_peaks

    # Parameters
    local_radius_um = 40
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    extract_waveforms = ExtractSparseWaveforms(
        recording=recording, ms_before=ms_before, ms_after=ms_after, local_radius_um=local_radius_um, return_output=True
    )
    pca_denoising = TemporalPCADenoising(
        recording=recording, model_folder_path=model_folder_path, parents=[extract_waveforms]
    )
    pipeline_nodes = [extract_waveforms, pca_denoising]  

    # Extract projected waveforms and compare
    sparse_waveforms, denoised_waveforms = run_peak_pipeline(recording, peaks, nodes=pipeline_nodes, job_kwargs=job_kwargs)
    assert sparse_waveforms.shape == denoised_waveforms.shape
    
def test_pca_projection(mearec_recording, detected_peaks, model_path_of_trained_pca):

    recording = mearec_recording
    model_folder_path = model_path_of_trained_pca
    peaks = detected_peaks

    # Parameters
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, ms_before=ms_before, ms_after=ms_after, return_output=False
    )
    temporal_pca = TemporalPCAProjection(
        recording=recording, model_folder_path=model_folder_path, parents=[extract_waveforms]
    )
    pipeline_nodes = [extract_waveforms, temporal_pca]

    # Extract projected waveforms and compare
    projected_waveforms = run_peak_pipeline(recording, peaks, nodes=pipeline_nodes, job_kwargs=job_kwargs)
    extracted_n_peaks, extracted_n_components, extracted_n_channels = projected_waveforms.shape
    n_peaks = peaks.shape[0]
    assert extracted_n_peaks == n_peaks
    assert extracted_n_components == temporal_pca.pca_model.n_components
    assert extracted_n_channels == recording.get_num_channels()

def test_pca_projection_sparsity(mearec_recording, detected_peaks, model_path_of_trained_pca):

    recording = mearec_recording
    model_folder_path = model_path_of_trained_pca
    peaks = detected_peaks

    # Parameters
    local_radius_um = 40
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    extract_waveforms = ExtractSparseWaveforms(
        recording=recording, ms_before=ms_before, ms_after=ms_after, local_radius_um=local_radius_um, return_output=True
    )
    temporal_pca = TemporalPCAProjection(
        recording=recording, model_folder_path=model_folder_path, parents=[extract_waveforms]
    )
    pipeline_nodes = [extract_waveforms, temporal_pca]

    # Extract features and compare
    sparse_waveforms, projected_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=job_kwargs
    )
    extracted_n_peaks, extracted_n_components, extracted_n_channels = projected_waveforms.shape
    n_peaks = peaks.shape[0]
    max_n_channels = sparse_waveforms.shape[2]

    assert extracted_n_peaks == n_peaks
    assert extracted_n_components == temporal_pca.pca_model.n_components
    assert extracted_n_channels == max_n_channels

def test_initialization_with_wrong_parents_failure(mearec_recording, model_path_of_trained_pca):
    
    recording = mearec_recording
    model_folder_path = model_path_of_trained_pca
    dummy_parent = PipelineNode(recording=recording)
    extract_waveforms = ExtractSparseWaveforms(
        recording=recording, ms_before=1, ms_after=1, local_radius_um=40, return_output=True
    )
    
    match_error = f"TemporalPCA should have a single {WaveformExtractorNode.__name__} in its parents"

    # Parents without waveform extraction
    with pytest.raises(TypeError, match=match_error):
        TemporalPCAProjection(recording=recording, model_folder_path=model_folder_path, parents=[dummy_parent])

    # Empty parents
    with pytest.raises(TypeError, match=match_error):
        TemporalPCAProjection(recording=recording, model_folder_path=model_folder_path, parents=None)

    # Multiple parents
    with pytest.raises(TypeError, match=match_error):
        TemporalPCAProjection(recording=recording, model_folder_path=model_folder_path,
                              parents=[extract_waveforms, extract_waveforms])


def test_pca_waveform_extract_and_model_mismatch(mearec_recording, model_path_of_trained_pca):
    
    recording = mearec_recording
    model_folder_path = model_path_of_trained_pca

    # Node initialization
    ms_before = 0.5
    ms_after = 1.5
    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, ms_before=ms_before, ms_after=ms_after, return_output=False
    )
    # Pytest raise assertion
    with pytest.raises(ValueError, match="PCA model and waveforms mismatch *"):
        TemporalPCAProjection(recording=recording, model_folder_path=model_folder_path, parents=[extract_waveforms])
        
def test_pca_incorrect_model_path(mearec_recording, model_path_of_trained_pca):
    
    recording = mearec_recording
    model_folder_path = model_path_of_trained_pca / "a_file_that_does_not_exist.pkl"

    # Node initialization
    ms_before = 0.5
    ms_after = 1.5
    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, ms_before=ms_before, ms_after=ms_after, return_output=False
    )
    # Pytest raise assertion
    with pytest.raises(TypeError, match="model_path folder is not a folder or does not exist. *"):
        TemporalPCAProjection(recording=recording, model_folder_path=model_folder_path, parents=[extract_waveforms])