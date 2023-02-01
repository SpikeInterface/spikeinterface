
import pytest

import numpy as np

import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.sortingcomponents.waveforms.pca import PCAProjection, PCBaseNode
from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline, ExtractDenseWaveforms
from spikeinterface.sortingcomponents.peak_detection import detect_peaks


def test_pca_dimensionality_reduction(tmp_path):
    local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting = se.read_mearec(local_path)
    
    # Parameters
    local_radius_um = 100
    ms_before = 1.0
    ms_after = 1.0
    job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)
    model_path = tmp_path / "buffer_pca.pkl"

    # Node initialization
    extract_waveforms = ExtractDenseWaveforms(recording=recording, name='extract_dense_waveforms',ms_before=ms_before,
                                              ms_after=ms_after, return_ouput=False)
    temporal_pca = PCAProjection(recording=recording, model_path=model_path, parents=["extract_dense_waveforms"],
                                 local_radius_um=local_radius_um)
    pipeline_nodes = [extract_waveforms, temporal_pca]

    # Fit the model
    n_components = 3
    n_peaks = recording.get_num_channels() * 1e3 # Heuristic for extracting around 1k waveforms per channel
    peak_selection_params = dict(method="uniform", select_per_channel=True,  n_peaks=n_peaks)
    detect_peaks_params = dict(method='by_channel', peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1)
    temporal_pca.fit(recording, n_components, detect_peaks_params, peak_selection_params, job_kwargs)

    # Extract features and compare
    peaks, projected_waveforms = detect_peaks(recording, pipeline_nodes=pipeline_nodes, **job_kwargs)
    extracted_n_peaks, extracted_n_components, extracted_n_channels =  projected_waveforms.shape    
    n_peaks = peaks.shape[0]
    assert extracted_n_peaks == n_peaks
    assert extracted_n_components == n_components
    assert extracted_n_channels == recording.get_num_channels()
    
def test_pca_dimensionality_reduction_parallel(tmp_path):
    local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting = se.read_mearec(local_path)

    # Parameters
    local_radius_um = 100
    ms_before = 1.0
    ms_after = 1.0
    job_kwargs = dict(n_jobs=4, chunk_size=10000, progress_bar=True)
    model_path = tmp_path / "buffer_pca.pkl"

    # Node initialization
    extract_waveforms = ExtractDenseWaveforms(recording=recording, name='extract_dense_waveforms',ms_before=ms_before,
                                              ms_after=ms_after, return_ouput=False)
    temporal_pca = PCAProjection(recording=recording, model_path=model_path, parents=["extract_dense_waveforms"],
                                 local_radius_um=local_radius_um)
    pipeline_nodes = [extract_waveforms, temporal_pca]

    # Fit the model
    n_components = 3
    n_peaks = recording.get_num_channels() * 1e3 # Heuristic for extracting around 1k waveforms per channel
    peak_selection_params = dict(method="uniform", select_per_channel=True,  n_peaks=n_peaks)
    detect_peaks_params = dict(method='by_channel', peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1)
    temporal_pca.fit(recording, n_components, detect_peaks_params, peak_selection_params, job_kwargs)

    # Extract features and compare
    peaks, projected_waveforms = detect_peaks(recording, pipeline_nodes=pipeline_nodes, **job_kwargs)
    extracted_n_peaks, extracted_n_components, extracted_n_channels =  projected_waveforms.shape    
    n_peaks = peaks.shape[0]
    assert extracted_n_peaks == n_peaks
    assert extracted_n_components == n_components
    assert extracted_n_channels == recording.get_num_channels()
    assert extracted_n_channels == recording.get_num_channels()
    
    
def test_pca_dimensionality_reduction_sparsity(tmp_path):
    local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
    recording, sorting = se.read_mearec(local_path)
    
    # Parameters
    local_radius_um = 40
    ms_before = 1.0
    ms_after = 1.0
    job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)
    model_path = tmp_path / "buffer_pca.pkl"

    # Node initialization
    extract_waveforms = ExtractDenseWaveforms(recording=recording, name='extract_dense_waveforms',ms_before=ms_before,
                                              ms_after=ms_after, return_ouput=False)
    temporal_pca = PCAProjection(recording=recording, model_path=model_path, parents=["extract_dense_waveforms"],
                                 local_radius_um=local_radius_um)
    pipeline_nodes = [extract_waveforms, temporal_pca]

    # Fit the model
    n_components = 8
    n_peaks = 100
    peak_selection_params = dict(method="uniform", select_per_channel=True,  n_peaks=n_peaks)
    detect_peaks_params = dict(method='by_channel', peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1)
    temporal_pca.fit(recording, n_components, detect_peaks_params, peak_selection_params, job_kwargs)

    # Extract features and compare
    peaks, projected_waveforms = detect_peaks(recording, pipeline_nodes=pipeline_nodes, **job_kwargs)
    extracted_n_peaks, extracted_n_components, extracted_n_channels =  projected_waveforms.shape    
    
    # We represent sparsity with a dict where the key is the channel and the value is a list of `valid_channels`
    sparsity_dict = dict()
    channel_array, valid_channel_array = np.nonzero(temporal_pca.neighbours_mask)
    for channel, valid_channel in zip(channel_array, valid_channel_array):
        sparsity_dict.setdefault(channel, []).append(valid_channel)
    
    for peak_index in range(extracted_n_peaks):
        main_peak_channel = peaks[peak_index]['channel_ind']
        expected_sparsity = sparsity_dict[main_peak_channel]
        projected_waveform  = projected_waveforms[peak_index, 0, :]
        # All the channels that are invalid should be NaN
        invalid_channels = ~np.isnan(projected_waveform)
        waveform_sparsity, = np.nonzero(invalid_channels)    
        # Assert sparsity    
        np.testing.assert_array_equal(waveform_sparsity, expected_sparsity)

