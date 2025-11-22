import pytest

import numpy as np

from spikeinterface.sortingcomponents.waveforms.temporal_pca import (
    TemporalPCAProjection,
    TemporalPCADenoising,
    TemporalPCAProjectionByChannel,
)
from spikeinterface.core.node_pipeline import (
    PeakRetriever,
    SpikeRetriever,
    ExtractDenseWaveforms,
    ExtractSparseWaveforms,
    WaveformsNode,
    PipelineNode,
    run_node_pipeline,
)


@pytest.fixture(scope="module")
def folder_to_save_pca_model(tmp_path_factory):
    """
    Creates a temporary folder to save the PCA model which is available to all the tests in this module
    """
    model_folder_path = tmp_path_factory.mktemp("my_temp_dir") / "temporal_pca_model"
    model_folder_path.mkdir()
    return model_folder_path


@pytest.fixture(scope="module")
def model_path_of_trained_pca(folder_to_save_pca_model, generated_recording, chunk_executor_kwargs):
    """
    Trains a pca model and makes its folder available to all the tests in this module.
    """
    recording, _ = generated_recording

    # Parameters
    ms_before = 1.0
    ms_after = 1.0
    model_folder_path = folder_to_save_pca_model
    # model_folder_path.mkdir(parents=True, exist_ok=True)
    # Fit the model
    n_components = 3
    n_peaks = 100  # Heuristic for extracting around 1k waveforms per channel
    peak_selection_params = dict(method="uniform", select_per_channel=True, n_peaks=n_peaks)
    detect_peaks_params = dict(method="by_channel", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1)
    model_folder_path, pca_model = TemporalPCAProjection.fit(
        recording=recording,
        model_folder_path=model_folder_path,
        n_components=n_components,
        ms_before=ms_before,
        ms_after=ms_after,
        detect_peaks_params=detect_peaks_params,
        peak_selection_params=peak_selection_params,
        job_kwargs=chunk_executor_kwargs,
    )

    return model_folder_path


@pytest.fixture(scope="module")
def pca_models_fit_by_channel(folder_to_save_pca_model, generated_recording, chunk_executor_kwargs):
    """
    Trains a pca model and makes its folder available to all the tests in this module.
    """
    recording, _ = generated_recording

    # Parameters
    ms_before = 1.0
    ms_after = 1.0
    model_folder_path = folder_to_save_pca_model
    # model_folder_path.mkdir(parents=True, exist_ok=True)
    # Fit the model
    n_components = 3
    n_peaks = 100  # Heuristic for extracting around 1k waveforms per channel
    peak_selection_params = dict(method="uniform", select_per_channel=True, n_peaks=n_peaks)
    detect_peaks_params = dict(method="by_channel", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1)
    _, pca_models = TemporalPCAProjection.fit(
        recording=recording,
        model_folder_path=model_folder_path,
        n_components=n_components,
        ms_before=ms_before,
        ms_after=ms_after,
        detect_peaks_params=detect_peaks_params,
        peak_selection_params=peak_selection_params,
        job_kwargs=chunk_executor_kwargs,
        mode="by_channel_local",
    )

    return pca_models


def test_pca_denoising(generated_recording, detected_peaks, model_path_of_trained_pca, chunk_executor_kwargs):
    recording, _ = generated_recording
    model_folder_path = model_path_of_trained_pca
    peaks = detected_peaks

    # Parameters
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    peak_retriever = PeakRetriever(recording, peaks)
    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=True
    )
    pca_denoising = TemporalPCADenoising(
        recording=recording, model_folder_path=model_folder_path, parents=[peak_retriever, extract_waveforms]
    )
    pipeline_nodes = [peak_retriever, extract_waveforms, pca_denoising]

    # Extract projected waveforms and compare
    waveforms, denoised_waveforms = run_node_pipeline(recording, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs)
    assert waveforms.shape == denoised_waveforms.shape


def test_pca_denoising_sparse(generated_recording, detected_peaks, model_path_of_trained_pca, chunk_executor_kwargs):
    recording, _ = generated_recording
    model_folder_path = model_path_of_trained_pca
    peaks = detected_peaks

    # Parameters
    radius_um = 40
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    peak_retriever = PeakRetriever(recording, peaks)
    extract_waveforms = ExtractSparseWaveforms(
        recording=recording,
        parents=[peak_retriever],
        ms_before=ms_before,
        ms_after=ms_after,
        radius_um=radius_um,
        return_output=True,
    )
    pca_denoising = TemporalPCADenoising(
        recording=recording, model_folder_path=model_folder_path, parents=[peak_retriever, extract_waveforms]
    )
    pipeline_nodes = [peak_retriever, extract_waveforms, pca_denoising]

    # Extract projected waveforms and compare
    sparse_waveforms, denoised_waveforms = run_node_pipeline(
        recording, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )
    assert sparse_waveforms.shape == denoised_waveforms.shape


def test_pca_projection(generated_recording, detected_peaks, model_path_of_trained_pca, chunk_executor_kwargs):
    recording, _ = generated_recording
    model_folder_path = model_path_of_trained_pca
    peaks = detected_peaks

    # Parameters
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    peak_retriever = PeakRetriever(recording, peaks)
    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=False
    )
    temporal_pca = TemporalPCAProjection(
        recording=recording, model_folder_path=model_folder_path, parents=[peak_retriever, extract_waveforms]
    )
    pipeline_nodes = [peak_retriever, extract_waveforms, temporal_pca]

    # Extract projected waveforms and compare
    projected_waveforms = run_node_pipeline(recording, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs)
    extracted_n_peaks, extracted_n_components, extracted_n_channels = projected_waveforms.shape
    n_peaks = peaks.shape[0]
    assert extracted_n_peaks == n_peaks
    assert extracted_n_components == temporal_pca.pca_model.n_components
    assert extracted_n_channels == recording.get_num_channels()


def test_pca_projection_sparsity(generated_recording, detected_peaks, model_path_of_trained_pca, chunk_executor_kwargs):
    recording, _ = generated_recording
    model_folder_path = model_path_of_trained_pca
    peaks = detected_peaks

    # Parameters
    radius_um = 40
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    peak_retriever = PeakRetriever(recording, peaks)
    extract_waveforms = ExtractSparseWaveforms(
        recording=recording,
        parents=[peak_retriever],
        ms_before=ms_before,
        ms_after=ms_after,
        radius_um=radius_um,
        return_output=True,
    )
    temporal_pca = TemporalPCAProjection(
        recording=recording, model_folder_path=model_folder_path, parents=[peak_retriever, extract_waveforms]
    )
    pipeline_nodes = [peak_retriever, extract_waveforms, temporal_pca]

    # Extract features and compare

    sparse_waveforms, projected_waveforms = run_node_pipeline(
        recording, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )
    extracted_n_peaks, extracted_n_components, extracted_n_channels = projected_waveforms.shape
    n_peaks = peaks.shape[0]
    max_n_channels = sparse_waveforms.shape[2]

    assert extracted_n_peaks == n_peaks
    assert extracted_n_components == temporal_pca.pca_model.n_components
    assert extracted_n_channels == max_n_channels


def test_pca_projection_by_channel(
    generated_recording, detected_peaks, pca_models_fit_by_channel, chunk_executor_kwargs
):
    recording, _ = generated_recording
    pca_models = pca_models_fit_by_channel
    peaks = detected_peaks

    # Parameters
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    peak_retriever = PeakRetriever(recording, peaks)
    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=False
    )
    temporal_pca = TemporalPCAProjectionByChannel(
        recording=recording, pca_models=pca_models, parents=[peak_retriever, extract_waveforms]
    )
    pipeline_nodes = [peak_retriever, extract_waveforms, temporal_pca]

    # Extract projected waveforms and compare
    projected_waveforms = run_node_pipeline(recording, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs)
    extracted_n_peaks, extracted_n_components, extracted_n_channels = projected_waveforms.shape
    n_peaks = peaks.shape[0]
    assert extracted_n_peaks == n_peaks
    assert extracted_n_components == temporal_pca.pca_model[0].n_components
    assert extracted_n_channels == recording.get_num_channels()


def test_pca_projection_by_channel_sparse(generated_recording, pca_models_fit_by_channel, chunk_executor_kwargs):
    recording, sorting = generated_recording
    pca_models = pca_models_fit_by_channel
    spikes = sorting.to_spike_vector()

    # Parameters
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    extremum_channel_inds = {unit_id: 0 for unit_id in sorting.unit_ids}
    peak_retriever = SpikeRetriever(sorting, recording, extremum_channel_inds=extremum_channel_inds)
    extract_waveforms = ExtractSparseWaveforms(
        recording=recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=False
    )
    temporal_pca = TemporalPCAProjectionByChannel(
        recording=recording, pca_models=pca_models, parents=[peak_retriever, extract_waveforms]
    )
    pipeline_nodes = [peak_retriever, extract_waveforms, temporal_pca]

    # Extract projected waveforms and compare
    chunk_executor_kwargs["n_jobs"] = 1  # for sparse, force to
    projected_waveforms = run_node_pipeline(recording, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs)
    extracted_n_peaks, extracted_n_components, extracted_n_channels = projected_waveforms.shape
    n_peaks = spikes.shape[0]
    assert extracted_n_peaks == n_peaks
    assert extracted_n_components == temporal_pca.pca_model[0].n_components
    assert extracted_n_channels == np.max(extract_waveforms.neighbours_mask.sum(axis=1))


def test_initialization_with_wrong_parents_failure(generated_recording, model_path_of_trained_pca):
    recording, _ = generated_recording
    model_folder_path = model_path_of_trained_pca
    dummy_parent = PipelineNode(recording=recording)
    extract_waveforms = ExtractSparseWaveforms(
        recording=recording, ms_before=1, ms_after=1, radius_um=40, return_output=True
    )

    match_error = f"TemporalPCA should have a single {WaveformsNode.__name__} in its parents"

    # Parents without waveform extraction
    with pytest.raises(TypeError, match=match_error):
        TemporalPCAProjection(recording=recording, model_folder_path=model_folder_path, parents=[dummy_parent])

    # Empty parents
    with pytest.raises(TypeError, match=match_error):
        TemporalPCAProjection(recording=recording, model_folder_path=model_folder_path, parents=None)

    # Multiple parents
    ### This test is deactivate waiting for the find_parents methods
    # with pytest.raises(TypeError, match=match_error):
    #    TemporalPCAProjection(
    #        recording=recording, model_folder_path=model_folder_path, parents=[extract_waveforms, extract_waveforms]
    #    )


def test_pca_waveform_extract_and_model_mismatch(generated_recording, model_path_of_trained_pca):
    recording, _ = generated_recording
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


def test_pca_incorrect_model_path(generated_recording, model_path_of_trained_pca):
    recording, _ = generated_recording
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
