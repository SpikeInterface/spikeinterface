import pytest


from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection, TemporalPCADenoising
from spikeinterface.core.node_pipeline import (
    PeakRetriever,
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
def model_path_of_trained_pca(folder_to_save_pca_model, mearec_recording, chunk_executor_kwargs):
    """
    Trains a pca model and makes its folder available to all the tests in this module.
    """
    recording = mearec_recording

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
    TemporalPCAProjection.fit(
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


def test_pca_denoising(mearec_recording, detected_peaks, model_path_of_trained_pca, chunk_executor_kwargs):
    recording = mearec_recording
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


def test_pca_denoising_sparse(mearec_recording, detected_peaks, model_path_of_trained_pca, chunk_executor_kwargs):
    recording = mearec_recording
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


def test_pca_projection(mearec_recording, detected_peaks, model_path_of_trained_pca, chunk_executor_kwargs):
    recording = mearec_recording
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


def test_pca_projection_sparsity(mearec_recording, detected_peaks, model_path_of_trained_pca, chunk_executor_kwargs):
    recording = mearec_recording
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


def test_initialization_with_wrong_parents_failure(mearec_recording, model_path_of_trained_pca):
    recording = mearec_recording
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
