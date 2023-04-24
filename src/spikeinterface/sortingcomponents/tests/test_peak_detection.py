import numpy as np
import shutil
from pathlib import Path
import pytest

from spikeinterface import download_dataset
from spikeinterface.extractors.neoextractors.mearec import MEArecRecordingExtractor, MEArecSortingExtractor

from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from spikeinterface.sortingcomponents.peak_pipeline import ExtractDenseWaveforms
from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass
from spikeinterface.sortingcomponents.features_from_peaks import PeakToPeakFeature

from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCADenoising
from spikeinterface.sortingcomponents.peak_detection import IterativePeakDetector, DetectPeakLocallyExclusive
from spikeinterface.sortingcomponents.peak_pipeline import run_node_pipeline

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sortingcomponents"
else:
    cache_folder = Path("cache_folder") / "sortingcomponents"

try:
    import pyopencl

    HAVE_PYOPENCL = True
except:
    HAVE_PYOPENCL = False

try:
    import torch

    HAVE_TORCH = True
except:
    HAVE_TORCH = False


@pytest.fixture(scope="module")
def recording():
    repo = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data"
    remote_path = "mearec/mearec_test_10s.h5"
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    return recording

@pytest.fixture(scope="module")
def sorting():
    repo = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data"
    remote_path = "mearec/mearec_test_10s.h5"
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    sorting = MEArecSortingExtractor(local_path)
    return sorting

@pytest.fixture(scope="module")
def spike_trains(sorting):
    spike_trains = sorting.get_all_spike_trains()[0][0]
    return spike_trains

@pytest.fixture(scope="module")
def job_kwargs():
    return dict(n_jobs=-1, chunk_size=10000, progress_bar=True, verbose=True)

# Don't know why this was different normal job_kwargs
@pytest.fixture(scope="module")
def torch_job_kwargs(job_kwargs):
    torch_job_kwargs = job_kwargs.copy()
    torch_job_kwargs["n_jobs"] = 2
    return torch_job_kwargs


def test_iterative_peak_detection(recording, job_kwargs, tmp_path):
    # Train a PCA denoised model as a temporary solution
    ms_before = 1.0
    ms_after = 1.0
    extract_dense_waveforms = ExtractDenseWaveforms(recording=recording, ms_before=ms_before, ms_after=ms_after)

    model_folder_path = tmp_path / "temporal_pca_model"
    model_folder_path.mkdir()
    # model_folder_path.mkdir(parents=True, exist_ok=True)
    # Fit the model
    n_components = 3
    n_peaks = 100  # Heuristic for extracting around 1k waveforms per channel
    peak_selection_params = dict(method="uniform", select_per_channel=True, n_peaks=n_peaks)
    detect_peaks_params = dict(method="by_channel", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1)
    TemporalPCADenoising.fit(
        recording=recording,
        model_folder_path=model_folder_path,
        n_components=n_components,
        ms_before=ms_before,
        ms_after=ms_after,
        detect_peaks_params=detect_peaks_params,
        peak_selection_params=peak_selection_params,
        job_kwargs=job_kwargs,
    )

    waveform_denoising = TemporalPCADenoising(
        recording=recording, parents=[extract_dense_waveforms], model_folder_path=model_folder_path
    )

    # Peak detection parameters
    exclude_sweep_ms = 1.0
    peak_sign = "both"
    detect_threshold = 5
    local_radius_um = 50
    detect_threshold = 5

    detect_peaks_instance = DetectPeakLocallyExclusive(
        recording=recording,
        detect_threshold=detect_threshold,
        peak_sign=peak_sign,
        local_radius_um=local_radius_um,
        exclude_sweep_ms=exclude_sweep_ms,
    )
    num_iterations = 2
    iterative_peak_detector = IterativePeakDetector(
        recording=recording,
        peak_detector_node=detect_peaks_instance,
        waveform_extraction_node=extract_dense_waveforms,
        waveform_denoising_node=waveform_denoising,
        num_iterations=num_iterations,
        return_output=(True, True)
    )

    peaks, waveforms = run_node_pipeline(recording=recording, nodes=[iterative_peak_detector], job_kwargs=job_kwargs)
    # Assert there is a field call iteration in structured array peaks
    assert "iteration" in peaks.dtype.names
    assert peaks.shape[0] == waveforms.shape[0]
        
    num_peaks_in_first_iteration = peaks[peaks["iteration"] == 0].size
    num_peaks_in_second_iteration = peaks[peaks["iteration"] == 1].size
    num_total_peaks = peaks.size
    assert num_total_peaks == (num_peaks_in_first_iteration + num_peaks_in_second_iteration)

def test_detect_peaks_by_channel(recording, spike_trains, job_kwargs, torch_job_kwargs):
    peaks_by_channel_np = detect_peaks(
        recording, method="by_channel", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1, **job_kwargs
    )
            
    if HAVE_TORCH:
        peaks_by_channel_torch = detect_peaks(
            recording,
            method="by_channel_torch",
            peak_sign="neg",
            detect_threshold=5,
            exclude_sweep_ms=0.1,
            **torch_job_kwargs,
        )

        # Test that torch and numpy implementation match
        assert np.isclose(np.array(len(peaks_by_channel_np)), np.array(len(peaks_by_channel_torch)), rtol=0.1)
        
def test_detect_peaks_locally_exclusive(recording, spike_trains, job_kwargs, torch_job_kwargs):
    peaks_by_channel_np = detect_peaks(
        recording, method="by_channel", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1, **job_kwargs
    )


    peaks_local_numba = detect_peaks(
        recording, method="locally_exclusive", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1, **job_kwargs
    )
    assert len(peaks_by_channel_np) > len(peaks_local_numba)

    if HAVE_TORCH:

        peaks_local_torch = detect_peaks(
            recording,
            method="locally_exclusive_torch",
            peak_sign="neg",
            detect_threshold=5,
            exclude_sweep_ms=0.1,
            **torch_job_kwargs,
        )
        assert np.isclose(np.array(len(peaks_local_numba)), np.array(len(peaks_local_torch)), rtol=0.1)
            
    if HAVE_PYOPENCL:
        peaks_local_cl = detect_peaks(
            recording,
            method="locally_exclusive_cl",
            peak_sign="neg",
            detect_threshold=5,
            exclude_sweep_ms=0.1,
            **job_kwargs,
        )
        assert len(peaks_local_numba) == len(peaks_local_cl)
        
    
def test_peak_detection_with_pipeline(recording, job_kwargs, torch_job_kwargs):
    
    extract_dense_waveforms = ExtractDenseWaveforms(recording, ms_before=1.0, ms_after=1.0, return_output=False)

    pipeline_nodes = [
        extract_dense_waveforms,
        PeakToPeakFeature(recording,  all_channels=False, parents=[extract_dense_waveforms]),
        LocalizeCenterOfMass(recording, local_radius_um=50., parents=[extract_dense_waveforms]),
    ]
    peaks, ptp, peak_locations = detect_peaks(recording, method='locally_exclusive',
                                              peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                                              pipeline_nodes=pipeline_nodes, **job_kwargs)
    assert peaks.shape[0] == ptp.shape[0]
    assert peaks.shape[0] == peak_locations.shape[0]
    assert 'x' in peak_locations.dtype.fields

    # same pipeline but saved to npy
    folder = cache_folder / 'peak_detection_folder'
    if folder.is_dir():
        shutil.rmtree(folder)
    peaks2, ptp2, peak_locations2 = detect_peaks(recording, method='locally_exclusive',
                                                 peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                                                 pipeline_nodes=pipeline_nodes, gather_mode='npy',
                                                 folder=folder, names=['peaks', 'ptps', 'peak_locations'],
                                                 **job_kwargs)
    peak_file = folder / 'peaks.npy'
    assert peak_file.is_file()
    peaks3 = np.load(peak_file)
    assert np.array_equal(peaks, peaks2)
    assert np.array_equal(peaks2, peaks3)

    ptp_file = folder / 'ptps.npy'
    assert ptp_file.is_file()
    ptp3 = np.load(ptp_file)
    assert np.array_equal(ptp, ptp2)
    assert np.array_equal(ptp2, ptp3)

    peak_location_file = folder / 'peak_locations.npy'
    assert peak_location_file.is_file()
    peak_locations3 = np.load(peak_location_file)
    assert np.array_equal(peak_locations, peak_locations2)
    assert np.array_equal(peak_locations2, peak_locations3)


    if HAVE_TORCH:
        peaks_torch, ptp_torch, peak_locations_torch = detect_peaks(recording, method='locally_exclusive_torch',
                                                                    peak_sign='neg', detect_threshold=5,
                                                                    exclude_sweep_ms=0.1, pipeline_nodes=pipeline_nodes,
                                                                    **torch_job_kwargs)
        assert peaks_torch.shape[0] == ptp_torch.shape[0]
        assert peaks_torch.shape[0] == peak_locations_torch.shape[0]
        assert 'x' in peak_locations_torch.dtype.fields

    if HAVE_PYOPENCL:
        peaks_cl, ptp_cl, peak_locations_cl = detect_peaks(recording, method='locally_exclusive_cl',
                                                           peak_sign='neg', detect_threshold=5,
                                                           exclude_sweep_ms=0.1, pipeline_nodes=pipeline_nodes,
                                                           **job_kwargs)
        assert peaks_cl.shape[0] == ptp_cl.shape[0]
        assert peaks_cl.shape[0] == peak_locations_cl.shape[0]
        assert 'x' in peak_locations_cl.dtype.fields
    

    # DEBUG
    DEBUG = False
    if DEBUG:
        import matplotlib.pyplot as plt
        import spikeinterface.widgets as sw
        from probeinterface.plotting import plot_probe

        sample_inds, chan_inds, amplitudes = peaks['sample_index'], peaks['channel_index'], peaks['amplitude']
        chan_offset = 500
        traces = recording.get_traces()
        traces += np.arange(traces.shape[1])[None, :] * chan_offset
        fig, ax = plt.subplots()
        ax.plot(traces, color='k')
        ax.scatter(sample_inds, chan_inds * chan_offset + amplitudes, color='r')
        plt.show()

        fig, ax = plt.subplots()
        probe = recording.get_probe()
        plot_probe(probe, ax=ax)
        ax.scatter(peak_locations['x'], peak_locations['y'], color='k', s=1, alpha=0.5)
        # MEArec is "yz" in 2D
        import MEArec
        recgen = MEArec.load_recordings(recordings=local_path, return_h5_objects=True,
        check_suffix=False,
        load=['recordings', 'spiketrains', 'channel_positions'],
        load_waveforms=False)
        soma_positions = np.zeros((len(recgen.spiketrains), 3), dtype='float32')
        for i, st in enumerate(recgen.spiketrains):
            soma_positions[i, :] = st.annotations['soma_position']
        ax.scatter(soma_positions[:, 1], soma_positions[:, 2], color='g', s=20, marker='*')
        plt.show()

if __name__ == '__main__':
    pass
    