import numpy as np
import shutil
from pathlib import Path
import tempfile

import pytest


from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from spikeinterface.core.node_pipeline import ExtractDenseWaveforms, ExtractSparseWaveforms
from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass
from spikeinterface.sortingcomponents.features_from_peaks import PeakToPeakFeature

from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCADenoising
from spikeinterface.sortingcomponents.peak_detection import IterativePeakDetector
from spikeinterface.sortingcomponents.peak_detection import (
    DetectPeakByChannel,
    DetectPeakByChannelTorch,
    DetectPeakLocallyExclusive,
    DetectPeakLocallyExclusiveTorch,
)

from spikeinterface.core.node_pipeline import run_node_pipeline
from spikeinterface.sortingcomponents.tools import get_prototype_spike

from spikeinterface.sortingcomponents.tests.common import make_dataset


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


@pytest.fixture(name="dataset", scope="module")
def dataset_fixture():
    return make_dataset()


@pytest.fixture(name="recording", scope="module")
def recording(dataset):
    recording, sorting = dataset
    return recording


@pytest.fixture(name="sorting", scope="module")
def sorting(dataset):
    recording, sorting = dataset
    return sorting


def job_kwargs():
    return dict(n_jobs=1, chunk_size=10000, progress_bar=True, verbose=True, mp_context="spawn")


@pytest.fixture(name="job_kwargs", scope="module")
def job_kwargs_fixture():
    return job_kwargs()


def torch_job_kwargs(job_kwargs):
    torch_job_kwargs = job_kwargs.copy()
    torch_job_kwargs["n_jobs"] = 2
    return torch_job_kwargs


@pytest.fixture(name="torch_job_kwargs", scope="module")
def torch_job_kwargs_fixture(job_kwargs):
    return torch_job_kwargs(job_kwargs)


def pca_model_folder_path(recording, job_kwargs, tmp_path):
    ms_before = 1.0
    ms_after = 1.0

    model_folder_path = Path(tmp_path) / "temporal_pca_model"
    model_folder_path.mkdir(parents=True, exist_ok=True)
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

    return model_folder_path


@pytest.fixture(name="pca_model_folder_path", scope="module")
def pca_model_folder_path_fixture(recording, job_kwargs, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("my_temp_dir")
    return pca_model_folder_path(recording, job_kwargs, tmp_path)


def peak_detector_kwargs(recording):
    peak_detector_keyword_arguments = dict(
        recording=recording,
        exclude_sweep_ms=1.0,
        peak_sign="both",
        detect_threshold=5,
        radius_um=50,
    )

    return peak_detector_keyword_arguments


@pytest.fixture(name="peak_detector_kwargs", scope="module")
def peak_detector_kwargs_fixture(recording):
    return peak_detector_kwargs(recording)


def test_iterative_peak_detection(recording, job_kwargs, pca_model_folder_path, peak_detector_kwargs):
    peak_detector_node = DetectPeakLocallyExclusive(**peak_detector_kwargs)

    ms_before = 1.0
    ms_after = 1.0
    waveform_extraction_node = ExtractDenseWaveforms(recording=recording, ms_before=ms_before, ms_after=ms_after)

    waveform_denoising_node = TemporalPCADenoising(
        recording=recording,
        parents=[waveform_extraction_node],
        model_folder_path=pca_model_folder_path,
    )

    num_iterations = 2
    iterative_peak_detector = IterativePeakDetector(
        recording=recording,
        peak_detector_node=peak_detector_node,
        waveform_extraction_node=waveform_extraction_node,
        waveform_denoising_node=waveform_denoising_node,
        num_iterations=num_iterations,
        return_output=(True, True),
    )

    peaks, waveforms = run_node_pipeline(recording=recording, nodes=[iterative_peak_detector], job_kwargs=job_kwargs)
    # Assert there is a field call iteration in structured array peaks
    assert "iteration" in peaks.dtype.names
    assert peaks.shape[0] == waveforms.shape[0]

    sample_indices = peaks["sample_index"]
    # Assert that sample_indices are ordered
    ordered_sample_indices = np.sort(sample_indices)
    assert np.array_equal(sample_indices, ordered_sample_indices)

    num_peaks_in_first_iteration = peaks[peaks["iteration"] == 0].size
    num_peaks_in_second_iteration = peaks[peaks["iteration"] == 1].size
    num_total_peaks = peaks.size
    assert num_total_peaks == (num_peaks_in_first_iteration + num_peaks_in_second_iteration)


def test_iterative_peak_detection_sparse(recording, job_kwargs, pca_model_folder_path, peak_detector_kwargs):
    peak_detector_node = DetectPeakLocallyExclusive(**peak_detector_kwargs)

    ms_before = 1.0
    ms_after = 1.0
    radius_um = 40
    waveform_extraction_node = ExtractSparseWaveforms(
        recording=recording,
        ms_before=ms_before,
        ms_after=ms_after,
        radius_um=radius_um,
    )

    waveform_denoising_node = TemporalPCADenoising(
        recording=recording,
        parents=[waveform_extraction_node],
        model_folder_path=pca_model_folder_path,
    )

    num_iterations = 2
    iterative_peak_detector = IterativePeakDetector(
        recording=recording,
        peak_detector_node=peak_detector_node,
        waveform_extraction_node=waveform_extraction_node,
        waveform_denoising_node=waveform_denoising_node,
        num_iterations=num_iterations,
        return_output=(True, True),
    )

    peaks, waveforms = run_node_pipeline(recording=recording, nodes=[iterative_peak_detector], job_kwargs=job_kwargs)
    # Assert there is a field call iteration in structured array peaks
    assert "iteration" in peaks.dtype.names
    assert peaks.shape[0] == waveforms.shape[0]

    # Assert that sample_indices are ordered
    sample_indices = peaks["sample_index"]
    ordered_sample_indices = np.sort(sample_indices)
    assert np.array_equal(sample_indices, ordered_sample_indices)

    num_peaks_in_first_iteration = peaks[peaks["iteration"] == 0].size
    num_peaks_in_second_iteration = peaks[peaks["iteration"] == 1].size
    num_total_peaks = peaks.size
    assert num_total_peaks == (num_peaks_in_first_iteration + num_peaks_in_second_iteration)


def test_iterative_peak_detection_thresholds(recording, job_kwargs, pca_model_folder_path, peak_detector_kwargs):
    peak_detector_node = DetectPeakLocallyExclusive(**peak_detector_kwargs)

    ms_before = 1.0
    ms_after = 1.0

    waveform_extraction_node = ExtractDenseWaveforms(recording=recording, ms_before=ms_before, ms_after=ms_after)

    waveform_denoising_node = TemporalPCADenoising(
        recording=recording,
        parents=[waveform_extraction_node],
        model_folder_path=pca_model_folder_path,
    )

    num_iterations = 3
    tresholds = [5.0, 3.0, 1.0]
    iterative_peak_detector = IterativePeakDetector(
        recording=recording,
        peak_detector_node=peak_detector_node,
        waveform_extraction_node=waveform_extraction_node,
        waveform_denoising_node=waveform_denoising_node,
        num_iterations=num_iterations,
        return_output=(True, True),
        tresholds=tresholds,
    )

    peaks, waveforms = run_node_pipeline(recording=recording, nodes=[iterative_peak_detector], job_kwargs=job_kwargs)
    # Assert there is a field call iteration in structured array peaks
    assert "iteration" in peaks.dtype.names
    assert peaks.shape[0] == waveforms.shape[0]

    # Assert that sample_indices are ordered
    sample_indices = peaks["sample_index"]
    ordered_sample_indices = np.sort(sample_indices)
    assert np.array_equal(sample_indices, ordered_sample_indices)

    num_peaks_in_first_iteration = peaks[peaks["iteration"] == 0].size
    num_peaks_in_second_iteration = peaks[peaks["iteration"] == 1].size
    num_peaks_in_third_iteration = peaks[peaks["iteration"] == 2].size
    num_total_peaks = peaks.size
    num_cumulative_peaks = num_peaks_in_first_iteration + num_peaks_in_second_iteration + num_peaks_in_third_iteration
    assert num_total_peaks == num_cumulative_peaks


def test_detect_peaks_by_channel(recording, job_kwargs, torch_job_kwargs):
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


def test_detect_peaks_locally_exclusive(recording, job_kwargs, torch_job_kwargs):
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


def test_detect_peaks_locally_exclusive_matched_filtering(recording, job_kwargs):
    peaks_by_channel_np = detect_peaks(
        recording, method="locally_exclusive", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1, **job_kwargs
    )

    ms_before = 1.0
    ms_after = 1.0
    prototype = get_prototype_spike(recording, peaks_by_channel_np, ms_before, ms_after, **job_kwargs)

    peaks_local_mf_filtering = detect_peaks(
        recording,
        method="matched_filtering",
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        prototype=prototype,
        **job_kwargs,
    )
    assert len(peaks_local_mf_filtering) > len(peaks_by_channel_np)

    DEBUG = False
    if DEBUG:
        import matplotlib.pyplot as plt

        peaks = peaks_local_mf_filtering

        sample_inds, chan_inds, amplitudes = peaks["sample_index"], peaks["channel_index"], peaks["amplitude"]
        chan_offset = 500
        traces = recording.get_traces().copy()
        traces += np.arange(traces.shape[1])[None, :] * chan_offset
        fig, ax = plt.subplots()
        ax.plot(traces, color="k")
        ax.scatter(sample_inds, chan_inds * chan_offset + amplitudes, color="r")
        plt.show()


detection_classes = [
    DetectPeakByChannel,
    DetectPeakByChannelTorch,
    DetectPeakLocallyExclusive,
    DetectPeakLocallyExclusiveTorch,
]


@pytest.mark.parametrize("detection_class", detection_classes)
def test_peak_sign_consistency(recording, job_kwargs, detection_class):
    peak_sign = "neg"
    peak_detection_node = detection_class(recording=recording, peak_sign=peak_sign)
    negative_peaks = run_node_pipeline(recording=recording, nodes=[peak_detection_node], job_kwargs=job_kwargs)

    peak_sign = "pos"
    peak_detection_node = detection_class(recording=recording, peak_sign=peak_sign)
    positive_peaks = run_node_pipeline(recording=recording, nodes=[peak_detection_node], job_kwargs=job_kwargs)

    peak_sign = "both"
    peak_detection_node = detection_class(recording=recording, peak_sign=peak_sign)
    all_peaks = run_node_pipeline(recording=recording, nodes=[peak_detection_node], job_kwargs=job_kwargs)

    # To account for exclusion of positive peaks that are to close to negative peaks.
    # This should be excluded by the detection method when is exclusive so using peak_sign="both" should
    # Generate less peaks in this case
    if detection_class not in (DetectPeakByChannelTorch, DetectPeakLocallyExclusiveTorch):
        # TODO later Torch do not pass this test
        assert (negative_peaks.size + positive_peaks.size) >= all_peaks.size

    # Original case that prompted this test
    if negative_peaks.size > 0 or positive_peaks.size > 0:
        assert all_peaks.size > 0


def test_peak_detection_with_pipeline(recording, job_kwargs, torch_job_kwargs, tmp_path):
    extract_dense_waveforms = ExtractDenseWaveforms(recording, ms_before=1.0, ms_after=1.0, return_output=False)

    pipeline_nodes = [
        extract_dense_waveforms,
        PeakToPeakFeature(recording, all_channels=False, parents=[extract_dense_waveforms]),
        LocalizeCenterOfMass(recording, radius_um=50.0, parents=[extract_dense_waveforms]),
    ]
    peaks, ptp, peak_locations = detect_peaks(
        recording,
        method="locally_exclusive",
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        pipeline_nodes=pipeline_nodes,
        **job_kwargs,
    )
    assert peaks.shape[0] == ptp.shape[0]
    assert peaks.shape[0] == peak_locations.shape[0]
    assert "x" in peak_locations.dtype.fields

    # same pipeline but saved to npy
    folder = tmp_path / "peak_detection_folder"
    if folder.is_dir():
        shutil.rmtree(folder)
    peaks2, ptp2, peak_locations2 = detect_peaks(
        recording,
        method="locally_exclusive",
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        pipeline_nodes=pipeline_nodes,
        gather_mode="npy",
        folder=folder,
        names=["peaks", "ptps", "peak_locations"],
        **job_kwargs,
    )
    peak_file = folder / "peaks.npy"
    assert peak_file.is_file()
    peaks3 = np.load(peak_file)
    assert np.array_equal(peaks, peaks2)
    assert np.array_equal(peaks2, peaks3)

    ptp_file = folder / "ptps.npy"
    assert ptp_file.is_file()
    ptp3 = np.load(ptp_file)
    assert np.array_equal(ptp, ptp2)
    assert np.array_equal(ptp2, ptp3)

    peak_location_file = folder / "peak_locations.npy"
    assert peak_location_file.is_file()
    peak_locations3 = np.load(peak_location_file)
    assert np.array_equal(peak_locations, peak_locations2)
    assert np.array_equal(peak_locations2, peak_locations3)

    if HAVE_TORCH:
        peaks_torch, ptp_torch, peak_locations_torch = detect_peaks(
            recording,
            method="locally_exclusive_torch",
            peak_sign="neg",
            detect_threshold=5,
            exclude_sweep_ms=0.1,
            pipeline_nodes=pipeline_nodes,
            **torch_job_kwargs,
        )
        assert peaks_torch.shape[0] == ptp_torch.shape[0]
        assert peaks_torch.shape[0] == peak_locations_torch.shape[0]
        assert "x" in peak_locations_torch.dtype.fields

    if HAVE_PYOPENCL:
        peaks_cl, ptp_cl, peak_locations_cl = detect_peaks(
            recording,
            method="locally_exclusive_cl",
            peak_sign="neg",
            detect_threshold=5,
            exclude_sweep_ms=0.1,
            pipeline_nodes=pipeline_nodes,
            **job_kwargs,
        )
        assert peaks_cl.shape[0] == ptp_cl.shape[0]
        assert peaks_cl.shape[0] == peak_locations_cl.shape[0]
        assert "x" in peak_locations_cl.dtype.fields

    # DEBUG
    DEBUG = False
    if DEBUG:
        import matplotlib.pyplot as plt
        import spikeinterface.widgets as sw
        from probeinterface.plotting import plot_probe

        sample_inds, chan_inds, amplitudes = peaks["sample_index"], peaks["channel_index"], peaks["amplitude"]
        chan_offset = 500
        traces = recording.get_traces()
        traces += np.arange(traces.shape[1])[None, :] * chan_offset
        fig, ax = plt.subplots()
        ax.plot(traces, color="k")
        ax.scatter(sample_inds, chan_inds * chan_offset + amplitudes, color="r")
        plt.show()

        fig, ax = plt.subplots()
        probe = recording.get_probe()
        plot_probe(probe, ax=ax)
        ax.scatter(peak_locations["x"], peak_locations["y"], color="k", s=1, alpha=0.5)
        # MEArec is "yz" in 2D
        # import MEArec

        # recgen = MEArec.load_recordings(
        #     recordings=local_path,
        #     return_h5_objects=True,
        #     check_suffix=False,
        #     load=["recordings", "spiketrains", "channel_positions"],
        #     load_waveforms=False,
        # )
        # soma_positions = np.zeros((len(recgen.spiketrains), 3), dtype="float32")
        # for i, st in enumerate(recgen.spiketrains):
        #     soma_positions[i, :] = st.annotations["soma_position"]
        # ax.scatter(soma_positions[:, 1], soma_positions[:, 2], color="g", s=20, marker="*")
        plt.show()


if __name__ == "__main__":
    recording, sorting = make_dataset()
    tmp_path = Path(tempfile.mkdtemp())

    job_kwargs_main = job_kwargs()
    # torch_job_kwargs_main = torch_job_kwargs(job_kwargs_main)
    # Create a temporary directory using the standard library
    # tmp_dir_main = tempfile.mkdtemp()
    # pca_model_folder_path_main = pca_model_folder_path(recording, job_kwargs_main, tmp_dir_main)
    # peak_detector_kwargs_main = peak_detector_kwargs(recording)

    # test_iterative_peak_detection(
    #     recording, job_kwargs_main, pca_model_folder_path_main, peak_detector_kwargs_main
    # )

    # test_peak_sign_consistency(recording, torch_job_kwargs_main, DetectPeakLocallyExclusiveTorch)
    # test_peak_detection_with_pipeline(recording, job_kwargs_main, torch_job_kwargs_main, tmp_path)

    test_detect_peaks_locally_exclusive_matched_filtering(
        recording,
        job_kwargs_main,
    )
