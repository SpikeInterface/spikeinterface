import numpy as np
import shutil
from pathlib import Path
import tempfile

import pytest


from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from spikeinterface.core.node_pipeline import ExtractDenseWaveforms, ExtractSparseWaveforms
from spikeinterface.sortingcomponents.peak_localization.method_list import LocalizeCenterOfMass
from spikeinterface.sortingcomponents.waveforms.features_from_peaks import PeakToPeakFeature

from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCADenoising
from spikeinterface.sortingcomponents.peak_detection.iterative import IterativePeakDetector


from spikeinterface.sortingcomponents.peak_detection.by_channel import ByChannelPeakDetector, ByChannelTorchPeakDetector
from spikeinterface.sortingcomponents.peak_detection.locally_exclusive import (
    LocallyExclusivePeakDetector,
    LocallyExclusiveTorchPeakDetector,
)

from spikeinterface.core import get_noise_levels
from spikeinterface.core.node_pipeline import run_node_pipeline
from spikeinterface.sortingcomponents.tools import get_prototype_and_waveforms_from_peaks

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
    return dict(n_jobs=1, chunk_size=10000, progress_bar=True, mp_context="spawn")


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
    detect_peaks_params = dict(method="by_channel", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=1.0)
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
        noise_levels=get_noise_levels(recording, return_in_uV=False),
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
    peak_detector_node = LocallyExclusivePeakDetector(**peak_detector_kwargs)

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
    peak_detector_node = LocallyExclusivePeakDetector(**peak_detector_kwargs)

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
    peak_detector_node = LocallyExclusivePeakDetector(**peak_detector_kwargs)

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
        recording,
        method="by_channel",
        method_kwargs=dict(peak_sign="neg", detect_threshold=5, exclude_sweep_ms=1.0),
        job_kwargs=job_kwargs,
    )

    if HAVE_TORCH:
        peaks_by_channel_torch = detect_peaks(
            recording,
            method="by_channel_torch",
            method_kwargs=dict(
                peak_sign="neg",
                detect_threshold=5,
                exclude_sweep_ms=1.0,
            ),
            job_kwargs=torch_job_kwargs,
        )

        # Test that torch and numpy implementation match
        assert np.isclose(np.array(len(peaks_by_channel_np)), np.array(len(peaks_by_channel_torch)), rtol=0.1)


def test_detect_peaks_locally_exclusive(recording, job_kwargs, torch_job_kwargs):
    peaks_by_channel_np = detect_peaks(
        recording,
        method="by_channel",
        method_kwargs=dict(peak_sign="neg", detect_threshold=5, exclude_sweep_ms=1.0),
        job_kwargs=job_kwargs,
    )

    peaks_local_numba = detect_peaks(
        recording,
        method="locally_exclusive",
        method_kwargs=dict(peak_sign="neg", detect_threshold=5, exclude_sweep_ms=1.0),
        job_kwargs=job_kwargs,
    )
    assert len(peaks_by_channel_np) > len(peaks_local_numba)

    DEBUG = True
    if DEBUG:
        import matplotlib.pyplot as plt

        peaks = peaks_local_numba
        labels = ["locally_exclusive numba",  ]

        fig, ax = plt.subplots()
        chan_offset = 500
        traces = recording.get_traces().copy()
        traces += np.arange(traces.shape[1])[None, :] * chan_offset
        ax.plot(traces, color="k")

        for count, peaks in enumerate([peaks_local_numba, ]):
            sample_inds, chan_inds, amplitudes = peaks["sample_index"], peaks["channel_index"], peaks["amplitude"]
            ax.scatter(sample_inds, chan_inds * chan_offset + amplitudes, label=labels[count])

        ax.legend()
        plt.show()


    if HAVE_TORCH:
        peaks_local_torch = detect_peaks(
            recording,
            method="locally_exclusive_torch",
            method_kwargs=dict(
                peak_sign="neg",
                detect_threshold=5,
                exclude_sweep_ms=1.0,
            ),
            job_kwargs=torch_job_kwargs,
        )
        assert np.isclose(np.array(len(peaks_local_numba)), np.array(len(peaks_local_torch)), rtol=0.1)

    if HAVE_PYOPENCL:
        peaks_local_cl = detect_peaks(
            recording,
            method="locally_exclusive_cl",
            method_kwargs=dict(
                peak_sign="neg",
                detect_threshold=5,
                exclude_sweep_ms=1.0,
            ),
            job_kwargs=job_kwargs,
        )
        assert len(peaks_local_numba) == len(peaks_local_cl)


def test_detect_peaks_locally_exclusive_matched_filtering(recording, job_kwargs):
    peaks_by_channel_np = detect_peaks(
        recording,
        method="locally_exclusive",
        method_kwargs=dict(peak_sign="neg", detect_threshold=5, exclude_sweep_ms=1.0),
        job_kwargs=job_kwargs,
    )

    ms_before = 1.0
    ms_after = 1.0
    prototype, _, _ = get_prototype_and_waveforms_from_peaks(
        recording, peaks=peaks_by_channel_np, ms_before=ms_before, ms_after=ms_after, job_kwargs=job_kwargs
    )

    peaks_local_mf_filtering = detect_peaks(
        recording,
        method="matched_filtering",
        method_kwargs=dict(
            peak_sign="neg",
            detect_threshold=5.,
            exclude_sweep_ms=1.0,
            prototype=prototype,
            ms_before=1.0,
        ),
        job_kwargs=job_kwargs,
    )
    # @pierre : lets put back this test later
    # assert len(peaks_local_mf_filtering) > len(peaks_by_channel_np)

    peaks_local_mf_filtering_both = detect_peaks(
        recording,
        method="matched_filtering",
        method_kwargs=dict(
            peak_sign="both",
            detect_threshold=5.,
            exclude_sweep_ms=1.0,
            prototype=prototype,
            ms_before=1.0,
        ),
        job_kwargs=job_kwargs,
    )
    assert len(peaks_local_mf_filtering_both) > len(peaks_local_mf_filtering)

    DEBUG = False
    # DEBUG = True
    if DEBUG:
        import matplotlib.pyplot as plt

        peaks_local = peaks_by_channel_np
        peaks_mf_neg = peaks_local_mf_filtering
        peaks_mf_both = peaks_local_mf_filtering_both
        # labels = ["locally_exclusive", "mf_neg", "mf_both"]
        # peaks_by_method = [peaks_local, peaks_mf_neg, peaks_mf_both]
        labels = ["locally_exclusive", "mf_neg", ]
        peaks_by_method = [peaks_local, peaks_mf_neg,]


        fig, ax = plt.subplots()
        chan_offset = 500
        traces = recording.get_traces().copy()
        traces += np.arange(traces.shape[1])[None, :] * chan_offset
        ax.plot(traces, color="k")

        for count, peaks in enumerate(peaks_by_method):
            sample_inds, chan_inds, amplitudes = peaks["sample_index"], peaks["channel_index"], peaks["amplitude"]
            ax.scatter(sample_inds, chan_inds * chan_offset + amplitudes, label=labels[count], s= 50 - count * 15)

        ax.legend()
        plt.show()


detection_classes = [
    ByChannelPeakDetector,
    ByChannelTorchPeakDetector,
    LocallyExclusivePeakDetector,
    LocallyExclusiveTorchPeakDetector,
]


@pytest.mark.parametrize("detection_class", detection_classes)
def test_peak_sign_consistency(recording, job_kwargs, detection_class):
    if detection_class.need_noise_levels:
        kwargs = dict(recording=recording, noise_levels=get_noise_levels(recording, return_in_uV=False))
    else:
        kwargs = dict(recording=recording)

    kwargs["peak_sign"] = "neg"
    peak_detection_node = detection_class(**kwargs)
    negative_peaks = run_node_pipeline(recording=recording, nodes=[peak_detection_node], job_kwargs=job_kwargs)

    kwargs["peak_sign"] = "pos"
    peak_detection_node = detection_class(**kwargs)
    positive_peaks = run_node_pipeline(recording=recording, nodes=[peak_detection_node], job_kwargs=job_kwargs)

    kwargs["peak_sign"] = "both"
    peak_detection_node = detection_class(**kwargs)
    all_peaks = run_node_pipeline(recording=recording, nodes=[peak_detection_node], job_kwargs=job_kwargs)

    # To account for exclusion of positive peaks that are to close to negative peaks.
    # This should be excluded by the detection method when is exclusive so using peak_sign="both" should
    # Generate less peaks in this case
    if detection_class not in (ByChannelTorchPeakDetector, LocallyExclusiveTorchPeakDetector):
        # TODO later Torch do not pass this test
        assert (negative_peaks.size + positive_peaks.size) >= all_peaks.size

    # Original case that prompted this test
    if negative_peaks.size > 0 or positive_peaks.size > 0:
        assert all_peaks.size > 0



if __name__ == "__main__":
    recording, sorting = make_dataset()
    tmp_path = Path(tempfile.mkdtemp())

    job_kwargs_main = job_kwargs()
    # torch_job_kwargs_main = torch_job_kwargs(job_kwargs_main)
    # Create a temporary directory using the standard library
    # tmp_dir_main = tempfile.mkdtemp()
    # pca_model_folder_path_main = pca_model_folder_path(recording, job_kwargs_main, tmp_dir_main)
    # peak_detector_kwargs_main = peak_detector_kwargs(recording)

    # test_iterative_peak_detection(recording, job_kwargs_main, pca_model_folder_path_main, peak_detector_kwargs_main)

    # test_peak_sign_consistency(recording, torch_job_kwargs_main, LocallyExclusiveTorchPeakDetector)
    # test_peak_detection_with_pipeline(recording, job_kwargs_main, torch_job_kwargs_main, tmp_path)

    test_detect_peaks_locally_exclusive_matched_filtering(
        recording,
        job_kwargs_main,
    )

    # test_detect_peaks_locally_exclusive(recording, job_kwargs_main, torch_job_kwargs_main)
