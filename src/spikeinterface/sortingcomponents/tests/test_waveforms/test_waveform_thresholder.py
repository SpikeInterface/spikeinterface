import pytest
import numpy as np
import operator


from spikeinterface.sortingcomponents.waveforms.waveform_thresholder import WaveformThresholder
from spikeinterface.sortingcomponents.peak_pipeline import (
    ExtractDenseWaveforms,
    ExtractSparseWaveforms,
    WaveformsNode,
    PipelineNode,
    run_peak_pipeline,
)


@pytest.fixture(scope="module")
def extract_waveforms(mearec_recording):
    # Parameters
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    return ExtractDenseWaveforms(recording=mearec_recording, ms_before=ms_before, ms_after=ms_after, return_output=True)


def test_waveform_thresholder_ptp(extract_waveforms, mearec_recording, detected_peaks, chunk_executor_kwargs):
    recording = mearec_recording
    peaks = detected_peaks

    tresholded_waveforms_ptp = WaveformThresholder(
        recording=recording, parents=[extract_waveforms], feature="ptp", threshold=3, return_output=True
    )
    noise_levels = tresholded_waveforms_ptp.noise_levels

    pipeline_nodes = [extract_waveforms, tresholded_waveforms_ptp]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )

    data = tresholded_waveforms.ptp(axis=1) / noise_levels
    assert np.all(data[data != 0] > 3)


def test_waveform_thresholder_mean(extract_waveforms, mearec_recording, detected_peaks, chunk_executor_kwargs):
    recording = mearec_recording
    peaks = detected_peaks

    tresholded_waveforms_mean = WaveformThresholder(
        recording=recording, parents=[extract_waveforms], feature="mean", threshold=0, return_output=True
    )

    pipeline_nodes = [extract_waveforms, tresholded_waveforms_mean]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )

    assert np.all(tresholded_waveforms.mean(axis=1) >= 0)


def test_waveform_thresholder_energy(extract_waveforms, mearec_recording, detected_peaks, chunk_executor_kwargs):
    recording = mearec_recording
    peaks = detected_peaks

    tresholded_waveforms_energy = WaveformThresholder(
        recording=recording, parents=[extract_waveforms], feature="energy", threshold=3, return_output=True
    )
    noise_levels = tresholded_waveforms_energy.noise_levels

    pipeline_nodes = [extract_waveforms, tresholded_waveforms_energy]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )

    data = np.linalg.norm(tresholded_waveforms, axis=1) / noise_levels
    assert np.all(data[data != 0] > 3)


def test_waveform_thresholder_operator(extract_waveforms, mearec_recording, detected_peaks, chunk_executor_kwargs):
    recording = mearec_recording
    peaks = detected_peaks

    import operator

    tresholded_waveforms_peak = WaveformThresholder(
        recording=recording,
        parents=[extract_waveforms],
        feature="peak_voltage",
        threshold=5,
        operator=operator.ge,
        return_output=True,
    )
    noise_levels = tresholded_waveforms_peak.noise_levels

    pipeline_nodes = [extract_waveforms, tresholded_waveforms_peak]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )

    data = tresholded_waveforms[:, extract_waveforms.nbefore, :] / noise_levels
    assert np.all(data[data != 0] <= 5)
