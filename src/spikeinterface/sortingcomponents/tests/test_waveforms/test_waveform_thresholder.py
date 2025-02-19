import pytest
import numpy as np
import operator


from spikeinterface.sortingcomponents.waveforms.waveform_thresholder import WaveformThresholder
from spikeinterface.core.node_pipeline import ExtractDenseWaveforms
from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline


@pytest.fixture(scope="module")
def extract_dense_waveforms_node(generated_recording):
    # Parameters
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    return ExtractDenseWaveforms(
        recording=generated_recording, ms_before=ms_before, ms_after=ms_after, return_output=True
    )


def test_waveform_thresholder_ptp(
    extract_dense_waveforms_node, generated_recording, detected_peaks, chunk_executor_kwargs
):
    recording = generated_recording
    peaks = detected_peaks

    tresholded_waveforms_ptp = WaveformThresholder(
        recording=recording, parents=[extract_dense_waveforms_node], feature="ptp", threshold=3, return_output=True
    )
    noise_levels = tresholded_waveforms_ptp.noise_levels

    pipeline_nodes = [extract_dense_waveforms_node, tresholded_waveforms_ptp]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )

    data = np.ptp(tresholded_waveforms, axis=1) / noise_levels
    assert np.all(data[data != 0] > 3)


def test_waveform_thresholder_mean(
    extract_dense_waveforms_node, generated_recording, detected_peaks, chunk_executor_kwargs
):
    recording = generated_recording
    peaks = detected_peaks

    tresholded_waveforms_mean = WaveformThresholder(
        recording=recording, parents=[extract_dense_waveforms_node], feature="mean", threshold=0, return_output=True
    )

    pipeline_nodes = [extract_dense_waveforms_node, tresholded_waveforms_mean]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )

    assert np.all(tresholded_waveforms.mean(axis=1) >= 0)


def test_waveform_thresholder_energy(
    extract_dense_waveforms_node, generated_recording, detected_peaks, chunk_executor_kwargs
):
    recording = generated_recording
    peaks = detected_peaks

    tresholded_waveforms_energy = WaveformThresholder(
        recording=recording, parents=[extract_dense_waveforms_node], feature="energy", threshold=3, return_output=True
    )
    noise_levels = tresholded_waveforms_energy.noise_levels

    pipeline_nodes = [extract_dense_waveforms_node, tresholded_waveforms_energy]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )

    data = np.linalg.norm(tresholded_waveforms, axis=1) / noise_levels
    assert np.all(data[data != 0] > 3)


def test_waveform_thresholder_operator(
    extract_dense_waveforms_node, generated_recording, detected_peaks, chunk_executor_kwargs
):
    recording = generated_recording
    peaks = detected_peaks

    import operator

    tresholded_waveforms_peak = WaveformThresholder(
        recording=recording,
        parents=[extract_dense_waveforms_node],
        feature="peak_voltage",
        threshold=5,
        operator=operator.ge,
        return_output=True,
    )
    noise_levels = tresholded_waveforms_peak.noise_levels

    pipeline_nodes = [extract_dense_waveforms_node, tresholded_waveforms_peak]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )

    data = tresholded_waveforms[:, extract_dense_waveforms_node.nbefore, :] / noise_levels
    assert np.all(data[data != 0] <= 5)
