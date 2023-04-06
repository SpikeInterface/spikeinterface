import pytest
import numpy as np


from spikeinterface.sortingcomponents.waveforms.waveform_thresholder import WaveformThresholder
from spikeinterface.sortingcomponents.peak_pipeline import (
    ExtractDenseWaveforms,
    ExtractSparseWaveforms,
    WaveformExtractorNode,
    PipelineNode,
    run_peak_pipeline,
)


def test_waveform_thresholder(mearec_recording, detected_peaks, chunk_executor_kwargs):
    recording = mearec_recording
    peaks = detected_peaks

    # Parameters
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, ms_before=ms_before, ms_after=ms_after, return_output=True
    )

    tresholded_waveforms_ptp = WaveformThresholder(recording=recording, parents=[extract_waveforms], feature='ptp', threshold=3, return_output=True)
    tresholded_waveforms_mean = WaveformThresholder(recording=recording, parents=[extract_waveforms], feature='mean', threshold=3, return_output=True)
    tresholded_waveforms_energy = WaveformThresholder(recording=recording, parents=[extract_waveforms], feature='energy', threshold=3, return_output=True)
    tresholded_waveforms_peak  = WaveformThresholder(recording=recording, parents=[extract_waveforms], feature='v_peak', threshold=-5, return_output=True)
    
    pipeline_nodes = [extract_waveforms, tresholded_waveforms_ptp]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )
    assert np.all(tresholded_waveforms.ptp(axis=1) >= 3)

    pipeline_nodes = [extract_waveforms, tresholded_waveforms_mean]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )
    assert np.all(tresholded_waveforms.mean(axis=1) >= 0)

    pipeline_nodes = [extract_waveforms, tresholded_waveforms_energy]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )
    assert np.all(np.linalg.norm(tresholded_waveforms, axis=1) >= 3)

    pipeline_nodes = [extract_waveforms, tresholded_waveforms_peak]
    # Extract projected waveforms and compare
    waveforms, tresholded_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )

    assert np.all(tresholded_waveforms[:, extract_waveforms.nbefore, :] >= -5)