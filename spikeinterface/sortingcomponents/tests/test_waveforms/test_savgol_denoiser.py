import pytest


from spikeinterface.sortingcomponents.waveforms.savgol_denoiser import SavGolDenoiser
from spikeinterface.sortingcomponents.peak_pipeline import (
    ExtractDenseWaveforms,
    ExtractSparseWaveforms,
    WaveformExtractorNode,
    PipelineNode,
    run_peak_pipeline,
)



def test_savgol_denoising(mearec_recording, detected_peaks, chunk_executor_kwargs):
    recording = mearec_recording
    peaks = detected_peaks

    # Parameters
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, ms_before=ms_before, ms_after=ms_after, return_output=True
    )

    spline_denoiser = SavGolDenoiser(
        recording=recording, parents=[extract_waveforms]
    )
    pipeline_nodes = [extract_waveforms, spline_denoiser]

    # Extract projected waveforms and compare
    waveforms, denoised_waveforms = run_peak_pipeline(
        recording, peaks, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs
    )
    assert waveforms.shape == denoised_waveforms.shape


