import pytest


from spikeinterface.sortingcomponents.waveforms.savgol_denoiser import SavGolDenoiser

from spikeinterface.core.node_pipeline import (
    PeakRetriever,
    ExtractDenseWaveforms,
    run_node_pipeline,
)


def test_savgol_denoising(mearec_recording, detected_peaks, chunk_executor_kwargs):
    recording = mearec_recording
    peaks = detected_peaks

    # Parameters
    ms_before = 1.0
    ms_after = 1.0

    # Node initialization
    peak_retriever = PeakRetriever(recording, peaks)

    extract_waveforms = ExtractDenseWaveforms(
        recording=recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=True
    )

    spline_denoiser = SavGolDenoiser(recording=recording, parents=[peak_retriever, extract_waveforms])
    pipeline_nodes = [peak_retriever, extract_waveforms, spline_denoiser]

    # Extract projected waveforms and compare
    waveforms, denoised_waveforms = run_node_pipeline(recording, nodes=pipeline_nodes, job_kwargs=chunk_executor_kwargs)
    assert waveforms.shape == denoised_waveforms.shape
