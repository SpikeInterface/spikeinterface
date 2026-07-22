from spikeinterface.core.node_pipeline import run_node_pipeline, PeakRetriever, ExtractDenseWaveforms
from spikeinterface.sortingcomponents.waveforms.denoising.neural_network_denoiser import SingleChannelDenoiser


def test_single_channel_toy_denoiser_in_peak_pipeline(generated_recording, detected_peaks, chunk_executor_kwargs):
    recording = generated_recording
    peaks = detected_peaks

    ms_before = 2.0
    ms_after = 2.0

    # Build nodes for computation
    peak_retriever = PeakRetriever(recording, peaks)
    waveform_extraction = ExtractDenseWaveforms(
        recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=True
    )
    toy_denoiser = SingleChannelDenoiser(
        recording,
        parents=[peak_retriever, waveform_extraction],
        repo_id="SpikeInterface/waveform_denoiser",
        model_name="toy_model_mearec",
    )

    nodes = [peak_retriever, waveform_extraction, toy_denoiser]
    waveforms, denoised_waveforms = run_node_pipeline(recording, nodes=nodes, job_kwargs=chunk_executor_kwargs)

    assert waveforms.shape == denoised_waveforms.shape


def test_single_channel_yass_denoiser(generated_recording_30khz, detected_peaks, chunk_executor_kwargs):
    recording = generated_recording_30khz
    peaks = detected_peaks

    nbefore = 42
    nafter = 79

    # Build nodes for computation
    peak_retriever = PeakRetriever(recording, peaks)
    waveform_extraction = ExtractDenseWaveforms(
        recording, parents=[peak_retriever], nbefore=nbefore, nafter=nafter, return_output=True
    )
    yass_denoiser = SingleChannelDenoiser(
        recording,
        parents=[peak_retriever, waveform_extraction],
        repo_id="spikeinterface/waveform_denoiser",
        model_name="yass_ibl",
    )

    nodes = [peak_retriever, waveform_extraction, yass_denoiser]
    waveforms, denoised_waveforms = run_node_pipeline(recording, nodes=nodes, job_kwargs=chunk_executor_kwargs)

    assert waveforms.shape == denoised_waveforms.shape
