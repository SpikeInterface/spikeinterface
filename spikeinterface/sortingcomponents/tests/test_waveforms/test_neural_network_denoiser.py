import numpy as np
import pytest

from spikeinterface.extractors import MEArecRecordingExtractor
from spikeinterface import download_dataset


from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline, ExtractDenseWaveforms
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.waveforms.neural_network_denoiser import SingleChannelToyDenoiser



def test_single_channel_toy_denoiser_in_peak_pipeline():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)


    job_kwargs = dict(chunk_duration='0.5s', n_jobs=2, progress_bar=False)

    peaks = detect_peaks(recording, method='locally_exclusive',
                            peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                            **job_kwargs)

    ms_before = 2.0
    ms_after = 2.0
    sampling_frequency = recording.get_sampling_frequency()
    waveform_extraction = ExtractDenseWaveforms(recording, ms_before=ms_before, ms_after=ms_after, return_output=True)

    # # Test post_check mechanism for SingleChannelToyDenoiser
    # with pytest.raises(ValueError) as e_info:
    #     toy_denoiser = SingleChannelToyDenoiser(recording, parents=[waveform_extraction])
        
    
    # Build nodes for computation
    waveform_extraction = ExtractDenseWaveforms(recording, ms_before=ms_before, ms_after=ms_after, return_output=True)
    toy_denoiser = SingleChannelToyDenoiser(recording, parents=[waveform_extraction])


    nodes = [waveform_extraction, toy_denoiser]
    waveforms, denoised_waveforms = run_peak_pipeline(recording, peaks=peaks, nodes=nodes, job_kwargs=job_kwargs)

    assert waveforms.shape == denoised_waveforms.shape
    assert 3==6 