import numpy as np

from spikeinterface.extractors import MEArecRecordingExtractor
from spikeinterface import download_dataset


from spikeinterface.sortingcomponents.peak_pipeline import WaveformStep, run_peak_pipeline
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.waveforms.normalization import MaxValueNormalization



def test_single_waveform_transformer_in_peak_pipeline():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)


    job_kwargs = dict(chunk_duration='0.5s', n_jobs=1, progress_bar=False)

    peaks = detect_peaks(recording, method='locally_exclusive',
                            peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                            **job_kwargs)

    waveform_extracting_step = WaveformStep(recording, ms_before=1., ms_after=1.,  peak_sign='neg', all_channels=True)
    # Add the waveform normalization to the step:
    max_value_normalization = MaxValueNormalization()
    waveform_extracting_step.waveform_transformer_pipe = [max_value_normalization] 

    steps = [waveform_extracting_step]
    waveforms = run_peak_pipeline(recording, peaks=peaks, steps=steps, job_kwargs=job_kwargs)

    waveforms_are_normalized = np.all(np.isclose(np.abs(waveforms).max(axis=1), 1)) 
    assert waveforms_are_normalized
    

def test_single_waveform_transformer_in_peak_pipeline_parallel():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)


    job_kwargs = dict(chunk_duration='0.5s', n_jobs=2, progress_bar=False)

    peaks = detect_peaks(recording, method='locally_exclusive',
                            peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                            **job_kwargs)

    waveform_extracting_step = WaveformStep(recording, ms_before=1., ms_after=1.,  peak_sign='neg', all_channels=True)
    # Add the waveform normalization to the step:
    max_value_normalization = MaxValueNormalization()
    waveform_extracting_step.waveform_transformer_pipe = [max_value_normalization] 

    steps = [waveform_extracting_step]
    waveforms = run_peak_pipeline(recording, peaks=peaks, steps=steps, job_kwargs=job_kwargs)
    waveforms_are_normalized = np.all(np.isclose(np.abs(waveforms).max(axis=1), 1)) 
    assert waveforms_are_normalized