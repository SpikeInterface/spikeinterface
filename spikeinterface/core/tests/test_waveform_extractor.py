from pathlib import Path
import shutil


from spikeinterface.core.tests.testing_tools import generate_recording, generate_sorting

from spikeinterface import WaveformExtractor, extract_waveforms

def test_WaveformExtractor():
    durations = [30, 40]
    sampling_frequency = 30000.
    
    # 2 segments
    recording = generate_recording(num_channels = 2, durations=durations, sampling_frequency=sampling_frequency)
    sorting = generate_sorting(num_units=5, sampling_frequency = sampling_frequency, durations=durations)

    recording = recording.save()
    sorting = sorting.save()
    
    
    folder = Path('test_waveform_extractor')
    if folder.is_dir():
        shutil.rmtree(folder)
    
    
    we = WaveformExtractor.create(recording, sorting, folder)
    
    we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)
    
    we.run(n_jobs=1, chunk_size=30000)
    
    we.run(n_jobs=4, chunk_size=30000, progress_bar=True)
    
    wfs = we.get_waveforms(0)
    assert wfs.shape[0] <= 500
    assert wfs.shape[1:] == (210, 2)
    
    wfs, sampled_index = we.get_waveforms(0, with_index=True)
    
    # load back
    we = WaveformExtractor.load_from_folder(folder)
    
    wfs = we.get_waveforms(0)
    
    template = we.get_template(0)
    assert template.shape == (210, 2)
    templates = we.get_all_templates()
    assert templates.shape == (5, 210, 2)
    


def test_extract_waveforms():
    # 2 segments
    durations = [30, 40]
    sampling_frequency = 30000.
    
    recording = generate_recording(num_channels = 2, durations=durations, sampling_frequency=sampling_frequency)
    sorting =generate_sorting(num_units=5, sampling_frequency = sampling_frequency, durations=durations)
    recording = recording.save()
    sorting = sorting.save()

    folder = Path('test_extract_waveforms')
    if folder.is_dir():
        shutil.rmtree(folder)
    
    we = extract_waveforms(recording, sorting, folder)
    print(we)
    


if __name__ == '__main__':
    test_WaveformExtractor()
    test_extract_waveforms()

