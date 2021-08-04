from pathlib import Path
import shutil
import numpy as np

from spikeinterface.core.tests.testing_tools import generate_recording, generate_sorting
from spikeinterface import WaveformExtractor, extract_waveforms


def _clean_all():
    folders = ["wf_rec1", "wf_rec2", "wf_sort2", "test_waveform_extractor",
               "test_extract_waveforms_1job", "test_extract_waveforms_2job"]
    for folder in folders:
        if Path(folder).exists():
            shutil.rmtree(folder)


def setup_module():
    _clean_all()


def teardown_module():
    _clean_all()


def test_WaveformExtractor():
    durations = [30, 40]
    sampling_frequency = 30000.

    # 2 segments
    recording = generate_recording(num_channels=2, durations=durations, sampling_frequency=sampling_frequency)
    recording.annotate(is_filtered=True)
    folder_rec = "wf_rec1"
    recording = recording.save(folder=folder_rec)
    sorting = generate_sorting(num_units=5, sampling_frequency=sampling_frequency, durations=durations)

    # test with dump !!!!
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

    recording = generate_recording(num_channels=2, durations=durations, sampling_frequency=sampling_frequency)
    recording.annotate(is_filtered=True)
    folder_rec = "wf_rec2"
    recording = recording.save(folder=folder_rec)
    sorting = generate_sorting(num_units=5, sampling_frequency=sampling_frequency, durations=durations)
    folder_sort = "wf_sort2"
    sorting = sorting.save(folder=folder_sort)
    # test without dump !!!!
    #  recording = recording.save()
    #  sorting = sorting.save()

    folder1 = Path('test_extract_waveforms_1job')
    if folder1.is_dir():
        shutil.rmtree(folder1)
    we1 = extract_waveforms(recording, sorting, folder1, max_spikes_per_unit=None, return_scaled=False)

    folder2 = Path('test_extract_waveforms_2job')
    if folder2.is_dir():
        shutil.rmtree(folder2)
    we2 = extract_waveforms(recording, sorting, folder2, n_jobs=2, total_memory="10M", max_spikes_per_unit=None,
                            return_scaled=False)

    folder3 = Path('test_extract_waveforms_returnscaled')
    if folder3.is_dir():
        shutil.rmtree(folder3)

    # set scaling values to recording
    gain = 0.1
    recording.set_channel_gains(gain)
    recording.set_channel_offsets(0)
    we3 = extract_waveforms(recording, sorting, folder3, n_jobs=2, total_memory="10M", max_spikes_per_unit=None,
                            return_scaled=True)

    wf1 = we1.get_waveforms(0)
    wf2 = we2.get_waveforms(0)
    assert np.array_equal(wf1, wf2)

    wf3 = we3.get_waveforms(0)
    assert np.array_equal((wf1).astype("float32") * gain, wf3)


if __name__ == '__main__':
    test_WaveformExtractor()
    test_extract_waveforms()
