import unittest
import shutil
from pathlib import Path

import pytest



from spikeinterface import WaveformExtractor
from spikeinterface.extractors import toy_example
from spikeinterface.toolkit.postprocessing import (get_template_amplitudes,
    get_template_extremum_channel, get_template_extremum_channel_peak_shift,
    get_template_best_channels, get_template_extremum_amplitude,
    compute_unit_centers_of_mass)


def setup_module():
    for folder in ('toy_rec', 'toy_sorting', 'toy_waveforms'):
        if Path(folder).is_dir():
            shutil.rmtree(folder)
    
    recording, sorting = toy_example(num_segments=2, num_units=10)
    recording = recording.save(folder='toy_rec')
    sorting = sorting.save(folder='toy_sorting')
    
    we = WaveformExtractor.create(recording, sorting, 'toy_waveforms')
    we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)
    we.run(n_jobs=1, chunk_size=30000)


def test_get_template_amplitudes():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    peak_values= get_template_amplitudes(we)
    print(peak_values)

def test_get_template_extremum_channel():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    extremum_channels_ids = get_template_extremum_channel(we, peak_sign='both')
    print(extremum_channels_ids)

def test_get_template_extremum_channel_peak_shift():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    shifts = get_template_extremum_channel_peak_shift(we, peak_sign='neg')
    print(shifts)
    
    # DEBUG
    # import matplotlib.pyplot as plt
    # extremum_channels_ids = get_template_extremum_channel(we, peak_sign='both')
    # for unit_id in we.sorting.unit_ids:
    #     chan_id = extremum_channels_ids[unit_id]
    #     chan_ind = we.recording.id_to_index(chan_id)
    #     template = we.get_template(unit_id)
    #     shift = shifts[unit_id]
    #     fig, ax = plt.subplots()
    #     template_chan = template[:, chan_ind]
    #     ax.plot(template_chan)
    #     ax.axvline(we.nbefore, color='grey')
    #     ax.axvline(we.nbefore + shift, color='red')
    #     plt.show()


def test_get_template_best_channels():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    best_channels = get_template_best_channels(we, num_channels=2)
    print(best_channels)
    
def test_get_template_extremum_amplitude():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    
    extremum_channels_ids = get_template_extremum_amplitude(we, peak_sign='both')
    print(extremum_channels_ids)
    
def test_compute_unit_centers_of_mass():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    
    coms = compute_unit_centers_of_mass(we, num_channels=4)
    print(coms)


if __name__ == '__main__':
    # setup_module()
    
    # test_get_template_amplitudes()
    # test_get_template_extremum_channel()
    test_get_template_extremum_channel_peak_shift()
    # test_get_template_best_channels()
    # test_get_template_extremum_amplitude()
    # test_compute_unit_centers_of_mass()
