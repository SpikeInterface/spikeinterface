import numpy as np


def get_template_amplitudes(waveform_extractor, peak_sign='neg'):
    """
    Get amplitude per channel for each units
    """
    assert peak_sign in ('both', 'neg', 'pos')
    unit_ids = waveform_extractor.sorting.unit_ids
    channel_ids = waveform_extractor.recording.channel_ids
    
    before = waveform_extractor.nbefore
    
    peak_values = {}
    
    for unit_id in unit_ids:
        template = waveform_extractor.get_template(unit_id)
        
        if peak_sign == 'both':
            values = np.abs(template[before, :])
        elif peak_sign == 'neg':
            values = -template[before, :]
        elif peak_sign == 'pos':
            values = template[before, :]
        peak_values[unit_id] = values

    return peak_values

    
def  get_template_extremum_channel(waveform_extractor, peak_sign='neg'):
    """
    Compute for each unit on which channel id the extremum is.
    
    """
    unit_ids = waveform_extractor.sorting.unit_ids
    channel_ids = waveform_extractor.recording.channel_ids

    peak_values = get_template_amplitudes(waveform_extractor, peak_sign=peak_sign)
    extremum_channels_ids = {}
    for unit_id in unit_ids:
        max_ind = np.argmax(peak_values[unit_id])
        extremum_channels_ids[unit_id] = channel_ids[max_ind]
    
    return extremum_channels_ids

def get_template_best_channels(waveform_extractor, num_channels, peak_sign='neg'):
    """
    Get channels with hiher amplitudes for each unit.
    """
    unit_ids = waveform_extractor.sorting.unit_ids
    channel_ids = waveform_extractor.recording.channel_ids

    peak_values = get_template_amplitudes(waveform_extractor, peak_sign=peak_sign)
    best_channels_ids = {}
    for unit_id in unit_ids:
        chan_inds = np.argsort(np.abs(peak_values[unit_id]))[::-1]
        chan_inds = chan_inds[:num_channels]
        best_channels_ids[unit_id] = channel_ids[chan_inds]
    
    return best_channels_ids


def get_template_extremum_amplitude(waveform_extractor, peak_sign='neg'):
    """
    Computes amplitudes on the best channel.
    """
    
    unit_ids = waveform_extractor.sorting.unit_ids
    
    before = waveform_extractor.nbefore
    
    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign)
    
    unit_amplitudes = {}
    for unit_id in unit_ids:
        template = waveform_extractor.get_template(unit_id)
        chan_id = extremum_channels_ids[unit_id]
        chan_ind = waveform_extractor.recording.id_to_index(chan_id)
        unit_amplitudes[unit_id] = template[before, chan_ind]
    
    
    return unit_amplitudes


def compute_unit_centers_of_mass(waveform_extractor, peak_sign='neg', num_channels=10):
    '''
    Computes the center of mass (COM) of a unit based on the template amplitudes.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor

    Returns
    -------
    centers_of_mass: dict of np.array
    '''
    unit_ids = waveform_extractor.sorting.unit_ids
    
    recording = waveform_extractor.recording
    if num_channels is None:
        num_channels = recording.get_num_channels()
    locations = recording.get_channel_locations()

    best_channel_ids = get_template_best_channels(waveform_extractor, num_channels, peak_sign=peak_sign)
    amplitudes = get_template_amplitudes(waveform_extractor, peak_sign=peak_sign)
    

    coms = []
    for unit_id in unit_ids:
        template = waveform_extractor.get_template(unit_id)
        
        chan_ids = best_channel_ids[unit_id]
        chan_inds = recording.ids_to_indices(chan_ids)
        
        amps = amplitudes[unit_id][chan_inds]
        amps = np.abs(amps)
        
        com = np.sum(amps[:, np.newaxis] * locations[chan_inds, :], axis=0) / np.sum(amps)
        coms.append(com)
    
    coms = dict(zip(unit_ids, coms))

    return coms
