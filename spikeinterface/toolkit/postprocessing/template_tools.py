import numpy as np


def  get_unit_extremum_channel(waveform_extractor, peak_sign='neg'):
    """
    Compute for each unit on which channel id the extremum is.
    
    """
    
    assert peak_sign in ('both', 'neg', 'pos')
    unit_ids = waveform_extractor.sorting.unit_ids
    channel_ids = waveform_extractor.recording.channel_ids
    
    before = waveform_extractor.nbefore
    
    extremum_channels_ids = {}
    
    for unit_id in unit_ids:
        template = waveform_extractor.get_template(unit_id)
        
        if peak_sign == 'both':
            peak_values = np.abs(template[before, :])
        elif peak_sign == 'neg':
            peak_values = -template[before, :]
        elif peak_sign == 'pos':
            peak_values = template[before, :]
        
        max_ind = np.argmax(peak_values)
        extremum_channels_ids[unit_id] = channel_ids[max_ind]
    
    return extremum_channels_ids



def get_unit_extremum_amplitude(waveform_extractor, peak_sign='neg'):
    """
    Computes the center of mass (COM) of a unit based on the template amplitudes.
    """
    
    unit_ids = waveform_extractor.sorting.unit_ids
    
    before = waveform_extractor.nbefore
    
    extremum_channels_ids = get_unit_extremum_channel(waveform_extractor, peak_sign=peak_sign)
    
    unit_amplitudes = {}
    for unit_id in unit_ids:
        template = waveform_extractor.get_template(unit_id)
        chan_id = extremum_channels_ids[unit_id]
        chan_ind = waveform_extractor.recording.id_to_index(chan_id)
        unit_amplitudes[unit_id] = template[before, chan_ind]
    
    
    return unit_amplitudes


def compute_unit_centers_of_mass(waveform_extractor):
    raise NotImplementedError
    
    
    