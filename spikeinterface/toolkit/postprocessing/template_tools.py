import numpy as np

from ..utils import get_channel_distances


def get_template_amplitudes(waveform_extractor, peak_sign='neg', mode='extremum'):
    """
    Get amplitude per channel for each units
    
    Parameters
    ----------
    mode: 
        'extremum':  max or min
        'at_index': take value at spike index

    """
    assert peak_sign in ('both', 'neg', 'pos')
    assert mode in ('extremum', 'at_index')
    unit_ids = waveform_extractor.sorting.unit_ids
    channel_ids = waveform_extractor.recording.channel_ids

    before = waveform_extractor.nbefore

    peak_values = {}

    for unit_id in unit_ids:
        template = waveform_extractor.get_template(unit_id)

        if mode == 'extremum':
            if peak_sign == 'both':
                values = np.max(np.abs(template), axis=0)
            elif peak_sign == 'neg':
                values = -np.min(template, axis=0)
            elif peak_sign == 'pos':
                values = np.max(template, axis=0)
        elif mode == 'at_index':
            if peak_sign == 'both':
                values = np.abs(template[before, :])
            elif peak_sign == 'neg':
                values = -template[before, :]
            elif peak_sign == 'pos':
                values = template[before, :]

        peak_values[unit_id] = values

    return peak_values


def get_template_extremum_channel(waveform_extractor, peak_sign='neg', outputs='id'):
    """
    Compute for each unit on which channel id the extremum is.
    """
    unit_ids = waveform_extractor.sorting.unit_ids
    channel_ids = waveform_extractor.recording.channel_ids

    peak_values = get_template_amplitudes(waveform_extractor, peak_sign=peak_sign)
    extremum_channels_id = {}
    extremum_channels_index = {}
    for unit_id in unit_ids:
        max_ind = np.argmax(peak_values[unit_id])
        extremum_channels_id[unit_id] = channel_ids[max_ind]
        extremum_channels_index[unit_id] = max_ind

    if outputs == 'id':
        return extremum_channels_id
    elif outputs == 'index':
        return extremum_channels_index


def get_template_channel_sparsity(waveform_extractor, method='best_channels',
                                  peak_sign='neg', num_channels=None, radius_um=None, outputs='id'):
    """
    Get channel sparsity for each template with several methods:
      * "best_channels": get N best channel, channels are ordered in that case
      * "radius": radius un um around the best channel, channels are not ordered
      * "threshold" : TODO
      
    """
    assert method in ('best_channels', 'radius', 'threshold')
    assert outputs in ('id', 'index')
    we = waveform_extractor

    unit_ids = we.sorting.unit_ids
    channel_ids = we.recording.channel_ids

    sparsity_with_index = {}
    if method == 'best_channels':
        assert num_channels is not None
        # take 
        peak_values = get_template_amplitudes(we, peak_sign=peak_sign)
        for unit_id in unit_ids:
            chan_inds = np.argsort(np.abs(peak_values[unit_id]))[::-1]
            chan_inds = chan_inds[:num_channels]
            sparsity_with_index[unit_id] = chan_inds

    elif method == 'radius':
        assert radius_um is not None
        best_chan = get_template_extremum_channel(we, outputs='index')
        distances = get_channel_distances(we.recording)
        for unit_id in unit_ids:
            chan_ind = best_chan[unit_id]
            chan_inds, = np.nonzero(distances[chan_ind, :] <= radius_um)
            sparsity_with_index[unit_id] = chan_inds

    elif method == 'threshold':
        raise NotImplementedError

    # handle output ids or indexes
    if outputs == 'id':
        sparsity_with_id = {}
        for unit_id in unit_ids:
            chan_inds = sparsity_with_index[unit_id]
            sparsity_with_id[unit_id] = channel_ids[chan_inds]
        return sparsity_with_id
    elif outputs == 'index':
        return sparsity_with_index


def get_template_extremum_channel_peak_shift(waveform_extractor, peak_sign='neg'):
    """
    In some situtaion some sorters, return spike index with a smal shift related to the extremum peak
    (min or max).
    
    Here a function to estimtate this shift.
    
    This function is internally used by `get_spike_amplitudes()` to accuratly retrieve the min/max amplitudes
    """
    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids

    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign)

    shifts = {}
    for unit_id in unit_ids:
        chan_id = extremum_channels_ids[unit_id]
        chan_ind = recording.id_to_index(chan_id)

        template = waveform_extractor.get_template(unit_id)

        if peak_sign == 'both':
            peak_pos = np.argmax(np.abs(template[:, chan_ind]))
        elif peak_sign == 'neg':
            peak_pos = np.argmin(template[:, chan_ind])
        elif peak_sign == 'pos':
            peak_pos = np.argmax(template[:, chan_ind])
        shift = peak_pos - waveform_extractor.nbefore
        shifts[unit_id] = shift

    return shifts


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

    best_channel_ids = get_template_channel_sparsity(waveform_extractor, method='best_channels',
                                                     peak_sign=peak_sign, num_channels=num_channels, outputs='id')

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
