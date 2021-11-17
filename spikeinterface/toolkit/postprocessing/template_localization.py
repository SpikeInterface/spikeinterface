import numpy as np

from .template_tools import get_template_channel_sparsity, get_template_amplitudes

def compute_unit_centers_of_mass(waveform_extractor, peak_sign='neg', num_channels=10):
    '''
    Computes the center of mass (COM) of a unit based on the template amplitudes.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    peak_sign: str
        Sign of the template to compute best channels ('neg', 'pos', 'both')
    num_channels: int
        Number of channels used to compute COM

    Returns
    -------
    centers_of_mass: dict of np.array
        Dictionary with unit ids as keys and centers of mass as values
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
        chan_ids = best_channel_ids[unit_id]
        chan_inds = recording.ids_to_indices(chan_ids)

        amps = amplitudes[unit_id][chan_inds]
        amps = np.abs(amps)
        com = np.sum(amps[:, np.newaxis] * locations[chan_inds, :], axis=0) / np.sum(amps)
        coms.append(com)

    coms = dict(zip(unit_ids, coms))

    return coms