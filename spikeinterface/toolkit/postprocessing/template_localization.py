import numpy as np
import scipy.optimize

from .template_tools import get_template_channel_sparsity, get_template_amplitudes



dtype_localize_by_method = {
    'center_of_mass':  [('x', 'float64'), ('z', 'float64')],
    'monopolar_triangulation': [('x', 'float64'),  ('z', 'float64'), ('y', 'float64'), ('alpha', 'float64')],
}

_possible_localization_methods = list(dtype_localize_by_method.keys())


def localize_template(waveform_extractor, method='center_of_mass', output='numpy', **method_kwargs):
    assert method in _possible_localization_methods
    
    if method == 'center_of_mass':
        unit_location = compute_center_of_mass(waveform_extractor,  **method_kwargs)
    elif method == 'monopolar_triangulation':
        unit_location = compute_monopolar_triangulation(waveform_extractor,  **method_kwargs)
    
    # handle some outputs
    if output == 'dict':
        return dict(zip(unit_ids, coms))
    elif output == 'numpy':
        return unit_location
    
    


def _minimize_dist(vec, wf_ptp, local_contact_locations):
    # vec dims ar (x, z, y, amplitude_factor)
    # given that for contact_location x=dim0 + z=dim1 and y is orthogonal to probe
    dist = np.sqrt(((local_contact_locations - vec[np.newaxis, :2])**2).sum(axis=1) + vec[2]**2)
    ptp_estimated = vec[3] / dist
    err = wf_ptp  - ptp_estimated
    return err
    

def compute_monopolar_triangulation(waveform_extractor, radius_um=150):
    '''
    Localize unit with monopolar triangulation.
    This method is from Julien Boussard
    https://www.biorxiv.org/content/10.1101/2021.11.05.467503v1
    '''


    unit_ids = waveform_extractor.sorting.unit_ids

    recording = waveform_extractor.recording
    contact_locations = recording.get_channel_locations()

    channel_sparsity = get_template_channel_sparsity(waveform_extractor, method='radius', radius_um=radius_um,
                                                                                                    outputs='index')
    
    templates = waveform_extractor.get_all_templates(mode='average')
    #~ amplitudes = get_template_amplitudes(waveform_extractor, peak_sign=peak_sign)

    unit_location = np.zeros((unit_ids.size, 3), dtype='float64')
    for i, unit_id in enumerate(unit_ids):
    
        chan_inds = channel_sparsity[unit_id]
    
        local_contact_locations = contact_locations[chan_inds, :]

        # wf is (nsample, nchan) - chann is only nieghboor
        wf = templates[i, :, :]
        wf_ptp = wf[:, chan_inds].ptp(axis=0)

        # constant for initial guess and bounds
        max_border = 300
        max_distance = 1000
        max_alpha = max(wf_ptp) * max_distance

        # initial guess is the center of mass
        com = np.sum(wf_ptp[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_ptp)
        x0 = np.zeros(4, dtype='float32')
        x0[:2] = com
        x0[2] = 20
        x0[3] = max_alpha / 50.
        
        # bounds depend on geometry
        bounds = ([x0[0] - max_border, x0[1] - max_border, 1, 0],
                  [x0[0] + max_border,  x0[1] + max_border, max_border*10, max_alpha])
        # print('x0', x0)
        # print('bounds',bounds)


        # 
        # print('z_initial', z_initial)
        
        args = (wf_ptp, local_contact_locations)
        # print('x0', x0)
        # print('bounds', bounds)
        output = scipy.optimize.least_squares(_minimize_dist, x0=x0, bounds=bounds, args = args)
        #~ print(output['x'][3],  max(wf_ptp) * max_distance)
        # print('i', com, output['x'][:2])
        # print('yep')
        # print('output', output)
        # print('output', output['x'].shape, output['x'])
        
        unit_location[i] = tuple(output['x'][:3])
        
    return unit_location


def compute_center_of_mass(waveform_extractor, peak_sign='neg', num_channels=10):
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

    coms = np.zeros((unit_ids.size, 2), dtype='float64')
    for i, unit_id in enumerate(unit_ids):
        chan_ids = best_channel_ids[unit_id]
        chan_inds = recording.ids_to_indices(chan_ids)

        amps = amplitudes[unit_id][chan_inds]
        amps = np.abs(amps)
        com = np.sum(amps[:, np.newaxis] * locations[chan_inds, :], axis=0) / np.sum(amps)
        coms[i, :] = com
        #~ coms.append(com)

    #~ coms = dict(zip(unit_ids, coms))

    return coms
