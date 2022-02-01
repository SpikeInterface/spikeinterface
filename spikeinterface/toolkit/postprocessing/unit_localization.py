import numpy as np
import scipy.optimize

from .template_tools import get_template_channel_sparsity, get_template_amplitudes



dtype_localize_by_method = {
    'center_of_mass':  [('x', 'float64'), ('y', 'float64')],
    'monopolar_triangulation': [('x', 'float64'),  ('y', 'float64'), ('z', 'float64'), ('alpha', 'float64')],
}

possible_localization_methods = list(dtype_localize_by_method.keys())


def localize_units(waveform_extractor, method='center_of_mass', output='numpy', **method_kwargs):
    """
    Localise units in 2D or 3D with several methods given the template.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object
    method: str
        'center_of_mass' / 'monopolar_triangulation'
    output: str 
        'numpy' (default) / 'numpy_dtype' / 'dict'
    method_kwargs: 
        other kwargs method dependant

    Returns
    -------
    unit_location: np.array
        unit location with shape (num_unit, 2) or (num_unit, 3) or (num_unit, 3) (with alpha)
    """
    assert method in possible_localization_methods

    if method == 'center_of_mass':
        unit_location = compute_center_of_mass(waveform_extractor,  **method_kwargs)
    elif method == 'monopolar_triangulation':
        unit_location = compute_monopolar_triangulation(waveform_extractor,  **method_kwargs)

    # handle some outputs
    if output == 'numpy':
        return unit_location
    elif output == 'dict':
        return dict(zip(waveform_extractor.sorting.unit_ids, unit_location))
    elif output == 'numpy_dtype':
        raise NotImplementedError


def estimate_distance_error(vec, wf_ptp, local_contact_locations):
    # vec dims ar (x, y, z amplitude_factor)
    # given that for contact_location x=dim0 + z=dim1 and y is orthogonal to probe
    dist = np.sqrt(((local_contact_locations - vec[np.newaxis, :2])**2).sum(axis=1) + vec[2]**2)
    ptp_estimated = vec[3] / dist
    err = wf_ptp - ptp_estimated
    return err


def make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um):
    # constant for initial guess and bounds
    initial_z = 20

    ind_max = np.argmax(wf_ptp)
    max_ptp = wf_ptp[ind_max]
    max_alpha = max_ptp * max_distance_um

    # initial guess is the center of mass
    com = np.sum(wf_ptp[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_ptp)
    x0 = np.zeros(4, dtype='float32')
    x0[:2] = com
    x0[2] = initial_z
    initial_alpha = np.sqrt(np.sum((com - local_contact_locations[ind_max, :])**2) + initial_z**2) * max_ptp
    x0[3] = initial_alpha

    # bounds depend on initial guess
    bounds = ([x0[0] - max_distance_um, x0[1] - max_distance_um, 1, 0],
              [x0[0] + max_distance_um,  x0[1] + max_distance_um, max_distance_um*10, max_alpha])

    return x0, bounds


def compute_monopolar_triangulation(waveform_extractor, radius_um=50, max_distance_um=1000, return_alpha=False):
    '''
    Localize unit with monopolar triangulation.
    This method is from Julien Boussard
    https://www.biorxiv.org/content/10.1101/2021.11.05.467503v1

    Important note about axis:
      * x/y are dimmension on the probe plane (dim0, dim1)
      * y is the depth by convention
      * z it the orthogonal axis to the probe plan


    Parameters
    ----------
    waveform_extractor:WaveformExtractor
        A waveform extractor object
    radius_um: float
        For channel sparsiry
    max_distance_um: float
        to make bounddary in x, y, z and also for alpha

    Returns
    -------
    unit_location: np.array
        3d or 4d, x, y, z, alpha
        alpha is the amplitude at source estimation
    '''

    unit_ids = waveform_extractor.sorting.unit_ids

    recording = waveform_extractor.recording
    contact_locations = recording.get_channel_locations()


    channel_sparsity = get_template_channel_sparsity(waveform_extractor, method='radius', 
                                                     radius_um=radius_um, outputs='index')

    templates = waveform_extractor.get_all_templates(mode='average')

    unit_location = np.zeros((unit_ids.size, 4), dtype='float64')
    for i, unit_id in enumerate(unit_ids):

        chan_inds = channel_sparsity[unit_id]

        local_contact_locations = contact_locations[chan_inds, :]

        # wf is (nsample, nchan) - chann is only nieghboor
        wf = templates[i, :, :]

        wf_ptp = wf[:, chan_inds].ptp(axis=0)

        x0, bounds = make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um)

        # run optimization
        args = (wf_ptp, local_contact_locations)
        output = scipy.optimize.least_squares(estimate_distance_error, x0=x0, bounds=bounds, args = args)

        unit_location[i] = tuple(output['x'])

    if not return_alpha:
        unit_location = unit_location[:, :3]

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
    unit_location: np.array
    '''
    unit_ids = waveform_extractor.sorting.unit_ids

    recording = waveform_extractor.recording
    contact_locations = recording.get_channel_locations()

    channel_sparsity = get_template_channel_sparsity(waveform_extractor, method='best_channels',
                                                     num_channels=num_channels, outputs='index')

    templates = waveform_extractor.get_all_templates(mode='average')

    unit_location = np.zeros((unit_ids.size, 2), dtype='float64')
    for i, unit_id in enumerate(unit_ids):
        chan_inds = channel_sparsity[unit_id]
        local_contact_locations = contact_locations[chan_inds, :]

        wf = templates[i, :, :]

        wf_ptp = wf[:, chan_inds].ptp(axis=0)

        # center of mass
        com = np.sum(wf_ptp[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_ptp)
        unit_location[i, :] = com

    return unit_location