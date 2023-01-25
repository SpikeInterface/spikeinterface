import warnings

import numpy as np
import scipy.optimize

from scipy.spatial.distance import cdist

try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

from ..core import compute_sparsity
from ..core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension



dtype_localize_by_method = {
    'center_of_mass': [('x', 'float64'), ('y', 'float64')],
    'peak_channel': [('x', 'float64'), ('y', 'float64')],
    'monopolar_triangulation': [('x', 'float64'), ('y', 'float64'), ('z', 'float64'), ('alpha', 'float64')],
}

possible_localization_methods = list(dtype_localize_by_method.keys())


class UnitLocationsCalculator(BaseWaveformExtractorExtension):
    """
    Comput unit locations from WaveformExtractor.
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object
    """
    extension_name = 'unit_locations'

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

    def _set_params(self, method='center_of_mass', method_kwargs={}):

        params = dict(method=method,
                      method_kwargs=method_kwargs)
        return params

    def _select_extension_data(self, unit_ids):
        unit_inds = self.waveform_extractor.sorting.ids_to_indices(unit_ids)
        new_unit_location = self._extension_data['unit_locations'][unit_inds]
        return dict(unit_locations=new_unit_location)

    def _run(self, **job_kwargs):
        method = self._params['method']
        method_kwargs = self._params['method_kwargs']
        
        assert method in possible_localization_methods

        if method == 'center_of_mass':
            unit_location = compute_center_of_mass(self.waveform_extractor,  **method_kwargs)
        elif method == 'monopolar_triangulation':
            unit_location = compute_monopolar_triangulation(self.waveform_extractor,  **method_kwargs)
        self._extension_data['unit_locations'] = unit_location

    def get_data(self, outputs='numpy'):
        """
        Get the computed unit locations.

        Parameters
        ----------
        outputs : str, optional
            'numpy' or 'by_unit', by default 'numpy'

        Returns
        -------
        unit_locations : np.array or dict
            The unit locations as a Nd array (outputs='numpy') or
            as a dict with units as key and locations as values.
        """
        if outputs == 'numpy':
            return self._extension_data['unit_locations']

        elif outputs == 'by_unit':
            locations_by_unit = {}
            for unit_ind, unit_id in enumerate(self.waveform_extractor.sorting.unit_ids):
                locations_by_unit[unit_id] = self._extension_data['unit_locations'][unit_ind]
            return locations_by_unit

    @staticmethod
    def get_extension_function():
        return compute_unit_locations


WaveformExtractor.register_extension(UnitLocationsCalculator)


def compute_unit_locations(waveform_extractor, 
                           load_if_exists=False,
                           method='center_of_mass', 
                           outputs='numpy', **method_kwargs):
    """
    Localize units in 2D or 3D with several methods given the template.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object.
    load_if_exists : bool, optional, default: False
        Whether to load precomputed unit locations, if they already exist.
    method: str
        'center_of_mass' / 'monopolar_triangulation'
    outputs: str 
        'numpy' (default) / 'by_unit'
    method_kwargs: 
        Other kwargs depending on the method.

    Returns
    -------
    unit_locations: np.array
        unit location with shape (num_unit, 2) or (num_unit, 3) or (num_unit, 3) (with alpha)
    """
    if load_if_exists and waveform_extractor.is_extension(UnitLocationsCalculator.extension_name):
        ulc = waveform_extractor.load_extension(UnitLocationsCalculator.extension_name)
    else:
        ulc = UnitLocationsCalculator(waveform_extractor)
        ulc.set_params(method=method, method_kwargs=method_kwargs)
        ulc.run()

    unit_locations = ulc.get_data(outputs=outputs)
    return unit_locations


def localize_units(*args, **kwargs):
    warnings.warn("The 'localize_units' function is deprecated. "
                  "Use 'compute_unit_locations' instead",
                  DeprecationWarning, stacklevel=2)
    return compute_unit_locations(*args, **kwargs)


def make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um, initial_z=20):

    # constant for initial guess and bounds
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
              [x0[0] + max_distance_um, x0[1] + max_distance_um, max_distance_um * 10, max_alpha])

    return x0, bounds


def solve_monopolar_triangulation(wf_ptp, local_contact_locations, max_distance_um, optimizer):
    x0, bounds = make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um)

    if optimizer == 'least_square':
        args = (wf_ptp, local_contact_locations)
        try:
            output = scipy.optimize.least_squares(estimate_distance_error, x0=x0, bounds=bounds, args=args)
            return tuple(output['x'])
        except Exception as e:
            print(f"scipy.optimize.least_squares error: {e}")
            return (np.nan, np.nan, np.nan, np.nan)

    if optimizer == 'minimize_with_log_penality':
        x0 = x0[:3]
        bounds = [(bounds[0][0], bounds[1][0]), (bounds[0][1], bounds[1][1]), (bounds[0][2], bounds[1][2])]
        maxptp = wf_ptp.max()
        args = (wf_ptp, local_contact_locations, maxptp)
        try:
            output = scipy.optimize.minimize(estimate_distance_error_with_log, x0=x0, bounds=bounds, args=args)
            # final alpha
            q = ptp_at(*output['x'], 1.0, local_contact_locations)
            alpha = (wf_ptp * q).sum() / np.square(q).sum()
            return (*output['x'], alpha)
        except Exception as e:
            print(f"scipy.optimize.minimize error: {e}")
            return (np.nan, np.nan, np.nan, np.nan)


# ----
# optimizer "least_square"


def estimate_distance_error(vec, wf_ptp, local_contact_locations):
    # vec dims ar (x, y, z amplitude_factor)
    # given that for contact_location x=dim0 + z=dim1 and y is orthogonal to probe
    dist = np.sqrt(((local_contact_locations - vec[np.newaxis, :2])**2).sum(axis=1) + vec[2]**2)
    ptp_estimated = vec[3] / dist
    err = wf_ptp - ptp_estimated
    return err


# ----
# optimizer "minimize_with_log_penality"


def ptp_at(x, y, z, alpha, local_contact_locations):
    return alpha / np.sqrt(
        np.square(x - local_contact_locations[:, 0])
        + np.square(y - local_contact_locations[:, 1])
        + np.square(z)
    )


def estimate_distance_error_with_log(vec, wf_ptp, local_contact_locations, maxptp):
    x, y, z = vec
    q = ptp_at(x, y, z, 1.0, local_contact_locations)
    alpha = (q * wf_ptp / maxptp).sum() / (q * q).sum()
    err = np.square(wf_ptp / maxptp - ptp_at(x, y, z, alpha, local_contact_locations)).mean() - np.log1p(10.0 * z) / 10000.0
    return err


def compute_monopolar_triangulation(waveform_extractor, optimizer='minimize_with_log_penality',
                                    radius_um=50, max_distance_um=1000, return_alpha=False):
    '''
    Localize unit with monopolar triangulation.
    This method is from Julien Boussard, Erdem Varol and Charlie Windolf
    https://www.biorxiv.org/content/10.1101/2021.11.05.467503v1

    There are 2 implementations of the 2 optimizer variants:
      * https://github.com/int-brain-lab/spikes_localization_registration/blob/main/localization_pipeline/localizer.py
      * https://github.com/cwindolf/spike-psvae/blob/main/spike_psvae/localization.py

    Important note about axis:
      * x/y are dimmension on the probe plane (dim0, dim1)
      * y is the depth by convention
      * z it the orthogonal axis to the probe plan (dim2)

    Code from Erdem, Julien and Charlie do not use the same convention!!!


    Parameters
    ----------
    waveform_extractor:WaveformExtractor
        A waveform extractor object
    method: str  ('least_square', 'minimize_with_log_penality')
       2 variants of the method
    radius_um: float
        For channel sparsity
    max_distance_um: float
        to make bounddary in x, y, z and also for alpha
    return_alpha: bool default False
        Return or not the alpha value

    Returns
    -------
    unit_location: np.array
        3d or 4d, x, y, z, alpha
        alpha is the amplitude at source estimation
    '''
    assert optimizer in ('least_square', 'minimize_with_log_penality')

    unit_ids = waveform_extractor.sorting.unit_ids

    recording = waveform_extractor.recording
    contact_locations = recording.get_channel_locations()

    sparsity = compute_sparsity(waveform_extractor, method='radius', radius_um=radius_um)
    templates = waveform_extractor.get_all_templates(mode='average')

    unit_location = np.zeros((unit_ids.size, 4), dtype='float64')
    for i, unit_id in enumerate(unit_ids):
        chan_inds = sparsity.unit_id_to_channel_indices[unit_id]
        local_contact_locations = contact_locations[chan_inds, :]

        # wf is (nsample, nchan) - chann is only nieghboor
        wf = templates[i, :, :]
        wf_ptp = wf[:, chan_inds].ptp(axis=0)
        unit_location[i] = solve_monopolar_triangulation(wf_ptp, local_contact_locations, max_distance_um, optimizer)

    if not return_alpha:
        unit_location = unit_location[:, :3]

    return unit_location


def compute_center_of_mass(waveform_extractor, peak_sign='neg', radius_um=50):
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

    # TODO
    sparsity = compute_sparsity(waveform_extractor, peak_sign=peak_sign, method='radius', radius_um=radius_um)
    templates = waveform_extractor.get_all_templates(mode='average')

    unit_location = np.zeros((unit_ids.size, 2), dtype='float64')
    for i, unit_id in enumerate(unit_ids):
        chan_inds = sparsity.unit_id_to_channel_indices[unit_id]
        local_contact_locations = contact_locations[chan_inds, :]

        wf = templates[i, :, :]

        wf_ptp = wf[:, chan_inds].ptp(axis=0)

        # center of mass
        com = np.sum(wf_ptp[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_ptp)
        unit_location[i, :] = com

    return unit_location


# ---
# waveform cleaning for localization. could be moved to another file


def make_shell(channel, geom, n_jumps=1):
    """See make_shells"""
    pt = geom[channel]
    dists = cdist([pt], geom).ravel()
    radius = np.unique(dists)[1 : n_jumps + 1][-1]
    return np.setdiff1d(np.flatnonzero(dists <= radius + 1e-8), [channel])


def make_shells(geom, n_jumps=1):
    """Get the neighbors of a channel within a radius

    That radius is found by figuring out the distance to the closest channel,
    then the channel which is the next closest (but farther than the closest),
    etc... for n_jumps.

    So, if n_jumps is 1, it will return the indices of channels which are
    as close as the closest channel. If n_jumps is 2, it will include those
    and also the indices of the next-closest channels. And so on...

    Returns
    -------
    shell_neighbors : list
        List of length geom.shape[0] (aka, the number of channels)
        The ith entry in the list is an array with the indices of the neighbors
        of the ith channel.
        i is not included in these arrays (a channel is not in its own shell).
    """
    return [make_shell(c, geom, n_jumps=n_jumps) for c in range(geom.shape[0])]


def make_radial_order_parents(
    geom, neighbours_mask, n_jumps_per_growth=1, n_jumps_parent=3
):
    """Pre-computes a helper data structure for enforce_decrease_shells"""
    n_channels = len(geom)

    # which channels should we consider as possible parents for each channel?
    shells = make_shells(geom, n_jumps=n_jumps_parent)

    radial_parents = []
    for channel, neighbors in enumerate(neighbours_mask):
        channel_parents = []

        # convert from boolean mask to list of indices
        neighbors = np.flatnonzero(neighbors)

        # the closest shell will do nothing
        already_seen = [channel]
        shell0 = make_shell(channel, geom, n_jumps=n_jumps_per_growth)
        already_seen += sorted(c for c in shell0 if c not in already_seen)

        # so we start at the second jump
        jumps = 2
        while len(already_seen) < (neighbors < n_channels).sum():
            # grow our search -- what are the next-closest channels?
            new_shell = make_shell(
                channel, geom, n_jumps=jumps * n_jumps_per_growth
            )
            new_shell = list(
                sorted(
                    c
                    for c in new_shell
                    if (c not in already_seen) and (c in neighbors)
                )
            )

            # for each new channel, find the intersection of the channels
            # from previous shells and that channel's shell in `shells`
            for new_chan in new_shell:
                parents = np.intersect1d(shells[new_chan], already_seen)
                parents_rel = np.flatnonzero(np.isin(neighbors, parents))
                if not len(parents_rel):
                    # this can happen for some strange geometries. in that case, bail.
                    continue
                channel_parents.append(
                    (np.flatnonzero(neighbors == new_chan).item(), parents_rel)
                )

            # add this shell to what we have seen
            already_seen += new_shell
            jumps += 1

        radial_parents.append(channel_parents)

    return radial_parents


def enforce_decrease_shells_ptp(
    wf_ptp, maxchan, radial_parents, in_place=False
):
    """Radial enforce decrease"""
    (C,) = wf_ptp.shape

    # allocate storage for decreasing version of PTP
    decreasing_ptp = wf_ptp if in_place else wf_ptp.copy()

    # loop to enforce ptp decrease from parent shells
    for c, parents_rel in radial_parents[maxchan]:
        if decreasing_ptp[c] > decreasing_ptp[parents_rel].max():
            decreasing_ptp[c] *= decreasing_ptp[parents_rel].max() / decreasing_ptp[c]

    return decreasing_ptp


if HAVE_NUMBA:
    enforce_decrease_shells = numba.jit(enforce_decrease_shells_ptp, nopython=True)
