import numpy as np
import scipy.optimize
import warnings

from .template_tools import get_template_channel_sparsity, get_template_amplitudes
from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension
from spikeinterface.core.job_tools import _shared_job_kwargs_doc


dtype_localize_by_method = {
    'center_of_mass':  [('x', 'float64'), ('y', 'float64')],
    'monopolar_triangulation': [('x', 'float64'),  ('y', 'float64'), ('z', 'float64'), ('alpha', 'float64')],
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

        self.unit_locations = None

    def _set_params(self, method='center_of_mass', method_kwargs={}):

        params = dict(method=method,
                      method_kwargs=method_kwargs)
        return params

    def _specific_load_from_folder(self):
        self.unit_locations = np.load(self.extension_folder / 'unit_locations.npy')

    def _reset(self):
        self.unit_locations = None

    def _specific_select_units(self, unit_ids, new_waveforms_folder):
        unit_inds = self.waveform_extractor.sorting.ids_to_indices(unit_ids)

        new_unit_location = self.unit_locations[unit_inds]
        np.save(new_waveforms_folder / self.extension_name / 'unit_locations.npy', new_unit_location)

    def run(self, **job_kwargs):
        method = self._params['method']
        method_kwargs = self._params['method_kwargs']
        
        assert method in possible_localization_methods

        if method == 'center_of_mass':
            unit_location = compute_center_of_mass(self.waveform_extractor,  **method_kwargs)
        elif method == 'monopolar_triangulation':
            unit_location = compute_monopolar_triangulation(self.waveform_extractor,  **method_kwargs)
        self.unit_locations = unit_location
        np.save(self.extension_folder /
                'unit_locations.npy', self.unit_locations)

    def get_data(self, outputs='numpy'):
        if outputs == 'numpy':
            return self.unit_locations

        elif outputs == 'by_unit':
            locations_by_unit = {}
            for unit_ind, unit_id in enumerate(self.waveform_extractor.sorting.unit_ids):
                locations_by_unit[unit_id] = self.unit_locations[unit_ind]
            return locations_by_unit


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
        'numpy' (default) / 'numpy_dtype' / 'dict'
    method_kwargs: 
        Other kwargs depending on the method.

    Returns
    -------
    unit_locations: np.array
        unit location with shape (num_unit, 2) or (num_unit, 3) or (num_unit, 3) (with alpha)
    """
    folder = waveform_extractor.folder
    ext_folder = folder / UnitLocationsCalculator.extension_name

    if load_if_exists and ext_folder.is_dir():
        ulc = UnitLocationsCalculator.load_from_folder(folder)
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


def make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um, initial_z = 20):
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
              [x0[0] + max_distance_um,  x0[1] + max_distance_um, max_distance_um*10, max_alpha])

    return x0, bounds


# ---- 
# optimizer "least_square"

def estimate_distance_error(vec, wf_ptp, local_contact_locations):
    # vec dims ar (x, y, z amplitude_factor)
    # given that for contact_location x=dim0 + z=dim1 and y is orthogonal to probe
    dist = np.sqrt(((local_contact_locations - vec[np.newaxis, :2])**2).sum(axis=1) + vec[2]**2)
    ptp_estimated = vec[3] / dist
    err = wf_ptp - ptp_estimated
    return err

# ---- 
# optimizer "minimize_with_log_penality"
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


def compute_monopolar_triangulation(waveform_extractor, optimizer='least_square', radius_um=50, max_distance_um=1000,
                                    return_alpha=False):
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

        # run optimization
        if optimizer == 'least_square':
            x0, bounds = make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um)
            args = (wf_ptp, local_contact_locations)
            output = scipy.optimize.least_squares(estimate_distance_error, x0=x0, bounds=bounds, args = args)
            unit_location[i] = tuple(output['x'])
        elif optimizer == 'minimize_with_log_penality':
            x0, bounds = make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um)
            x0 = x0[:3]
            bounds = [(bounds[0][i], bounds[1][i]) for i in range(3)]
            maxptp = wf_ptp.max()
            args = (wf_ptp, local_contact_locations, maxptp)
            output = scipy.optimize.minimize(estimate_distance_error_with_log, x0=x0, bounds=bounds, args=args)
            unit_location[i][:3] = tuple(output['x'])
            # final alpha
            q = ptp_at(*output['x'], 1.0, local_contact_locations)
            unit_location[i][3] = (wf_ptp * q).sum() / np.square(q).sum()            
        
        

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