import warnings

import numpy as np


try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

from ..core import compute_sparsity
from ..core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension
from ..core.template_tools import get_template_extremum_channel


dtype_localize_by_method = {
    "center_of_mass": [("x", "float64"), ("y", "float64")],
    "grid_convolution": [("x", "float64"), ("y", "float64")],
    "peak_channel": [("x", "float64"), ("y", "float64")],
    "monopolar_triangulation": [("x", "float64"), ("y", "float64"), ("z", "float64"), ("alpha", "float64")],
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

    extension_name = "unit_locations"

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

    def _set_params(self, method="center_of_mass", method_kwargs={}):
        params = dict(method=method, method_kwargs=method_kwargs)
        return params

    def _select_extension_data(self, unit_ids):
        unit_inds = self.waveform_extractor.sorting.ids_to_indices(unit_ids)
        new_unit_location = self._extension_data["unit_locations"][unit_inds]
        return dict(unit_locations=new_unit_location)

    def _run(self, **job_kwargs):
        method = self._params["method"]
        method_kwargs = self._params["method_kwargs"]

        assert method in possible_localization_methods

        if method == "center_of_mass":
            unit_location = compute_center_of_mass(self.waveform_extractor, **method_kwargs)
        elif method == "grid_convolution":
            unit_location = compute_grid_convolution(self.waveform_extractor, **method_kwargs)
        elif method == "monopolar_triangulation":
            unit_location = compute_monopolar_triangulation(self.waveform_extractor, **method_kwargs)
        self._extension_data["unit_locations"] = unit_location

    def get_data(self, outputs="numpy"):
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
        if outputs == "numpy":
            return self._extension_data["unit_locations"]

        elif outputs == "by_unit":
            locations_by_unit = {}
            for unit_ind, unit_id in enumerate(self.waveform_extractor.sorting.unit_ids):
                locations_by_unit[unit_id] = self._extension_data["unit_locations"][unit_ind]
            return locations_by_unit

    @staticmethod
    def get_extension_function():
        return compute_unit_locations


WaveformExtractor.register_extension(UnitLocationsCalculator)


def compute_unit_locations(
    waveform_extractor, load_if_exists=False, method="center_of_mass", outputs="numpy", **method_kwargs
):
    """
    Localize units in 2D or 3D with several methods given the template.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object.
    load_if_exists : bool, default: False
        Whether to load precomputed unit locations, if they already exist.
    method: str
        'center_of_mass' / 'monopolar_triangulation' / 'grid_convolution'
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


def make_initial_guess_and_bounds(wf_data, local_contact_locations, max_distance_um, initial_z=20):
    # constant for initial guess and bounds
    ind_max = np.argmax(wf_data)
    max_ptp = wf_data[ind_max]
    max_alpha = max_ptp * max_distance_um

    # initial guess is the center of mass
    com = np.sum(wf_data[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_data)
    x0 = np.zeros(4, dtype="float32")
    x0[:2] = com
    x0[2] = initial_z
    initial_alpha = np.sqrt(np.sum((com - local_contact_locations[ind_max, :]) ** 2) + initial_z**2) * max_ptp
    x0[3] = initial_alpha

    # bounds depend on initial guess
    bounds = (
        [x0[0] - max_distance_um, x0[1] - max_distance_um, 1, 0],
        [x0[0] + max_distance_um, x0[1] + max_distance_um, max_distance_um * 10, max_alpha],
    )

    return x0, bounds


def solve_monopolar_triangulation(wf_data, local_contact_locations, max_distance_um, optimizer):
    import scipy.optimize

    x0, bounds = make_initial_guess_and_bounds(wf_data, local_contact_locations, max_distance_um)

    if optimizer == "least_square":
        args = (wf_data, local_contact_locations)
        try:
            output = scipy.optimize.least_squares(estimate_distance_error, x0=x0, bounds=bounds, args=args)
            return tuple(output["x"])
        except Exception as e:
            print(f"scipy.optimize.least_squares error: {e}")
            return (np.nan, np.nan, np.nan, np.nan)

    if optimizer == "minimize_with_log_penality":
        x0 = x0[:3]
        bounds = [(bounds[0][0], bounds[1][0]), (bounds[0][1], bounds[1][1]), (bounds[0][2], bounds[1][2])]
        max_data = wf_data.max()
        args = (wf_data, local_contact_locations, max_data)
        try:
            output = scipy.optimize.minimize(estimate_distance_error_with_log, x0=x0, bounds=bounds, args=args)
            # final alpha
            q = data_at(*output["x"], 1.0, local_contact_locations)
            alpha = (wf_data * q).sum() / np.square(q).sum()
            return (*output["x"], alpha)
        except Exception as e:
            print(f"scipy.optimize.minimize error: {e}")
            return (np.nan, np.nan, np.nan, np.nan)


# ----
# optimizer "least_square"


def estimate_distance_error(vec, wf_data, local_contact_locations):
    # vec dims ar (x, y, z amplitude_factor)
    # given that for contact_location x=dim0 + z=dim1 and y is orthogonal to probe
    dist = np.sqrt(((local_contact_locations - vec[np.newaxis, :2]) ** 2).sum(axis=1) + vec[2] ** 2)
    data_estimated = vec[3] / dist
    err = wf_data - data_estimated
    return err


# ----
# optimizer "minimize_with_log_penality"


def data_at(x, y, z, alpha, local_contact_locations):
    return alpha / np.sqrt(
        np.square(x - local_contact_locations[:, 0]) + np.square(y - local_contact_locations[:, 1]) + np.square(z)
    )


def estimate_distance_error_with_log(vec, wf_data, local_contact_locations, max_data):
    x, y, z = vec
    q = data_at(x, y, z, 1.0, local_contact_locations)
    alpha = (q * wf_data / max_data).sum() / (q * q).sum()
    err = (
        np.square(wf_data / max_data - data_at(x, y, z, alpha, local_contact_locations)).mean()
        - np.log1p(10.0 * z) / 10000.0
    )
    return err


def compute_monopolar_triangulation(
    waveform_extractor,
    optimizer="minimize_with_log_penality",
    radius_um=75,
    max_distance_um=1000,
    return_alpha=False,
    enforce_decrease=False,
    feature="ptp",
):
    """
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
    enforce_decrease : bool (default False)
        Enforce spatial decreasingness for PTP vectors
    feature: string in ['ptp', 'energy', 'peak_voltage']
        The available features to consider for estimating the position via
        monopolar triangulation are peak-to-peak amplitudes ('ptp', default),
        energy ('energy', as L2 norm) or voltages at the center of the waveform
        ('peak_voltage')

    Returns
    -------
    unit_location: np.array
        3d or 4d, x, y, z, alpha
        alpha is the amplitude at source estimation
    """
    assert optimizer in ("least_square", "minimize_with_log_penality")

    assert feature in ["ptp", "energy", "peak_voltage"], f"{feature} is not a valid feature"
    unit_ids = waveform_extractor.sorting.unit_ids

    contact_locations = waveform_extractor.get_channel_locations()
    nbefore = waveform_extractor.nbefore

    sparsity = compute_sparsity(waveform_extractor, method="radius", radius_um=radius_um)
    templates = waveform_extractor.get_all_templates(mode="average")

    if enforce_decrease:
        neighbours_mask = np.zeros((templates.shape[0], templates.shape[2]), dtype=bool)
        for i, unit_id in enumerate(unit_ids):
            chan_inds = sparsity.unit_id_to_channel_indices[unit_id]
            neighbours_mask[i, chan_inds] = True
        enforce_decrease_radial_parents = make_radial_order_parents(contact_locations, neighbours_mask)
        best_channels = get_template_extremum_channel(waveform_extractor, outputs="index")

    unit_location = np.zeros((unit_ids.size, 4), dtype="float64")
    for i, unit_id in enumerate(unit_ids):
        chan_inds = sparsity.unit_id_to_channel_indices[unit_id]
        local_contact_locations = contact_locations[chan_inds, :]

        # wf is (nsample, nchan) - chann is only nieghboor
        wf = templates[i, :, :][:, chan_inds]
        if feature == "ptp":
            wf_data = wf.ptp(axis=0)
        elif feature == "energy":
            wf_data = np.linalg.norm(wf, axis=0)
        elif feature == "peak_voltage":
            wf_data = np.abs(wf[nbefore])

        # if enforce_decrease:
        #    enforce_decrease_shells_data(
        #        wf_data, best_channels[unit_id], enforce_decrease_radial_parents, in_place=True
        #    )

        unit_location[i] = solve_monopolar_triangulation(wf_data, local_contact_locations, max_distance_um, optimizer)

    if not return_alpha:
        unit_location = unit_location[:, :3]

    return unit_location


def compute_center_of_mass(waveform_extractor, peak_sign="neg", radius_um=75, feature="ptp"):
    """
    Computes the center of mass (COM) of a unit based on the template amplitudes.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    peak_sign: str
        Sign of the template to compute best channels ('neg', 'pos', 'both')
    radius_um: float
        Radius to consider in order to estimate the COM
    feature: str ['ptp', 'mean', 'energy', 'peak_voltage']
        Feature to consider for computation. Default is 'ptp'

    Returns
    -------
    unit_location: np.array
    """
    unit_ids = waveform_extractor.sorting.unit_ids

    recording = waveform_extractor.recording
    contact_locations = recording.get_channel_locations()

    assert feature in ["ptp", "mean", "energy", "peak_voltage"], f"{feature} is not a valid feature"

    sparsity = compute_sparsity(waveform_extractor, peak_sign=peak_sign, method="radius", radius_um=radius_um)
    templates = waveform_extractor.get_all_templates(mode="average")

    unit_location = np.zeros((unit_ids.size, 2), dtype="float64")
    for i, unit_id in enumerate(unit_ids):
        chan_inds = sparsity.unit_id_to_channel_indices[unit_id]
        local_contact_locations = contact_locations[chan_inds, :]

        wf = templates[i, :, :]

        if feature == "ptp":
            wf_data = (wf[:, chan_inds]).ptp(axis=0)
        elif feature == "mean":
            wf_data = (wf[:, chan_inds]).mean(axis=0)
        elif feature == "energy":
            wf_data = np.linalg.norm(wf[:, chan_inds], axis=0)
        elif feature == "peak_voltage":
            wf_data = wf[waveform_extractor.nbefore, chan_inds]

        # center of mass
        com = np.sum(wf_data[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_data)
        unit_location[i, :] = com

    return unit_location


@np.errstate(divide="ignore", invalid="ignore")
def compute_grid_convolution(
    waveform_extractor,
    peak_sign="neg",
    radius_um=40.0,
    upsampling_um=5,
    sigma_um=np.linspace(5.0, 25.0, 5),
    sigma_ms=0.25,
    margin_um=50,
    prototype=None,
    percentile=10,
    sparsity_threshold=0.01,
):
    """
    Estimate the positions of the templates from a large grid of fake templates

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    peak_sign: str
        Sign of the template to compute best channels ('neg', 'pos', 'both')
    radius_um: float
        Radius to consider for the fake templates
    upsampling_um: float
        Upsampling resolution for the grid of templates
    sigma_um: np.array
        Spatial decays of the fake templates
    sigma_ms: float
        The temporal decay of the fake templates
    margin_um: float
        The margin for the grid of fake templates
    prototype: np.array
        Fake waveforms for the templates. If None, generated as Gaussian
    percentile: float (default 10)
        The percentage  in [0, 100] of the best scalar products kept to
        estimate the position
    sparsity_threshold: float (default 0.01)
        The sparsity threshold (in 0-1) below which weights should be considered as 0.
    Returns
    -------
    unit_location: np.array
    """

    contact_locations = waveform_extractor.get_channel_locations()

    nbefore = waveform_extractor.nbefore
    nafter = waveform_extractor.nafter
    fs = waveform_extractor.sampling_frequency
    percentile = 100 - percentile
    assert 0 <= percentile <= 100, "Percentile should be in [0, 100]"
    assert 0 <= sparsity_threshold <= 1, "sparsity_threshold should be in [0, 1]"

    time_axis = np.arange(-nbefore, nafter) * 1000 / fs
    if prototype is None:
        prototype = np.exp(-(time_axis**2) / (2 * (sigma_ms**2)))

    prototype = prototype[:, np.newaxis]

    template_positions, weights, nearest_template_mask = get_grid_convolution_templates_and_weights(
        contact_locations, radius_um, upsampling_um, sigma_um, margin_um
    )

    templates = waveform_extractor.get_all_templates(mode="average")

    peak_channels = get_template_extremum_channel(waveform_extractor, peak_sign, outputs="index")
    unit_ids = waveform_extractor.sorting.unit_ids

    weights_sparsity_mask = weights > sparsity_threshold

    unit_location = np.zeros((unit_ids.size, 2), dtype="float64")
    for i, unit_id in enumerate(unit_ids):
        main_chan = peak_channels[unit_id]
        wf = templates[i, :, :]
        amplitude = wf[nbefore, main_chan]
        nearest_templates = nearest_template_mask[main_chan, :]

        channel_mask = np.sum(weights_sparsity_mask[:, :, nearest_templates], axis=(0, 2)) > 0
        num_templates = np.sum(nearest_templates)
        global_products = ((wf[:, channel_mask] / amplitude) * prototype).sum(axis=0)

        dot_products = np.zeros((weights.shape[0], num_templates), dtype=np.float32)
        for count in range(weights.shape[0]):
            w = weights[count, :, :][channel_mask, :][:, nearest_templates]
            # w = w / np.sum(w, axis=0)[np.newaxis, None]
            # w[np.isnan(w)] = 0.
            dot_products[count, :] = np.dot(global_products, w)

        dot_products = np.maximum(0, dot_products)
        if percentile < 100:
            thresholds = np.percentile(dot_products, percentile, axis=0)
            dot_products[dot_products < thresholds[np.newaxis, :]] = 0

        found_positions = np.zeros(2, dtype=np.float32)
        scalar_products = np.zeros(num_templates, dtype=np.float32)
        for count in range(weights.shape[0]):
            scalar_products += dot_products[count]
            found_positions += np.dot(dot_products[count], template_positions[nearest_templates])

        unit_location[i, :] = found_positions / scalar_products.sum()

    return unit_location


# ---
# waveform cleaning for localization. could be moved to another file


def make_shell(channel, geom, n_jumps=1):
    """See make_shells"""
    from scipy.spatial.distance import cdist

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


def make_radial_order_parents(geom, neighbours_mask, n_jumps_per_growth=1, n_jumps_parent=3):
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
            new_shell = make_shell(channel, geom, n_jumps=jumps * n_jumps_per_growth)
            new_shell = list(sorted(c for c in new_shell if (c not in already_seen) and (c in neighbors)))

            # for each new channel, find the intersection of the channels
            # from previous shells and that channel's shell in `shells`
            for new_chan in new_shell:
                parents = np.intersect1d(shells[new_chan], already_seen)
                parents_rel = np.flatnonzero(np.isin(neighbors, parents))
                if not len(parents_rel):
                    # this can happen for some strange geometries. in that case, bail.
                    continue
                channel_parents.append((np.flatnonzero(neighbors == new_chan).item(), parents_rel))

            # add this shell to what we have seen
            already_seen += new_shell
            jumps += 1

        radial_parents.append(channel_parents)

    return radial_parents


def enforce_decrease_shells_data(wf_data, maxchan, radial_parents, in_place=False):
    """Radial enforce decrease"""
    (C,) = wf_data.shape

    # allocate storage for decreasing version of data
    decreasing_data = wf_data if in_place else wf_data.copy()

    # loop to enforce data decrease from parent shells
    for c, parents_rel in radial_parents[maxchan]:
        if decreasing_data[c] > decreasing_data[parents_rel].max():
            decreasing_data[c] *= decreasing_data[parents_rel].max() / decreasing_data[c]

    return decreasing_data


def get_grid_convolution_templates_and_weights(
    contact_locations, local_radius_um=50, upsampling_um=5, sigma_um=np.linspace(10, 50.0, 5), margin_um=50
):
    x_min, x_max = contact_locations[:, 0].min(), contact_locations[:, 0].max()
    y_min, y_max = contact_locations[:, 1].min(), contact_locations[:, 1].max()

    x_min -= margin_um
    x_max += margin_um
    y_min -= margin_um
    y_max += margin_um

    dx = np.abs(x_max - x_min)
    dy = np.abs(y_max - y_min)

    eps = upsampling_um / 10

    all_x, all_y = np.meshgrid(
        np.arange(x_min, x_max + eps, upsampling_um), np.arange(y_min, y_max + eps, upsampling_um)
    )

    nb_templates = all_x.size

    template_positions = np.zeros((nb_templates, 2))
    template_positions[:, 0] = all_x.flatten()
    template_positions[:, 1] = all_y.flatten()

    import sklearn

    # mask to get nearest template given a channel
    dist = sklearn.metrics.pairwise_distances(contact_locations, template_positions)
    nearest_template_mask = dist < local_radius_um

    weights = np.zeros((len(sigma_um), len(contact_locations), nb_templates), dtype=np.float32)
    for count, sigma in enumerate(sigma_um):
        weights[count] = np.exp(-(dist**2) / (2 * (sigma**2)))

    # normalize
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.sqrt(np.sum(weights**2, axis=1))[:, np.newaxis, :]
        weights /= norm
        weights[np.isnan(weights)] = 0.0

    return template_positions, weights, nearest_template_mask


if HAVE_NUMBA:
    enforce_decrease_shells = numba.jit(enforce_decrease_shells_data, nopython=True)
