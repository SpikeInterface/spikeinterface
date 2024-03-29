from __future__ import annotations

import warnings

import numpy as np

try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

from ..core.template_tools import get_template_extremum_channel, _get_nbefore, _get_dense_templates_array


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
    contact_locations, radius_um=40, upsampling_um=5, margin_um=50, weight_method={"mode": "exponential_3d"}
):
    """Get a upsampled grid of artificial templates given a particular probe layout

    Parameters
    ----------
    contact_locations: array
        The positions of the channels
    radius_um: float
        Radius in um for channel sparsity.
    upsampling_um: float
        Upsampling resolution for the grid of templates
    margin_um: float
        The margin for the grid of fake templates
    weight_method: dict
        Parameter that should be provided to the get_convolution_weights() function
        in order to know how to estimate the positions. One argument is mode that could
        be either gaussian_2d (KS like) or exponential_3d (default)

    Returns
    -------
    template_positions: array
        The positions of the upsampled templates
    weights:
        The weights of the templates, on a per channel basis
    nearest_template_mask: array
        A sparsity mask to to know which template is close to the contact locations, given
        the radius_um parameter
    z_factors: array
        The z_factors that have been used to generate the weights along the third dimension
    """

    import sklearn.metrics

    x_min, x_max = contact_locations[:, 0].min(), contact_locations[:, 0].max()
    y_min, y_max = contact_locations[:, 1].min(), contact_locations[:, 1].max()

    x_min -= margin_um
    x_max += margin_um
    y_min -= margin_um
    y_max += margin_um

    eps = upsampling_um / 10

    all_x, all_y = np.meshgrid(
        np.arange(x_min, x_max + eps, upsampling_um), np.arange(y_min, y_max + eps, upsampling_um)
    )

    nb_templates = all_x.size

    template_positions = np.zeros((nb_templates, 2))
    template_positions[:, 0] = all_x.flatten()
    template_positions[:, 1] = all_y.flatten()

    # mask to get nearest template given a channel
    dist = sklearn.metrics.pairwise_distances(contact_locations, template_positions)
    nearest_template_mask = dist <= radius_um
    weights, z_factors = get_convolution_weights(dist, **weight_method)

    return template_positions, weights, nearest_template_mask, z_factors


def get_convolution_weights(
    distances,
    z_list_um=np.linspace(0, 120.0, 5),
    sigma_list_um=np.linspace(5, 25, 5),
    sparsity_threshold=None,
    sigma_3d=2.5,
    mode="exponential_3d",
):
    """Get normalized weights for creating artificial templates, given some precomputed distances

    Parameters
    ----------
    distances: 2D array
        The distances between the source channels (real ones) and the upsampled one (virual ones)
    sparsity_threshold: float, default None
        The sparsity_threshold below which weights are set to 0 (speeding up computations). If None,
        then a default value of 0.5/sqrt(distances.shape[0]) is set
    mode: exponential_3d | gaussian_2d
        The inference scheme to be used to get the convolution weights
        Keyword arguments for the chosen method:
            "gaussian_2d" (similar to KiloSort):
                * sigma_list_um: array, default np.linspace(5, 25, 5)
                    The list of sigma to consider for decaying exponentials
            "exponential_3d" (default):
                * z_list_um: array, default np.linspace(0, 120.0, 5)
                    The list of z to consider for putative depth of the sources
                * sigma_3d: float, default 2.5
                    The scaling factor controling the decay of the exponential

    Returns
    -------
    weights:
        The weights of the templates, on a per channel basis
    z_factors: array
        The z_factors that have been used to generate the weights along the third dimension
    """

    if sparsity_threshold is not None:
        assert 0 <= sparsity_threshold <= 1, "sparsity_threshold should be in [0, 1]"

    if mode == "exponential_3d":
        weights = np.zeros((len(z_list_um), distances.shape[0], distances.shape[1]), dtype=np.float32)
        for count, z in enumerate(z_list_um):
            dist_3d = np.sqrt(distances**2 + z**2)
            weights[count] = np.exp(-dist_3d / sigma_3d)
        z_factors = z_list_um
    elif mode == "gaussian_2d":
        weights = np.zeros((len(sigma_list_um), distances.shape[0], distances.shape[1]), dtype=np.float32)
        for count, sigma in enumerate(sigma_list_um):
            alpha = 2 * (sigma**2)
            weights[count] = np.exp(-(distances**2) / alpha)
        z_factors = sigma_list_um

    # normalize to get normalized values in [0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.linalg.norm(weights, axis=1)[:, np.newaxis, :]
        weights /= norm

    weights[~np.isfinite(weights)] = 0.0

    # If sparsity is None or non zero, we are pruning weights that are below the
    # sparsification factor. This will speed up furter computations
    if sparsity_threshold is None:
        sparsity_threshold = 0.5 / np.sqrt(distances.shape[0])
    weights[weights < sparsity_threshold] = 0

    # re normalize to ensure we have unitary norms
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.linalg.norm(weights, axis=1)[:, np.newaxis, :]
        weights /= norm

    weights[~np.isfinite(weights)] = 0.0

    return weights, z_factors


if HAVE_NUMBA:
    enforce_decrease_shells = numba.jit(enforce_decrease_shells_data, nopython=True)
