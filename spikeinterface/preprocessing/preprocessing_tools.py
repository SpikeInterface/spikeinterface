import numpy as np
import scipy.spatial


def get_spatial_interpolation_kernel(source_location, target_location, method='kriging',
                                     sigma_um=20., p=1, num_closest=3, dtype='float32',
                                     force_extrapolate=False):
    """
    Compute the spatial kernel for linear spatial interpolation.
    
    This is used for interpolation of bad channels or to correct the drift
    by interpolating between contacts.

    For reference, here is a simple overview on spatial interpolation:
    https://www.aspexit.com/spatial-data-interpolation-tin-idw-kriging-block-kriging-co-kriging-what-are-the-differences/

    Parameters
    ----------
    source_location: array shape (m, 2)
        The recording extractor to be transformed
    target_location: array shape (n, 2)
        Scale for the output distribution
    method: 'kriging' or 'idw' or 'nearest'
        Choice of the method
            'kriging' : the same one used in kilosort
            'idw' : inverse  distance weithed
            'nearest' : use nereast channel
    sigma_um : float (default 20.)
        Used in the 'kriging' formula
    p: int (default 1)
        Used in the 'kriging' formula
    num_closest: int (default 3)
        Used for 'idw'
    force_extrapolate: bool (false by default)
        How to handle when target location are outside source location.
        When False :  no extrapolation all target location outside are set to zero.
        When True : extrapolation done with the formula of the method.
                    In that case the sum of the kernel is not force to be 1.

    Returns
    -------
    interpolation_kernel: array (m, n)
    """

    target_is_inside = np.ones(target_location.shape[0], dtype=bool)
    for dim in range(source_location.shape[1]):
        l0, l1 = np.min(source_location[:, dim]), np.max(source_location[:, dim])
        target_is_inside &= (target_location[:, dim] >= l0) & (target_location[:, dim] <= l1)
    
    
    if method == 'kriging':
        # this is an adaptation of the pykilosort implementation by Kush Benga
        # https://github.com/int-brain-lab/pykilosort/blob/ibl_prod/pykilosort/datashift2.py#L352

        Kxx = get_kriging_kernel_distance(source_location, source_location, sigma_um, p)
        Kyx = get_kriging_kernel_distance(target_location, source_location, sigma_um, p)

        interpolation_kernel = Kyx @ np.linalg.pinv(Kxx + 0.01 * np.eye(Kxx.shape[0]))
        interpolation_kernel = interpolation_kernel.T.copy()

        # sparsify
        interpolation_kernel[interpolation_kernel < 0.001] = 0.

        # ensure sum = 1 for target inside
        s = np.sum(interpolation_kernel, axis=0)
        interpolation_kernel[:, target_is_inside] /= s[target_is_inside].reshape(1, -1)

    elif method == 'idw':
        distances = scipy.spatial.distance.cdist(source_location, target_location, metric='euclidean')
        interpolation_kernel = np.zeros((source_location.shape[0], target_location.shape[0]), dtype='float64')
        for c in range(target_location.shape[0]):
            ind_sorted = np.argsort(distances[:, c])
            chan_closest = ind_sorted[:num_closest]
            dists = distances[chan_closest, c]
            if dists[0] == 0.:
                # no interpolation the first have zeros distance
                interpolation_kernel[chan_closest[0], c] = 1.
            else:
                interpolation_kernel[chan_closest, c] = 1 / dists

        # ensure sum = 1 for target inside
        s = np.sum(interpolation_kernel, axis=0)
        interpolation_kernel[:, target_is_inside] /= s[target_is_inside].reshape(1, -1)


    elif method == 'nearest':
        distances = scipy.spatial.distance.cdist(source_location, target_location, metric='euclidean')
        interpolation_kernel = np.zeros((source_location.shape[0], target_location.shape[0]), dtype='float64')
        for c in range(target_location.shape[0]):
            ind_closest = np.argmin(distances[:, c])
            interpolation_kernel[ind_closest, c] = 1.

    else:
        raise ValueError('get_interpolation_kernel wrong method')
    
    if not force_extrapolate:
        interpolation_kernel[:, ~target_is_inside] = 0

    return interpolation_kernel.astype(dtype)


def get_kriging_kernel_distance(locations_1, locations_2, sigma_um, p):
    """
    Get the kriging kernel between two sets of locations.

    Parameters
    ----------

    locations_1 / locations_2 : 2D np.array
        Locations of shape (N, D) where N is number of
        channels and d is spatial dimension (e.g. 2 for [x, y])
    sigma_um : float
        Scale paremter on  the Gaussian kernel,
        typically distance between contacts in micrometers.
    p : float
        Weight parameter on the exponential function. Default
        in IBL kriging interpolation is 1.3.

    Results
    ----------
    kernal_dist : n x m array (i.e. locations 1 x locations 2) of
                  distances (gaussian kernel) between locations 1 and 2.

    """
    dist = scipy.spatial.distance.cdist(locations_1, locations_2, metric='euclidean')
    kernal_dist = np.exp(-(dist / sigma_um) ** p)
    return kernal_dist


def get_kriging_channel_weights(contact_positions1, contact_positions2, sigma_um, p,
                                weight_threshold=0.005):
    """
    Calculate weights for kriging interpolation. Weights below weight_threshold are set to 0.

    Based on the interpolate_bad_channels() function of the International Brain Laboratory.

    International Brain Laboratory et al. (2022). Spike sorting pipeline for the
    International Brain Laboratory. https://www.internationalbrainlab.com/repro-ephys
    """
    weights = get_kriging_kernel_distance(contact_positions1,
                                          contact_positions2,
                                          sigma_um,
                                          p)
    weights[weights < weight_threshold] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        weights /= np.sum(weights, axis=0)[None, :]

    weights[np.logical_or(weights < weight_threshold,
                          np.isnan(weights))] = 0

    return weights
