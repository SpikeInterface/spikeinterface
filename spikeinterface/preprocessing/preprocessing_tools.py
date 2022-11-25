import numpy as np
import scipy.spatial



def get_spatial_interpolation_kernel(source_location, target_location, method='kriging',
                                     sigma_um=20., p=1, num_closest=3, dtype='float32', force_extrapolate=False):
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
        dist_xx = scipy.spatial.distance.cdist(source_location, source_location, metric='euclidean')
        Kxx = np.exp(-(dist_xx / sigma_um) **p)

        dist_yx = scipy.spatial.distance.cdist(target_location, source_location, metric='euclidean')
        Kyx = np.exp(-(dist_yx / sigma_um) **p)

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
