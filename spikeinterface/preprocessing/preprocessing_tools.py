import numpy as np
import scipy.spatial



def get_spatial_interpolation_kernel(source_location, target_location, method='kriging',
                                     sigma_um=20., p=1, num_closest=3, ):
    """
    Compute the spatial kernel for linear spatial interpolation.
    
    This is used for interpolation of bad channel or to correct the drift
    by interpolating inbetween contact.

    here asimple overview on spatial interpolation:
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
            '' : 
    sigma_um : float (default 20.)
        Used in the krigging formula
    p: int (default 1)
        Used in the krigging formula
    num_closest: int (default 3)
        Used for 'idw'

    Returns
    -------
    interpolation_kernel: array (m, n)
    """


    if method == 'kriging':
        # this is an adaptation of  pykilosort implementation by Kush Benga
        # https://github.com/int-brain-lab/pykilosort/blob/ibl_prod/pykilosort/datashift2.py#L352
        dist_xx = scipy.spatial.distance.cdist(source_location, source_location, metric='euclidean')
        Kxx = np.exp(-(dist_xx / sigma_um) **p)

        dist_yx = scipy.spatial.distance.cdist(target_location, source_location, metric='euclidean')
        Kyx = np.exp(-(dist_yx / sigma_um) **p)

        interpolation_kernel = Kyx @ np.linalg.pinv(Kxx + 0.01 * np.eye(Kxx.shape[0]))
        interpolation_kernel = interpolation_kernel.T.astype('float32').copy()

        # sparse and ensure sum = 1
        # interpolation_kernel[interpolation_kernel < 0.05] = 0
        # interpolation_kernel /= np.sum(interpolation_kernel, axis=1).reshape(-1, 1)

    elif method == 'idw':
        distances = scipy.spatial.distance.cdist(source_location, target_location, metric='euclidean')

        interpolation_kernel = np.zeros((source_location.shape[0], target_location.shape[0]), dtype='float32')
        for c in range(target_location.shape[0]):
            ind_sorted = np.argsort(distances[c, :])
            chan_closest = ind_sorted[:num_closest]
            dists = distances[c, chan_closest]
            if dists[0] == 0.:
                # no interpolation the first have zeros distance
                interpolation_kernel[chan_closest[0], c] = 1.
            else:
                w = 1 / dists
                w /= np.sum(w)
                interpolation_kernel[chan_closest, c] = w
    elif method == 'nearest':
        raise NotImplementedError('get_spatial_interpolation_kernel nearest will be done soon')

    else:
        raise ValueError('get_interpolation_kernel wrong method')

    return interpolation_kernel
