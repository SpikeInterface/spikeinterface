import numpy as np
import scipy.signal


possible_motion_estimation_methods = ['decentralized_registration', ]


def init_kwargs_dict(method, method_kwargs):
    # handle kwargs method by method
    if method == 'decentralized_registration':
        method_kwargs_ = dict(pairwise_displacement_method='conv2d') # , maximum_displacement_um=400
    method_kwargs_.update(method_kwargs)
    return method_kwargs_



    
    

def estimate_motion(recording, peaks, peak_locations=None,
                    direction='y', bin_duration_s=1., bin_um=2.,
                    method='decentralized', method_kwargs={},
                    non_rigid_kwargs=None,):
    """
    Estimation motion given peaks and threre localization.
    
    Parameters
    ----------
    recording: RecordingExtractor
        The recording.
    peaks: numpy array
        Peak vector (complex dtype)
        Can also contain the x/y/z fields
    peak_locations: numpy array
        If not in peaks already contain the x/y/z field of spike location
    direction: 'x', 'y', 'z'
        Dimension on which the motion is estiomated
    bin_duration_s: float
        Bin duration in second
    bin_um: float
        Spatial bin size in micro meter
    method: 
        'decentralized_registration'
    method_kwargs: dict



    non_rigid_kwargs: None or dict.
        If None then the motion is consider as rigid.
        If dict then the motion is estimated in non rigid manner.

    Returns
    -------
    motion: numpy array
    """
    # TODO handle multi segment one day
    assert recording.get_num_segments() == 1

    assert method in possible_motion_estimation_methods
    method_kwargs = init_kwargs_dict(method, method_kwargs)
    
    
    if peak_locations is None:
        peak_locations = get_location_from_fields(peaks)
    else:
        peak_locations = get_location_from_fields(peak_locations)

    

    if method =='decentralized_registration':
        # make 2D histogram raster
        motion_histogram, temporal_bins, spatial_bins = make_motion_histogram(recording, peaks,
                                    peak_locations=peak_locations, bin_duration_s=1., bin_um=2.)

        # rigid or non rigid is handle with a family of gaussian windows
        windows = []
        if non_rigid_kwargs is None:
            # one unique block
            windows = [np.ones(spatial_bins.size, dtype='float64')]
        else:
            # todo make gaussian block for non rigid 
            raise NotImplementedError
        
        
        motion_per_block = []
        for i, win in enumerate(windows):
            print(i)
            motion_hist = win[np.newaxis, :] * motion_histogram
            pairwise_displacement = compute_pairwise_displacement(motion_hist,
                                            method=method_kwargs['pairwise_displacement_method'],
                                            maximum_displacement_um=method_kwargs['maximum_displacement_um'])

            motion = compute_global_displacement(pairwise_displacement)
            motion_per_block.append(motion)




    
    return motion


def get_location_from_fields(peaks_or_locations):
    dims = [dim for dim in ('x', 'y', 'z') if dim in peaks_or_locations.dtype.fields]
    peak_locations = np.zeros((peaks_or_locations.size, len(dims)), dtype='float64')
    for i, dim in enumerate(dims):
        peak_locations[:, i] = peaks_or_locations[dim]
    return peak_locations


def make_motion_histogram(recording, peaks, peak_locations=None,
        weight_with_amplitude=False, direction='y', bin_duration_s=1., bin_um=2., margin_um=50):
    """
    Generate motion histogram 
    
    """
    if peak_locations is None:
        peak_locations = get_location_from_fields(peaks)
    else:
        peak_locations = get_location_from_fields(peak_locations)
    
    fs = recording.get_sampling_frequency()
    num_sample = recording.get_num_samples(segment_index=0)
    bin = int(bin_duration_s * fs)
    sample_bins = np.arange(0, num_sample+bin, bin)
    temporal_bins = sample_bins / fs
    
    # contact along one axis
    probe = recording.get_probe()
    dim = ['x', 'y', 'z'].index(direction)
    contact_pos = probe.contact_positions[:, dim]
    
    
    min_ = np.min(contact_pos) - margin_um
    max_ = np.max(contact_pos) + margin_um
    spatial_bins = np.arange(min_, max_+bin_um, bin_um)

    arr = np.zeros((peaks.size, 2), dtype='float64')
    arr[:, 0] = peaks['sample_ind']
    arr[:, 1] = peak_locations[:, dim]
    
    if weight_with_amplitude:
        weights = np.abs(peaks['amplitude'])
    else:
        weights = None
    motion_histogram, edges = np.histogramdd(arr, bins=(sample_bins, spatial_bins), weights=weights)
    
    
    return motion_histogram, temporal_bins, spatial_bins


def compute_pairwise_displacement(motion_hist, bin_um, method='conv2d'): # maximum_displacement_um=400

    size = motion_hist.shape[0]
    pairwise_displacement = np.zeros((size, size), dtype='float32')

    if method =='conv2d':
        n = motion_hist.shape[1] // 2
        possible_displacement = np.arange(motion_hist.shape[1]) * bin_um
        possible_displacement -= possible_displacement[n]

        
        # todo find something faster
        for i in range(size):
            print(i, size)
            for j in range(size):
                conv = np.convolve(motion_hist[i, :], motion_hist[j, ::-1], mode='same')
                ind_max = np.argmax(conv)
                pairwise_displacement[i, j] = possible_displacement[ind_max]

        # for i in range(size):
        #     print(i, size)
        #     conv = scipy.ndimage.convolve1d(motion_hist, motion_hist[i, ::-1], axis=1)
        #     ind_max = np.argmax(conv, axis=1)
        #     pairwise_displacement[i, :] = possible_displacement[ind_max]


    elif method == 'phase_cross_correlation':
        import skimage.registration

        for i in range(size):
            print(i, size)
            for j in range(size):
                shift, error, diffphase = skimage.registration.phase_cross_correlation(motion_hist[i, :], motion_hist[j, :])
                pairwise_displacement[i, j] = shift

    else:
        raise ValueError(f'method do not exists for compute_pairwise_displacement {method}')


    return pairwise_displacement


def compute_global_displacement(pairwise_displacement, method='??', max_iter=1000):
    """

    """



    if method == '??':
        size = pairwise_displacement.shape[0]
        displacement = np.zeros(size, dtype='float64')


        # use variable name from paper
        # DECENTRALIZED MOTION INFERENCE AND REGISTRATION OF NEUROPIXEL DATA
        # Erdem Varol1, Julien Boussard
        D = pairwise_displacement
        p = displacement


        p_prev = p.copy()
        for i in range(max_iter):
            # print(i)
            repeat1 = np.tile(p[:, np.newaxis], [1, size])
            repeat2 = np.tile(p[np.newaxis, :], [size, 1])
            mat_norm = D + repeat1 - repeat2
            p += 2 * (np.sum(D - np.diag(D), axis=1) - (size - 1) * p) / np.linalg.norm(mat_norm)
            if np.allclose(p_prev, p):
                break
            else:
                p_prev = p.copy()


    elif method =='robust':
        pass
        # error_mat_S = error_mat[np.where(S != 0)]
        # W1 = np.exp(-((error_mat_S-error_mat_S.min())/(error_mat_S.max()-error_mat_S.min()))/error_sigma)

        # W2 = np.exp(-squareform(pdist(np.arange(error_mat.shape[0])[:,None]))/time_sigma)
        # W2 = W2[np.where(S != 0)]

        # W = (W2*W1)[:,None]
        
        # I, J = np.where(S != 0)
        # V = displacement_matrix[np.where(S != 0)]
        # M = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]),I)))
        # N = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]),J)))
        # A = M - N
        # idx = np.ones(A.shape[0]).astype(bool)
        # for i in notebook.tqdm(range(n_iter)):
        #     p = lsqr(A[idx].multiply(W[idx]), V[idx]*W[idx][:,0])[0]
        #     idx = np.where(np.abs(zscore(A@p-V)) <= robust_regression_sigma)
        # return p

    else:
        raise ValueError(f'method do not exists for compute_global_displacement {method}')


    return displacement



