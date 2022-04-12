import numpy as np
from tqdm import tqdm

possible_motion_estimation_methods = ['decentralized_registration', ]


def init_kwargs_dict(method, method_kwargs):
    # handle kwargs by method
    if method == 'decentralized_registration':
        method_kwargs_ = dict(pairwise_displacement_method='conv2d', convergence_method='gradient_descent') # , maximum_displacement_um=400
    method_kwargs_.update(method_kwargs)
    return method_kwargs_


def estimate_motion(recording, peaks, peak_locations=None,
                    direction='y', bin_duration_s=10., bin_um=10., margin_um=50,
                    method='decentralized_registration', method_kwargs={},
                    non_rigid_kwargs=None, output_extra_check=False, progress_bar=False,
                    verbose=False):
    """
    Estimate motion given peaks and their localization.

    Location of peaks can be be included in peaks or given separately in 'peak_locations' argument.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    peaks: numpy array
        Peak vector (complex dtype)
        It can also contain the x/y/z fields
    peak_locations: numpy array
        If not already contained in 'peaks', the x/y/z field of spike location
    direction: 'x', 'y', 'z'
        Dimension on which the motion is estimated
    bin_duration_s: float
        Bin duration in second
    bin_um: float
        Spatial bin size in micro meter
    margin_um: float
        Margin in um to exclude from histogram estimation and
        non-rigid smoothing functions to avoid edge effects
    method: str
        The method to be used ('decentralized_registration')
    method_kwargs: dict
        Specific options for the chosen method.
        * 'decentralized_registration'
    non_rigid_kwargs: None or dict.
        If None then the motion is consider as rigid.
        If dict then the motion is estimated in non rigid manner with fields:
        * bin_step_um: step in um to construct overlapping gaussian smoothing functions
    output_extra_check: bool
        If True then return an extra dict that contains variables
        to check intermediate steps (motion_histogram, non_rigid_windows, pairwise_displacement)
    progress_bar: bool
        Display progress bar or not.
    verbose: bool
        If True, output is verbose

    Returns
    -------
    motion: numpy array 2d
        Motion estimate in um.
        Shape (temporal bins, spatial bins)
        motion.shape[0] = temporal_bins.shape[0]
        motion.shape[1] = 1 (rigid) or spatial_bins.shape[1] (non rigid)
    temporal_bins: numpy.array 1d
        temporal bins (bin center)
    spatial_bins: numpy.array 1d or None
        If rigid then None
        else motion.shape[1]
    extra_check: dict
        Optional output if `output_extra_check=True`
        This dict contain histogram, pairwise_displacement usefull for ploting.
    """
    # TODO handle multi segment one day
    assert recording.get_num_segments() == 1

    assert method in possible_motion_estimation_methods
    method_kwargs = init_kwargs_dict(method, method_kwargs)

    if output_extra_check:
        extra_check = {}

    if method == 'decentralized_registration':
        # make 2D histogram raster
        if verbose:
            print('Computing motion histogram')
        motion_histogram, temporal_hist_bins, spatial_hist_bins = make_motion_histogram(recording, peaks,
                                                                                        peak_locations=peak_locations,
                                                                                        bin_duration_s=bin_duration_s,
                                                                                        bin_um=bin_um,
                                                                                        margin_um=margin_um)
        if output_extra_check:
            extra_check['motion_histogram'] = motion_histogram
            extra_check['temporal_hist_bins'] = temporal_hist_bins
            extra_check['spatial_hist_bins'] = spatial_hist_bins
        # temporal bins are bin center
        temporal_bins = temporal_hist_bins[:-1] + bin_duration_s // 2.

        # rigid or non rigid is handled with a family of gaussian non_rigid_windows
        non_rigid_windows = []
        if non_rigid_kwargs is None:
            # one unique block for all depth
            non_rigid_windows = [np.ones(motion_histogram.shape[1], dtype='float64')]
            spatial_bins = None
        else:
            assert 'bin_step_um' in non_rigid_kwargs, "'non_rigid_kwargs' needs to specify the 'bin_step_um' field"
            probe = recording.get_probe()
            dim = ['x', 'y', 'z'].index(direction)
            contact_pos = probe.contact_positions[:, dim]

            bin_step_um = non_rigid_kwargs['bin_step_um']
            sigma_um = non_rigid_kwargs.get('sigma', 3) * bin_step_um
            min_ = np.min(contact_pos) - margin_um
            max_ = np.max(contact_pos) + margin_um

            num_win = int(np.ceil((max_ - min_) / bin_step_um))
            spatial_bins = np.arange(num_win) * bin_step_um + bin_step_um / 2. + min_

            # TODO check this gaussian with julien
            for win_center in spatial_bins:
                win = np.exp(-(spatial_hist_bins[:-1] - win_center) ** 2 / (sigma_um ** 2))
                non_rigid_windows.append(win)

            if output_extra_check:
                extra_check['non_rigid_windows'] = non_rigid_windows

        if output_extra_check:
            extra_check['pairwise_displacement_list'] = []

        motion = []
        for i, win in enumerate(non_rigid_windows):
            motion_hist = win[np.newaxis, :] * motion_histogram
            if verbose:
                print(f'Computing pairwise displacement: {i + 1} / {len(non_rigid_windows)}')

            pairwise_displacement, pairwise_displacement_weight = compute_pairwise_displacement(motion_hist, bin_um,
                                                                  method=method_kwargs['pairwise_displacement_method'],
                                                                  progress_bar=progress_bar)
            if output_extra_check:
                extra_check['pairwise_displacement_list'].append(pairwise_displacement)

            if verbose:
                print(f'Computing global displacement: {i + 1} / {len(non_rigid_windows)}')

            one_motion = compute_global_displacement(pairwise_displacement,
                        pairwise_displacement_weight=pairwise_displacement_weight,
                        convergence_method=method_kwargs['convergence_method']
                        )
            motion.append(one_motion[:, np.newaxis])

        motion = np.concatenate(motion, axis=1)

    # replace nan by zeros
    motion[np.isnan(motion)] = 0

    if output_extra_check:
        return motion, temporal_bins, spatial_bins, extra_check
    else:
        return motion, temporal_bins, spatial_bins


def get_location_from_fields(peaks_or_locations):
    dims = [dim for dim in ('x', 'y', 'z') if dim in peaks_or_locations.dtype.fields]
    peak_locations = np.zeros((peaks_or_locations.size, len(dims)), dtype='float64')
    for i, dim in enumerate(dims):
        peak_locations[:, i] = peaks_or_locations[dim]
    return peak_locations


def make_motion_histogram(recording, peaks, peak_locations=None,
                          weight_with_amplitude=False, direction='y',
                          bin_duration_s=1., bin_um=2., margin_um=50):
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


def compute_pairwise_displacement(motion_hist, bin_um, method='conv2d',
                                  weight_mode='exp', error_sigma = 0.2,
                                  conv_engine='numpy',
                                  progress_bar=False): 
    """
    Compute pairwise displacement
    """
    size = motion_hist.shape[0]
    pairwise_displacement = np.zeros((size, size), dtype='float32')
    
    if conv_engine == 'torch':
        import torch

    if method == 'conv2d':
        n = motion_hist.shape[1] // 2
        possible_displacement = np.arange(motion_hist.shape[1]) * bin_um
        possible_displacement -= possible_displacement[n]
        conv_values = np.zeros((size, size), dtype='float32')
        
        # TODO find something faster
        loop = range(size)
        if progress_bar:
            loop = tqdm(loop)
        if conv_engine == 'numpy':
            for i in loop:
                for j in range(size):
                    conv = np.convolve(motion_hist[i, :], motion_hist[j, ::-1], mode='same')
                    ind_max = np.argmax(conv)
                    pairwise_displacement[i, j] = possible_displacement[ind_max]
                    # norm = np.linalg.norm(conv)  ## this is wring Erdem!
                    norm = np.sqrt(np.sum(motion_hist[i, :]**2) * np.sum(motion_hist[j, :]**2))
                    conv_values[i, j] = conv[ind_max]  / norm
                    if (ind_max == 0) or (ind_max == possible_displacement.size -1):
                        conv_values[i, j] = -1
        elif conv_engine == 'torch':
            # TODO clip to -max_disp + max_disp
            # possible_displacement = np.arange(-disp, disp + step_size, step_size)
            motion_hist_torch = torch.from_numpy(motion_hist[:, np.newaxis, :]).cuda().float()
            c2d = torch.nn.Conv2d(in_channels=1, out_channels=size,
                                    kernel_size=[1, motion_hist_torch.shape[-1]],
                                    stride=1, 
                                    padding=[0, possible_displacement.size//2],
                                    bias=False).cuda()
            c2d.weight[:, 0] = motion_hist_torch
            for i in loop:
                res = c2d(motion_hist_torch[i:i+1, None])[:,:,0,:].argmax(2)
                pairwise_displacement[i:i+1] = possible_displacement[res.cpu()]
                del res
            del c2d
            del motion_hist_torch
            torch.cuda.empty_cache()

        # put in range 0-1
        errors = - conv_values * 0.5 + 0.5

    elif method == 'phase_cross_correlation':
        try:
            import skimage.registration
        except ImportError:
            raise ImportError("To use 'phase_cross_correlation' method install scikit-image")
        
        errors = np.zeros((size, size), dtype='float32')
        loop = range(size)
        if progress_bar:
            loop = tqdm(loop)
        for i in loop:
            for j in range(size):
                shift, error, diffphase = skimage.registration.phase_cross_correlation(motion_hist[i, :], 
                                                                                       motion_hist[j, :])
                pairwise_displacement[i, j] = shift * bin_um 
                errors[i, j] = error
        
        # print(errors.min(), errors.max())
        # pairwise_displacement_weight = np.exp(-((errors - errors.min()) / (errors.max() - errors.min()))/ error_sigma )
        

    else:
        raise ValueError(f'method do not exists for compute_pairwise_displacement {method}')

    if weight_mode == 'linear':
        # between 0 and 1
        pairwise_displacement_weight = 1 - errors
    elif weight_mode == 'exp':
        pairwise_displacement_weight = np.exp(- errors / error_sigma )

    return pairwise_displacement, pairwise_displacement_weight


def compute_global_displacement(pairwise_displacement, pairwise_displacement_weight=None, convergence_method='gradient_descent', max_iter=1000):
    """
    Compute global displacement
    
    This come from
    https://github.com/int-brain-lab/spikes_localization_registration/blob/main/registration_pipeline/image_based_motion_estimate.py#L211
    
    """
    from scipy.spatial.distance import pdist, squareform
    size = pairwise_displacement.shape[0]

    if convergence_method == 'gradient_descent':
        
        displacement = np.zeros(size, dtype='float64')

        # use variable name from paper
        # DECENTRALIZED MOTION INFERENCE AND REGISTRATION OF NEUROPIXEL DATA
        # Erdem Varol1, Julien Boussard, Hyun Dong Lee

        # time_sigma = 20
        # W2 = np.exp(-squareform(pdist(np.arange(size)[:,None]))/time_sigma)

        D = pairwise_displacement
        p = displacement
        p_prev = p.copy()
        for i in range(max_iter):
            # print(i)
            repeat1 = np.tile(p[:, np.newaxis], [1, size])
            repeat2 = np.tile(p[np.newaxis, :], [size, 1])
            mat_norm = D + (repeat1 - repeat2)
            # mat_norm = mat_norm * W2
            p += 2 * (np.sum(D - np.diag(D) , axis=1) - (size - 1) * p) / np.linalg.norm(mat_norm)
            # p += 2 * (np.sum(D * W2 - np.diag(D * W2) , axis=1) - (size - 1) * p) / np.linalg.norm(mat_norm)
            if np.allclose(p_prev, p):
                break
            else:
                p_prev = p.copy()

    elif convergence_method == 'lsqr_robust':
        from scipy.spatial.distance import pdist, squareform
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import lsqr
        from scipy.stats import zscore
        
        # TODO expose this in signature
        error_sigma = 0.1
        time_sigma = 45
        robust_regression_sigma = 1
        # n_iter = 20
        n_iter = 1

        assert pairwise_displacement_weight is not None
        S = np.ones(pairwise_displacement.shape, dtype='bool')
        # error_mat_S = pairwise_displacement_error[np.where(S != 0)]
        # W1 = np.exp(-((error_mat_S-error_mat_S.min())/(error_mat_S.max()-error_mat_S.min()))/error_sigma)
        # W1 = pairwise_displacement_weight
        # W2 = np.exp(-squareform(pdist(np.arange(size)[:,None]))/time_sigma)
        # W2 = W2[np.where(S != 0)]
        # W = (W2*W1)[:,None]
        # W = W1[:,None]
        # W = W2[:,None]
        # W = np.ones((size, size)).flatten()[:, None]
        W = pairwise_displacement_weight[np.where(S != 0)][:,None]
        



        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(W1, color='g')
        # ax.plot(W2, color='r')
        # fig, ax = plt.subplots()
        # ax.plot(W[:, 0])


        I, J = np.where(S != 0)
        V = pairwise_displacement[np.where(S != 0)]
        M = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]),I)))
        N = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]),J)))
        A = M - N
        idx = np.ones(A.shape[0]).astype(bool)
        # fig, ax = plt.subplots()
        for i in range(n_iter):
            p = lsqr(A[idx].multiply(W[idx]), V[idx]*W[idx][:,0])[0]
            # ax.plot(p)
            idx = np.where(np.abs(zscore(A@p-V)) <= robust_regression_sigma)
        displacement = p

    else:
        raise ValueError(f"Method {method} doesn't exists for compute_global_displacement")

    return displacement
