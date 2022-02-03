import numpy as np
from tqdm import tqdm

possible_motion_estimation_methods = ['decentralized_registration', ]


def init_kwargs_dict(method, method_kwargs):
    # handle kwargs by method
    if method == 'decentralized_registration':
        method_kwargs_ = dict(pairwise_displacement_method='conv2d') # , maximum_displacement_um=400
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
            min_ = np.min(contact_pos) - margin_um
            max_ = np.max(contact_pos) + margin_um

            num_win = int(np.ceil((max_ - min_) / bin_step_um))
            spatial_bins = np.arange(num_win) * bin_step_um + bin_step_um / 2. + min_

            # TODO check this gaussian with julien
            for win_center in spatial_bins:
                sigma = bin_step_um
                win = np.exp(-(spatial_hist_bins[:-1] - win_center) ** 2 / (2 * sigma ** 2))
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

            pairwise_displacement = compute_pairwise_displacement(motion_hist, bin_um,
                                                                  method=method_kwargs['pairwise_displacement_method'],
                                                                  progress_bar=progress_bar)
            if output_extra_check:
                extra_check['pairwise_displacement_list'].append(pairwise_displacement)

            if verbose:
                print(f'Computing global displacement: {i + 1} / {len(non_rigid_windows)}')

            one_motion = compute_global_displacement(pairwise_displacement)
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


def compute_pairwise_displacement(motion_hist, bin_um, method='conv2d', progress_bar=False): 
    """
    Compute pairwise displacement
    """
    size = motion_hist.shape[0]
    pairwise_displacement = np.zeros((size, size), dtype='float32')

    if method == 'conv2d':
        n = motion_hist.shape[1] // 2
        possible_displacement = np.arange(motion_hist.shape[1]) * bin_um
        possible_displacement -= possible_displacement[n]

        # TODO find something faster
        loop = range(size)
        if progress_bar:
            loop = tqdm(loop)
        for i in loop:
            # print(i, size)
            for j in range(size):
                conv = np.convolve(motion_hist[i, :], motion_hist[j, ::-1], mode='same')
                ind_max = np.argmax(conv)
                pairwise_displacement[i, j] = possible_displacement[ind_max]
    elif method == 'phase_cross_correlation':
        try:
            import skimage.registration
        except ImportError:
            raise ImportError("To use 'phase_cross_correlation' method install scikit-image")

        for i in range(size):
            for j in range(size):
                shift, error, diffphase = skimage.registration.phase_cross_correlation(motion_hist[i, :], 
                                                                                       motion_hist[j, :])
                pairwise_displacement[i, j] = shift
    else:
        raise ValueError(f'method do not exists for compute_pairwise_displacement {method}')

    return pairwise_displacement


def compute_global_displacement(pairwise_displacement, method='gradient_descent', max_iter=1000):
    """
    Compute global displacement
    """
    if method == 'gradient_descent':
        size = pairwise_displacement.shape[0]
        displacement = np.zeros(size, dtype='float64')

        # use variable name from paper
        # DECENTRALIZED MOTION INFERENCE AND REGISTRATION OF NEUROPIXEL DATA
        # Erdem Varol1, Julien Boussard, Hyun Dong Lee
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

    elif method == 'robust':
        raise NotImplementedError
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
        # return p
    else:
        raise ValueError(f"Method {method} doesn't exists for compute_global_displacement")

    return displacement
