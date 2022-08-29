import numpy as np
from tqdm.auto import tqdm, trange

possible_motion_estimation_methods = ['decentralized_registration', ]


def init_kwargs_dict(method, method_kwargs):
    # handle kwargs by method
    if method == 'decentralized_registration':
        method_kwargs_ = dict(pairwise_displacement_method='conv', convergence_method='gradient_descent', max_displacement_um=1500)
    method_kwargs_.update(method_kwargs)
    return method_kwargs_


def estimate_motion(recording, peaks, peak_locations,
                    direction='y', bin_duration_s=10., bin_um=10., margin_um=50,
                    method='decentralized_registration', method_kwargs={},
                    non_rigid_kwargs=None, output_extra_check=False, progress_bar=False,
                    upsample_to_histogram_bin=True, verbose=False):
    """
    Estimate motion given peaks and their localization.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    peaks: numpy array
        Peak vector (complex dtype)
    peak_locations: numpy array
        Complex dtype with 'x', 'y', 'z' fields
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
    upsample_to_histogram_bin: bool
        If True then upsample the returned motion array to the number of depth bins specified
        by bin_um
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
        If upsample_to_histogram_bin, motion.shape[1] corresponds to spatial
        bins given by bin_um.
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
                                                                                        peak_locations, direction=direction,
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

            num_win = (max_ - min_) // bin_step_um
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
        windows_iter = non_rigid_windows
        if progress_bar:
            windows_iter = tqdm(windows_iter, desc="windows")
        for i, win in enumerate(windows_iter):
            window_slice = np.flatnonzero(win > 1e-5)
            window_slice = slice(window_slice[0], window_slice[-1])
            motion_hist = win[np.newaxis, window_slice] * motion_histogram[:, window_slice]
            if verbose:
                print(f'Computing pairwise displacement: {i + 1} / {len(non_rigid_windows)}')

            pairwise_displacement, pairwise_displacement_weight = compute_pairwise_displacement(
                motion_hist, bin_um,
                method=method_kwargs['pairwise_displacement_method'],
                weight_scale=method_kwargs.get("weight_scale", 'linear'),
                error_sigma=method_kwargs.get("error_sigma", 0.2),
                conv_engine=method_kwargs.get("conv_engine", 'numpy'),
                torch_device=method_kwargs.get("torch_device", None),
                batch_size=method_kwargs.get("batch_size", 1),
                max_displacement_um=method_kwargs.get("max_displacement_um", 1500),
                corr_threshold=method_kwargs.get("corr_threshold", 0),
                time_horizon_s=method_kwargs.get("time_horizon_s", None),
                sampling_frequency=method_kwargs.get("sampling_frequency", None),
                progress_bar=False
            )
            if output_extra_check:
                extra_check['pairwise_displacement_list'].append(pairwise_displacement)

            if verbose:
                print(f'Computing global displacement: {i + 1} / {len(non_rigid_windows)}')

            one_motion = compute_global_displacement(
                pairwise_displacement,
                pairwise_displacement_weight=pairwise_displacement_weight,
                convergence_method=method_kwargs['convergence_method'],
                robust_regression_sigma=method_kwargs.get("robust_regression_sigma", 2),
                gradient_descent_max_iter=method_kwargs.get("gradient_descent_max_iter", 1000),
                lsqr_robust_n_iter=method_kwargs.get("lsqr_robust_n_iter", 20),
                progress_bar=False,
            )
            motion.append(one_motion[:, np.newaxis])

        motion = np.concatenate(motion, axis=1)

    # replace nan by zeros
    motion[np.isnan(motion)] = 0

    if upsample_to_histogram_bin:
        # do upsample
        non_rigid_windows = np.array(non_rigid_windows)
        non_rigid_windows /= non_rigid_windows.sum(axis=0, keepdims=True)
        spatial_bins = spatial_hist_bins[:-1] + bin_um / 2
        motion = motion @ non_rigid_windows

    if output_extra_check:
        return motion, temporal_bins, spatial_bins, extra_check
    else:
        return motion, temporal_bins, spatial_bins


def make_motion_histogram(recording, peaks, peak_locations,
                          weight_with_amplitude=False, direction='y',
                          bin_duration_s=1., bin_um=2., margin_um=50):
    """
    Generate motion histogram
    """

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
    arr[:, 1] = peak_locations[direction]

    if weight_with_amplitude:
        weights = np.abs(peaks['amplitude'])
    else:
        weights = None

    motion_histogram, edges = np.histogramdd(arr, bins=(sample_bins, spatial_bins), weights=weights)

    # average amplitude in each bin
    if weight_with_amplitude:
        bin_counts, _ = np.histogramdd(arr, bins=(sample_bins, spatial_bins))
        bin_counts[bin_counts == 0] = 1
        motion_histogram = motion_histogram / bin_counts

    return motion_histogram, temporal_bins, spatial_bins


def compute_pairwise_displacement(motion_hist, bin_um, method='conv',
                                  weight_scale='linear', error_sigma=0.2,
                                  conv_engine='numpy', torch_device=None,
                                  batch_size=1, max_displacement_um=1500,
                                  corr_threshold=0, time_horizon_s=None,
                                  sampling_frequency=None, progress_bar=False): 
    """
    Compute pairwise displacement
    """
    from scipy import sparse
    assert conv_engine in ("torch", "numpy")
    size = motion_hist.shape[0]
    pairwise_displacement = np.zeros((size, size), dtype='float32')

    if time_horizon_s is not None:
        band_width = int(np.ceil(time_horizon_s * sampling_frequency))
    
    if conv_engine == 'torch':
        import torch
        if torch_device is None:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if method == 'conv':
        if max_displacement_um is None:
            n = motion_hist.shape[1] // 2
        else:
            n = min(
                motion_hist.shape[1] // 2,
                int(np.ceil(max_displacement_um // bin_um)),
            )
        possible_displacement = np.arange(-n, n + 1) * bin_um

        conv_values = np.zeros((size, size), dtype='float32')
        xrange = trange if progress_bar else range

        motion_hist_engine = motion_hist
        if conv_engine == "torch":
            motion_hist_engine = torch.as_tensor(motion_hist, dtype=torch.float32, device=torch_device)

        if time_horizon_s is not None and time_horizon_s > 0:
            pairwise_displacement = sparse.dok_matrix((size, size), dtype=np.float32)
            correlation = sparse.dok_matrix((size, size), dtype=motion_hist.dtype)

            for i in xrange(size):
                hist_i = motion_hist_engine[None, i]
                pairwise_displacement[i, i] = 0
                correlation[i, i] = 1
                j_max = size if time_horizon_s is None else min(size, i + band_width)
                for j in range(i + 1, j_max):
                    corr = normxcorr1d(
                        hist_i,
                        motion_hist_engine[None, j],
                        padding=possible_displacement.size // 2,
                        conv_engine=conv_engine,
                    )
                    ind_max = np.argmax(corr)
                    max_corr = corr[0, 0, ind_max]
                    if max_corr > corr_threshold:
                        pairwise_displacement[i, j] = possible_displacement[ind_max]
                        pairwise_displacement[j, i] = -possible_displacement[ind_max]
                        correlation[i, j] = correlation[j, i] = max_corr

            pairwise_displacement = pairwise_displacement.tocsr()
            correlation = correlation.tocsr()

        else:
            pairwise_displacement = np.empty((size, size), dtype=np.float32)
            correlation = np.empty((size, size), dtype=motion_hist.dtype)

            for i in xrange(0, size, batch_size):
                corr = normxcorr1d(
                    motion_hist_engine,
                    motion_hist_engine[i : i + batch_size],
                    padding=possible_displacement.size // 2,
                    conv_engine=conv_engine,
                )
                if conv_engine == "torch":
                    max_corr, best_disp_inds = torch.max(corr, dim=2)
                    best_disp = possible_displacement[best_disp_inds.cpu()]
                    pairwise_displacement[i : i + batch_size] = best_disp
                    correlation[i : i + batch_size] = max_corr.cpu()
                elif conv_engine == "numpy":
                    best_disp_inds = np.argmax(corr, axis=2)
                    max_corr = np.take_along_axis(corr, best_disp_inds[..., None], 2).squeeze()
                    best_disp = possible_displacement[best_disp_inds]
                    pairwise_displacement[i : i + batch_size] = best_disp
                    correlation[i : i + batch_size] = max_corr

            if corr_threshold > 0:
                which = correlation > corr_threshold
                pairwise_displacement *= which
                correlation *= which

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
        correlation = 1 - errors
        
    else:
        raise ValueError(f'method do not exists for compute_pairwise_displacement {method}')

    if weight_scale == 'linear':
        # between 0 and 1
        pairwise_displacement_weight = correlation
    elif weight_scale == 'exp':
        pairwise_displacement_weight = np.exp((correlation - 1) / error_sigma )

    return pairwise_displacement, pairwise_displacement_weight


def compute_global_displacement(
    pairwise_displacement,
    pairwise_displacement_weight=None,
    sparse_mask=None,
    convergence_method='gradient_descent',
    robust_regression_sigma=2,
    gradient_descent_max_iter=1000,
    lsqr_robust_n_iter=20,
    progress_bar=False,
):
    """
    Compute global displacement

    Reference:
    DECENTRALIZED MOTION INFERENCE AND REGISTRATION OF NEUROPIXEL DATA
    Erdem Varol1, Julien Boussard, Hyun Dong Lee

    Improved during Spike Sorting Hackathon 2022 by Erdem Varol and Charlie Windolf.
    
    This come from
    https://github.com/int-brain-lab/spikes_localization_registration/blob/main/registration_pipeline/image_based_motion_estimate.py#L211
    
    """
    size = pairwise_displacement.shape[0]

    if convergence_method == 'gradient_descent':
        from scipy.optimize import minimize
        from scipy.sparse import csr_matrix

        D = pairwise_displacement
        if pairwise_displacement_weight is not None or sparse_mask is not None:
            # weighted problem
            if pairwise_displacement_weight is None:
                pairwise_displacement_weight = np.ones_like(D)
            if sparse_mask is None:
                sparse_mask = np.ones_like(D)
            W = pairwise_displacement_weight * sparse_mask

            I, J = np.where(W > 0)
            Wij = W[I, J]
            Dij = D[I, J]
            W = csr_matrix((Wij, (I, J)), shape=W.shape)
            WD = csr_matrix((Wij * Dij, (I, J)), shape=W.shape)
            fixed_terms = (W @ WD).diagonal() - (WD @ W).diagonal()
            diag_WW = (W @ W).diagonal()
            Wsq = W.power(2)

            def obj(p):
                return 0.5 * np.square(Wij * (Dij - (p[I] - p[J]))).sum()

            def jac(p):
                return fixed_terms - 2 * (Wsq @ p) + 2 * p * diag_WW
        else:
            # unweighted problem, it's faster when we have no weights
            fixed_terms = -D.sum(axis=1) + D.sum(axis=0)

            def obj(p):
                v = np.square((D - (p[:, None] - p[None, :]))).sum()
                return 0.5 * v

            def jac(p):
                return fixed_terms + 2 * (size * p - p.sum())

        res = minimize(
            fun=obj, jac=jac, x0=D.mean(axis=1), method="L-BFGS-B"
        )
        if not res.success:
            print("Global displacement gradient descent had an error")
        displacement = res.x

    elif convergence_method == 'lsqr_robust':
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import lsqr
        from scipy.stats import zscore

        if sparse_mask is not None:
            I, J = np.where(sparse_mask > 0)
        elif pairwise_displacement_weight is not None:
            I, J = np.where(pairwise_displacement_weight > 0)
        else:
            I, J = np.where(np.ones_like(pairwise_displacement, dtype=bool))

        nnz_ones = np.ones(I.shape[0], dtype=pairwise_displacement.dtype)
        if pairwise_displacement_weight is not None:
            W = pairwise_displacement_weight[I, J][:,None]
        else:
            W = nnz_ones[:, None]
        V = pairwise_displacement[I, J]
        M = csr_matrix((nnz_ones, (range(I.shape[0]), I)))
        N = csr_matrix((nnz_ones, (range(I.shape[0]), J)))
        A = M - N
        idx = np.ones(A.shape[0], dtype=bool)
        xrange = trange if progress_bar else range
        for i in xrange(lsqr_robust_n_iter):
            p = lsqr(A[idx].multiply(W[idx]), V[idx] * W[idx][:,0])[0]
            idx = np.where(np.abs(zscore(A @ p - V)) <= robust_regression_sigma)
        displacement = p

    else:
        raise ValueError(f"Method {method} doesn't exists for compute_global_displacement")

    return displacement


def normxcorr1d(template, x, padding="same", conv_engine="torch"):
    """normxcorr1d: 1-D normalized cross-correlation

    Returns the cross-correlation of `template` and `x` at spatial lags
    determined by `mode`. Useful for estimating the location of `template`
    within `x`.
    This might not be the most efficient implementation -- ideas welcome.
    It uses a direct convolutional translation of the formula
        corr = (E[XY] - EX EY) / sqrt(var X * var Y)

    Arguments
    ---------
    template : tensor, shape (num_templates, length)
        The reference template signal
    x : tensor, 1d shape (length,) or 2d shape (num_inputs, length)
        The signal in which to find `template`
    padding : int, optional
        How far to look? if unset, we'll use half the length
    assume_centered : bool
        Avoid a copy if your data is centered already.

    Returns
    -------
    corr : tensor
    """
    if conv_engine == "torch":
        import torch
        import torch.nn.functional as F
        conv1d = F.conv1d
        npx = torch
    elif conv_engine == "numpy":
        conv1d = scipy_conv1d
        npx = np
    else:
        raise ValueError(f"Unknown conv_engine {conv_engine}")

    x = npx.atleast_2d(x)
    num_templates, length = template.shape
    num_inputs, length_ = template.shape
    assert length == length_

    # compute expectations
    if conv_engine == "torch":
        ones = npx.ones((1, 1, length), dtype=x.dtype, device=x.device)
    else:
        ones = npx.ones((1, 1, length), dtype=x.dtype)
    # how many points in each window? seems necessary to normalize
    # for numerical stability.
    N = conv1d(ones, ones, padding=padding)
    Et = conv1d(ones, template[:, None, :], padding=padding) / N
    Ex = conv1d(x[:, None, :], ones, padding=padding) / N

    # compute covariance
    corr = conv1d(x[:, None, :], template[:, None, :], padding=padding) / N
    corr -= Ex * Et

    # compute variances for denominator, using var X = E[X^2] - (EX)^2
    var_template = conv1d(
        ones, npx.square(template)[:, None, :], padding=padding
    )
    var_template = var_template / N - npx.square(Et)
    var_x = conv1d(
        npx.square(x)[:, None, :], ones, padding=padding
    )
    var_x = var_x / N - npx.square(Ex)

    # now find the final normxcorr and get rid of NaNs in zero-variance areas
    corr /= npx.sqrt(var_x * var_template)
    corr[~npx.isfinite(corr)] = 0

    return corr


def scipy_conv1d(input, weights, padding="valid"):
    """SciPy translation of torch F.conv1d"""
    from scipy.signal import correlate

    n, c_in, length = input.shape
    c_out, in_by_groups, kernel_size = weights.shape
    assert in_by_groups == c_in == 1

    if padding == "same":
        mode = "same"
        length_out = length
    elif padding == "valid":
        mode = "valid"
        length_out = length - 2 * (kernel_size // 2)
    elif isinstance(padding, int):
        mode = "valid"
        input = np.pad(input, [*[(0,0)] * (input.ndim - 1), (padding, padding)])
        length_out = length - (kernel_size - 1) + 2 * padding
    else:
        raise ValueError(f"Unknown padding {padding}")

    output = np.zeros((n, c_out, length_out), dtype=input.dtype)
    for m in range(n):
        for c in range(c_out):
            output[m, c] = correlate(input[m, 0], weights[c, 0], mode=mode)

    return output

