import numpy as np

from tqdm.auto import tqdm, trange


from .motion_utils import Motion, get_spatial_windows, get_spatial_bin_edges, make_2d_motion_histogram, scipy_conv1d

from .dredge import normxcorr1d


class DecentralizedRegistration:
    """
    Method developed by the Paninski's group from Columbia university:
    Charlie Windolf, Julien Boussard, Erdem Varol, Hyun Dong Lee

    This method is also known as DREDGe, but this implemenation does not use LFP signals.

    Original reference:
    DECENTRALIZED MOTION INFERENCE AND REGISTRATION OF NEUROPIXEL DATA
    https://ieeexplore.ieee.org/document/9414145
    https://proceedings.neurips.cc/paper/2021/hash/b950ea26ca12daae142bd74dba4427c8-Abstract.html

    This code was improved during Spike Sorting NY Hackathon 2022 by Erdem Varol and Charlie Windolf.
    An additional major improvement can be found in this paper:
    https://www.biorxiv.org/content/biorxiv/early/2022/12/05/2022.12.04.519043.full.pdf


    Here are some various implementations by the original team:
    https://github.com/int-brain-lab/spikes_localization_registration/blob/main/registration_pipeline/image_based_motion_estimate.py#L211
    https://github.com/cwindolf/spike-psvae/tree/main/spike_psvae
    https://github.com/evarol/DREDge
    """

    name = "decentralized"
    need_peak_location = True
    params_doc = """
    bin_um: float, default: 10
        Spatial bin size in micrometers
    hist_margin_um: float, default: 0
        Margin in um from histogram estimation.
        Positive margin extrapolate out of the probe the motion.
        Negative margin crop the motion on the border
    bin_s: float, default 1.0
        Bin duration in second
    histogram_depth_smooth_um: None or float
        Optional gaussian smoother on histogram on depth axis.
        This is given as the sigma of the gaussian in micrometers.
    histogram_time_smooth_s: None or float
        Optional gaussian smoother on histogram on time axis.
        This is given as the sigma of the gaussian in seconds.
    pairwise_displacement_method: "conv" or "phase_cross_correlation"
        How to estimate the displacement in the pairwise matrix.
    max_displacement_um: float
        Maximum possible displacement in micrometers.
    weight_scale: "linear" or "exp"
        For parwaise displacement, how to to rescale the associated weight matrix.
    error_sigma: float, default: 0.2
        In case weight_scale="exp" this controls the sigma of the exponential.
    conv_engine: "numpy" or "torch" or None, default: None
        In case of pairwise_displacement_method="conv", what library to use to compute
        the underlying correlation
    torch_device=None
        In case of conv_engine="torch", you can control which device (cpu or gpu)
    batch_size: int
        Size of batch for the convolution. Increasing this will speed things up dramatically
        on GPUs and sometimes on CPU as well.
    corr_threshold: float
        Minimum correlation between pair of time bins in order for these to be
        considered when optimizing a global displacment vector to align with
        the pairwise displacements.
    time_horizon_s: None or float
        When not None the parwise discplament matrix is computed in a small time horizon.
        In short only pair of bins close in time.
        So the pariwaise matrix is super sparse and have values only the diagonal.
    convergence_method: "lsmr" | "lsqr_robust" | "gradient_descent", default: "lsmr"
        Which method to use to compute the global displacement vector from the pairwise matrix.
    robust_regression_sigma: float
        Use for convergence_method="lsqr_robust" for iterative selection of the regression.
    temporal_prior : bool, default: True
        Ensures continuity across time, unless there is evidence in the recording for jumps.
    spatial_prior : bool, default: False
        Ensures continuity across space. Not usually necessary except in recordings with
        glitches across space.
    force_spatial_median_continuity: bool, default: False
        When spatial_prior=False we can optionally apply a median continuity across spatial windows.
    reference_displacement : string, one of: "mean", "median", "time", "mode_search"
        Strategy for picking what is considered displacement=0.
         - "mean" : the mean displacement is subtracted
         - "median" : the median displacement is subtracted
         - "time" : the displacement at a given time (in seconds) is subtracted
         - "mode_search" : an attempt is made to guess the mode. needs work.
    lsqr_robust_n_iter: int
        Number of iteration for convergence_method="lsqr_robust".
    """

    @classmethod
    def run(
        cls,
        recording,
        peaks,
        peak_locations,
        direction,
        rigid,
        win_shape,
        win_step_um,
        win_scale_um,
        win_margin_um,
        verbose,
        progress_bar,
        extra,
        bin_um=1.0,
        hist_margin_um=20.0,
        bin_s=1.0,
        histogram_depth_smooth_um=1.0,
        histogram_time_smooth_s=1.0,
        pairwise_displacement_method="conv",
        max_displacement_um=100.0,
        weight_scale="linear",
        error_sigma=0.2,
        conv_engine=None,
        torch_device=None,
        batch_size=1,
        corr_threshold=0.0,
        time_horizon_s=None,
        convergence_method="lsmr",
        soft_weights=False,
        normalized_xcorr=True,
        centered_xcorr=True,
        temporal_prior=True,
        spatial_prior=False,
        force_spatial_median_continuity=False,
        reference_displacement="median",
        reference_displacement_time_s=0,
        robust_regression_sigma=2,
        lsqr_robust_n_iter=20,
        weight_with_amplitude=False,
    ):

        dim = ["x", "y", "z"].index(direction)
        contact_depths = recording.get_channel_locations()[:, dim]

        # spatial histogram bins
        spatial_bin_edges = get_spatial_bin_edges(recording, direction, hist_margin_um, bin_um)
        spatial_bin_centers = 0.5 * (spatial_bin_edges[1:] + spatial_bin_edges[:-1])

        # get spatial windows
        non_rigid_windows, non_rigid_window_centers = get_spatial_windows(
            contact_depths,
            spatial_bin_centers,
            rigid=rigid,
            win_shape=win_shape,
            win_step_um=win_step_um,
            win_scale_um=win_scale_um,
            win_margin_um=win_margin_um,
            zero_threshold=None,
        )

        # make 2D histogram raster
        if verbose:
            print("Computing motion histogram")

        motion_histogram, temporal_hist_bin_edges, spatial_hist_bin_edges = make_2d_motion_histogram(
            recording,
            peaks,
            peak_locations,
            direction=direction,
            bin_s=bin_s,
            spatial_bin_edges=spatial_bin_edges,
            weight_with_amplitude=weight_with_amplitude,
            depth_smooth_um=histogram_depth_smooth_um,
            time_smooth_s=histogram_time_smooth_s,
        )

        if extra is not None:
            extra["motion_histogram"] = motion_histogram
            extra["pairwise_displacement_list"] = []
            extra["temporal_hist_bin_edges"] = temporal_hist_bin_edges
            extra["spatial_hist_bin_edges"] = spatial_hist_bin_edges

        # temporal bins are bin center
        temporal_bins = 0.5 * (temporal_hist_bin_edges[1:] + temporal_hist_bin_edges[:-1])

        motion_array = np.zeros((temporal_bins.size, len(non_rigid_windows)), dtype=np.float64)
        windows_iter = non_rigid_windows
        if progress_bar:
            windows_iter = tqdm(windows_iter, desc="windows")
        if spatial_prior:
            all_pairwise_displacements = np.empty(
                (len(non_rigid_windows), temporal_bins.size, temporal_bins.size), dtype=np.float64
            )
            all_pairwise_displacement_weights = np.empty(
                (len(non_rigid_windows), temporal_bins.size, temporal_bins.size), dtype=np.float64
            )
        for i, win in enumerate(windows_iter):
            window_slice = np.flatnonzero(win > 1e-5)
            window_slice = slice(window_slice[0], window_slice[-1])
            if verbose:
                print(f"Computing pairwise displacement: {i + 1} / {len(non_rigid_windows)}")

            
            pairwise_displacement, pairwise_displacement_weight = compute_pairwise_displacement(
                motion_histogram[:, window_slice],
                bin_um,
                window=win[window_slice],
                method=pairwise_displacement_method,
                weight_scale=weight_scale,
                error_sigma=error_sigma,
                conv_engine=conv_engine,
                torch_device=torch_device,
                batch_size=batch_size,
                max_displacement_um=max_displacement_um,
                normalized_xcorr=normalized_xcorr,
                centered_xcorr=centered_xcorr,
                corr_threshold=corr_threshold,
                time_horizon_s=time_horizon_s,
                bin_s=bin_s,
                progress_bar=False,
            )

            if spatial_prior:
                all_pairwise_displacements[i] = pairwise_displacement
                all_pairwise_displacement_weights[i] = pairwise_displacement_weight

            if extra is not None:
                extra["pairwise_displacement_list"].append(pairwise_displacement)

            if verbose:
                print(f"Computing global displacement: {i + 1} / {len(non_rigid_windows)}")

            # TODO: if spatial_prior, do this after the loop
            if not spatial_prior:
                motion_array[:, i] = compute_global_displacement(
                    pairwise_displacement,
                    pairwise_displacement_weight=pairwise_displacement_weight,
                    convergence_method=convergence_method,
                    robust_regression_sigma=robust_regression_sigma,
                    lsqr_robust_n_iter=lsqr_robust_n_iter,
                    temporal_prior=temporal_prior,
                    spatial_prior=spatial_prior,
                    soft_weights=soft_weights,
                    progress_bar=False,
                )

        if spatial_prior:
            motion_array = compute_global_displacement(
                all_pairwise_displacements,
                pairwise_displacement_weight=all_pairwise_displacement_weights,
                convergence_method=convergence_method,
                robust_regression_sigma=robust_regression_sigma,
                lsqr_robust_n_iter=lsqr_robust_n_iter,
                temporal_prior=temporal_prior,
                spatial_prior=spatial_prior,
                soft_weights=soft_weights,
                progress_bar=False,
            )
        elif len(non_rigid_windows) > 1:
            # if spatial_prior is False, we still want keep the spatial bins
            # correctly offset from each other
            if force_spatial_median_continuity:
                for i in range(len(non_rigid_windows) - 1):
                    motion_array[:, i + 1] -= np.median(motion_array[:, i + 1] - motion_array[:, i])

        # try to avoid constant offset
        # let the user choose how to do this. here are some ideas.
        # (one can also -= their own number on the result of this function.)
        if reference_displacement == "mean":
            motion_array -= motion_array.mean()
        elif reference_displacement == "median":
            motion_array -= np.median(motion_array)
        elif reference_displacement == "time":
            # reference the motion to 0 at a specific time, independently in each window
            reference_displacement_bin = np.digitize(reference_displacement_time_s, temporal_hist_bin_edges) - 1
            motion_array -= motion_array[reference_displacement_bin, :]
        elif reference_displacement == "mode_search":
            # just a sketch of an idea
            # things might want to change, should have a configurable bin size,
            # should use a call to histogram instead of the loop, ...
            step_size = 0.1
            round_mode = np.round  # floor?
            best_ref = np.median(motion_array)
            max_zeros = np.sum(round_mode(motion_array - best_ref) == 0)
            for ref in np.arange(np.floor(motion_array.min()), np.ceil(motion_array.max()), step_size):
                n_zeros = np.sum(round_mode(motion_array - ref) == 0)
                if n_zeros > max_zeros:
                    max_zeros = n_zeros
                    best_ref = ref
            motion_array -= best_ref

        # replace nan by zeros
        np.nan_to_num(motion_array, copy=False)

        motion = Motion([motion_array], [temporal_bins], non_rigid_window_centers, direction=direction)

        return motion


def compute_pairwise_displacement(
    motion_hist,
    bin_um,
    method="conv",
    weight_scale="linear",
    error_sigma=0.2,
    conv_engine="numpy",
    torch_device=None,
    batch_size=1,
    max_displacement_um=1500,
    corr_threshold=0,
    time_horizon_s=None,
    normalized_xcorr=True,
    centered_xcorr=True,
    bin_s=None,
    progress_bar=False,
    window=None,
):
    """
    Compute pairwise displacement
    """
    from scipy import linalg

    if conv_engine is None:
        # use torch if installed
        try:
            import torch

            conv_engine = "torch"
        except ImportError:
            conv_engine = "numpy"

    if conv_engine == "torch":
        import torch

    assert conv_engine in ("torch", "numpy"), f"'conv_engine' must be 'torch' or 'numpy'"
    size = motion_hist.shape[0]
    pairwise_displacement = np.zeros((size, size), dtype="float32")

    if time_horizon_s is not None:
        band_width = int(np.ceil(time_horizon_s / bin_s))
        if band_width >= size:
            time_horizon_s = None

    if conv_engine == "torch":
        if torch_device is None:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if method == "conv":
        if max_displacement_um is None:
            n = motion_hist.shape[1] // 2
        else:
            n = min(
                motion_hist.shape[1] // 2,
                int(np.ceil(max_displacement_um // bin_um)),
            )
        possible_displacement = np.arange(-n, n + 1) * bin_um

        xrange = trange if progress_bar else range

        motion_hist_engine = motion_hist
        window_engine = window
        if conv_engine == "torch":
            motion_hist_engine = torch.as_tensor(motion_hist, dtype=torch.float32, device=torch_device)
            window_engine = torch.as_tensor(window, dtype=torch.float32, device=torch_device)

        pairwise_displacement = np.empty((size, size), dtype=np.float32)
        correlation = np.empty((size, size), dtype=motion_hist.dtype)

        for i in xrange(0, size, batch_size):
            print('yep', i, size, conv_engine, motion_hist_engine.shape)
            corr = normxcorr1d(
                motion_hist_engine,
                motion_hist_engine[i : i + batch_size],
                weights=window_engine,
                padding=possible_displacement.size // 2,
                conv_engine=conv_engine,
                normalized=normalized_xcorr,
                centered=centered_xcorr,
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

        if corr_threshold is not None and corr_threshold > 0:
            which = correlation > corr_threshold
            correlation *= which

    elif method == "phase_cross_correlation":
        # this 'phase_cross_correlation' is an old idea from Julien/Charlie/Erden that is kept for testing
        # but this is not very releveant
        try:
            import skimage.registration
        except ImportError:
            raise ImportError("To use the 'phase_cross_correlation' method install scikit-image")

        errors = np.zeros((size, size), dtype="float32")
        loop = range(size)
        if progress_bar:
            loop = tqdm(loop)
        for i in loop:
            for j in range(size):
                shift, error, diffphase = skimage.registration.phase_cross_correlation(
                    motion_hist[i, :], motion_hist[j, :]
                )
                pairwise_displacement[i, j] = shift * bin_um
                errors[i, j] = error
        correlation = 1 - errors

    else:
        raise ValueError(
            f"method {method} does not exist for compute_pairwise_displacement. Current possible methods are"
            f" 'conv' or 'phase_cross_correlation'"
        )

    if weight_scale == "linear":
        # between 0 and 1
        pairwise_displacement_weight = correlation
    elif weight_scale == "exp":
        pairwise_displacement_weight = np.exp((correlation - 1) / error_sigma)

    # handle the time horizon by multiplying the weights by a
    # matrix with the time horizon on its diagonal bands.
    if method == "conv" and time_horizon_s is not None and time_horizon_s > 0:
        horizon_matrix = linalg.toeplitz(
            np.r_[np.ones(band_width, dtype=bool), np.zeros(size - band_width, dtype=bool)]
        )
        pairwise_displacement_weight *= horizon_matrix

    return pairwise_displacement, pairwise_displacement_weight


_possible_convergence_method = ("lsmr", "gradient_descent", "lsqr_robust")


def compute_global_displacement(
    pairwise_displacement,
    pairwise_displacement_weight=None,
    sparse_mask=None,
    temporal_prior=True,
    spatial_prior=True,
    soft_weights=False,
    convergence_method="lsmr",
    robust_regression_sigma=2,
    lsqr_robust_n_iter=20,
    progress_bar=False,
):
    """
    Compute global displacement

    Arguments
    ---------
    pairwise_displacement : time x time array
    pairwise_displacement_weight : time x time array
    sparse_mask : time x time array
    convergence_method : str
        One of "gradient"

    """
    import scipy
    from scipy.optimize import minimize
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr
    from scipy.stats import zscore

    if convergence_method == "gradient_descent":
        size = pairwise_displacement.shape[0]

        D = pairwise_displacement
        if pairwise_displacement_weight is not None or sparse_mask is not None:
            # weighted problem
            if pairwise_displacement_weight is None:
                pairwise_displacement_weight = np.ones_like(D)
            if sparse_mask is None:
                sparse_mask = np.ones_like(D)
            W = pairwise_displacement_weight * sparse_mask

            I, J = np.nonzero(W > 0)
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

        res = minimize(fun=obj, jac=jac, x0=D.mean(axis=1), method="L-BFGS-B")
        if not res.success:
            print("Global displacement gradient descent had an error")
        displacement = res.x

    elif convergence_method == "lsqr_robust":

        if sparse_mask is not None:
            I, J = np.nonzero(sparse_mask > 0)
        elif pairwise_displacement_weight is not None:
            I, J = pairwise_displacement_weight.nonzero()
        else:
            I, J = np.nonzero(np.ones_like(pairwise_displacement, dtype=bool))

        nnz_ones = np.ones(I.shape[0], dtype=pairwise_displacement.dtype)

        if pairwise_displacement_weight is not None:
            if isinstance(pairwise_displacement_weight, scipy.sparse.csr_matrix):
                W = np.array(pairwise_displacement_weight[I, J]).T
            else:
                W = pairwise_displacement_weight[I, J][:, None]
        else:
            W = nnz_ones[:, None]
        if isinstance(pairwise_displacement, scipy.sparse.csr_matrix):
            V = np.array(pairwise_displacement[I, J])[0]
        else:
            V = pairwise_displacement[I, J]
        M = csr_matrix((nnz_ones, (range(I.shape[0]), I)), shape=(I.shape[0], pairwise_displacement.shape[0]))
        N = csr_matrix((nnz_ones, (range(I.shape[0]), J)), shape=(I.shape[0], pairwise_displacement.shape[0]))
        A = M - N
        idx = np.ones(A.shape[0], dtype=bool)

        # TODO: this is already soft_weights
        xrange = trange if progress_bar else range
        for i in xrange(lsqr_robust_n_iter):
            p = lsqr(A[idx].multiply(W[idx]), V[idx] * W[idx][:, 0])[0]
            idx = np.nonzero(np.abs(zscore(A @ p - V)) <= robust_regression_sigma)
        displacement = p

    elif convergence_method == "lsmr":
        import gc
        from scipy import sparse

        D = pairwise_displacement

        # weighted problem
        if pairwise_displacement_weight is None:
            pairwise_displacement_weight = np.ones_like(D)
        if sparse_mask is None:
            sparse_mask = np.ones_like(D)
        W = pairwise_displacement_weight * sparse_mask
        if isinstance(W, scipy.sparse.csr_matrix):
            W = W.astype(np.float32).toarray()
            D = D.astype(np.float32).toarray()

        assert D.shape == W.shape

        # first dimension is the windows dim, which could be empty in rigid case
        # we expand dims so that below we can consider only the nonrigid case
        if D.ndim == 2:
            W = W[None]
            D = D[None]
        assert D.ndim == W.ndim == 3
        B, T, T_ = D.shape
        assert T == T_

        # sparsify the problem
        # we will make a list of temporal problems and then
        # stack over the windows axis to finish.
        # each matrix in coefficients will be (sparse_dim, T)
        coefficients = []
        # each vector in targets will be (T,)
        targets = []
        # we want to solve for a vector of shape BT, which we will reshape
        # into a (B, T) matrix.
        # after the loop below, we will stack a coefts matrix (sparse_dim, B, T)
        # and a target vector of shape (B, T), both to be vectorized on last two axes,
        # so that the target p is indexed by i = bT + t (block/window major).

        # calculate coefficients matrices and target vector
        # this list stores boolean masks corresponding to whether or not each
        # term comes from the prior or the likelihood. we can trim the likelihood terms,
        # but not the prior terms, in the trimmed least squares (robust iters) iterations below.
        cannot_trim = []
        for Wb, Db in zip(W, D):
            # indices of active temporal pairs in this window
            I, J = np.nonzero(Wb > 0)
            n_sampled = I.size

            # construct Kroneckers and sparse objective in this window
            pair_weights = np.ones(n_sampled)
            if soft_weights:
                pair_weights = Wb[I, J]
            Mb = sparse.csr_matrix((pair_weights, (range(n_sampled), I)), shape=(n_sampled, T))
            Nb = sparse.csr_matrix((pair_weights, (range(n_sampled), J)), shape=(n_sampled, T))
            block_sparse_kron = Mb - Nb
            block_disp_pairs = pair_weights * Db[I, J]
            cannot_trim_block = np.ones_like(block_disp_pairs, dtype=bool)

            # add the temporal smoothness prior in this window
            if temporal_prior:
                temporal_diff_operator = sparse.diags(
                    (
                        np.full(T - 1, -1, dtype=block_sparse_kron.dtype),
                        np.full(T - 1, 1, dtype=block_sparse_kron.dtype),
                    ),
                    offsets=(0, 1),
                    shape=(T - 1, T),
                )
                block_sparse_kron = sparse.vstack(
                    (block_sparse_kron, temporal_diff_operator),
                    format="csr",
                )
                block_disp_pairs = np.concatenate(
                    (block_disp_pairs, np.zeros(T - 1)),
                )
                cannot_trim_block = np.concatenate(
                    (cannot_trim_block, np.zeros(T - 1, dtype=bool)),
                )

            coefficients.append(block_sparse_kron)
            targets.append(block_disp_pairs)
            cannot_trim.append(cannot_trim_block)
        coefficients = sparse.block_diag(coefficients)
        targets = np.concatenate(targets, axis=0)
        cannot_trim = np.concatenate(cannot_trim, axis=0)

        # spatial smoothness prior: penalize difference of each block's
        # displacement with the next.
        # only if B > 1, and not in the last window.
        # this is a (BT, BT) sparse matrix D such that:
        # entry at (i, j) is:
        #  {   1 if i = j, i.e., i = j = bT + t for b = 0,...,B-2
        #  {  -1 if i = bT + t and j = (b+1)T + t for b = 0,...,B-2
        #  {   0 otherwise.
        # put more simply, the first (B-1)T diagonal entries are 1,
        # and entries (i, j) such that i = j - T are -1.
        if B > 1 and spatial_prior:
            spatial_diff_operator = sparse.diags(
                (
                    np.ones((B - 1) * T, dtype=block_sparse_kron.dtype),
                    np.full((B - 1) * T, -1, dtype=block_sparse_kron.dtype),
                ),
                offsets=(0, T),
                shape=((B - 1) * T, B * T),
            )
            coefficients = sparse.vstack((coefficients, spatial_diff_operator))
            targets = np.concatenate((targets, np.zeros((B - 1) * T, dtype=targets.dtype)))
            cannot_trim = np.concatenate((cannot_trim, np.zeros((B - 1) * T, dtype=bool)))
        coefficients = coefficients.tocsr()

        # initialize at the column mean of pairwise displacements (in each window)
        p0 = D.mean(axis=2).reshape(B * T)

        # use LSMR to solve the whole problem || targets - coefficients @ motion ||^2
        iters = range(max(1, lsqr_robust_n_iter))
        if progress_bar and lsqr_robust_n_iter > 1:
            iters = tqdm(iters, desc="robust lsqr")
        for it in iters:
            # trim active set -- start with no trimming
            idx = slice(None)
            if it:
                idx = np.flatnonzero(
                    cannot_trim | (np.abs(zscore(coefficients @ displacement - targets)) <= robust_regression_sigma)
                )

            # solve trimmed ols problem
            displacement, *_ = sparse.linalg.lsmr(coefficients[idx], targets[idx], x0=p0)

            # warm start next iteration
            p0 = displacement
            # Cleanup lsmr memory (see https://stackoverflow.com/questions/56147713/memory-leak-in-scipy)
            # TODO: check if this gets fixed in scipy
            gc.collect()

        displacement = displacement.reshape(B, T).T
    else:
        raise ValueError(
            f"Method {convergence_method} doesn't exist for compute_global_displacement"
            f" possible values for 'convergence_method' are {_possible_convergence_method}"
        )

    return np.squeeze(displacement)


# normxcorr1d is now implemented in dredge
# we keep the old version here but this will be removed soon

# def normxcorr1d(
#     template,
#     x,
#     weights=None,
#     centered=True,
#     normalized=True,
#     padding="same",
#     conv_engine="torch",
# ):
#     """normxcorr1d: Normalized cross-correlation, optionally weighted

#     The API is like torch's F.conv1d, except I have accidentally
#     changed the position of input/weights -- template acts like weights,
#     and x acts like input.

#     Returns the cross-correlation of `template` and `x` at spatial lags
#     determined by `mode`. Useful for estimating the location of `template`
#     within `x`.

#     This might not be the most efficient implementation -- ideas welcome.
#     It uses a direct convolutional translation of the formula
#         corr = (E[XY] - EX EY) / sqrt(var X * var Y)

#     This also supports weights! In that case, the usual adaptation of
#     the above formula is made to the weighted case -- and all of the
#     normalizations are done per block in the same way.

#     Parameters
#     ----------
#     template : tensor, shape (num_templates, length)
#         The reference template signal
#     x : tensor, 1d shape (length,) or 2d shape (num_inputs, length)
#         The signal in which to find `template`
#     weights : tensor, shape (length,)
#         Will use weighted means, variances, covariances if supplied.
#     centered : bool
#         If true, means will be subtracted (per weighted patch).
#     normalized : bool
#         If true, normalize by the variance (per weighted patch).
#     padding : str
#         How far to look? if unset, we'll use half the length
#     conv_engine : string, one of "torch", "numpy"
#         What library to use for computing cross-correlations.
#         If numpy, falls back to the scipy correlate function.

#     Returns
#     -------
#     corr : tensor
#     """
#     if conv_engine == "torch":
#         import torch
#         import torch.nn.functional as F
        
#         # assert HAVE_TORCH
#         conv1d = F.conv1d
#         npx = torch
#     elif conv_engine == "numpy":
#         conv1d = scipy_conv1d
#         npx = np
#     else:
#         raise ValueError(f"Unknown conv_engine {conv_engine}. 'conv_engine' must be 'torch' or 'numpy'")

#     x = npx.atleast_2d(x)
#     num_templates, length = template.shape
#     num_inputs, length_ = template.shape
#     assert length == length_

#     # generalize over weighted / unweighted case
#     device_kw = {} if conv_engine == "numpy" else dict(device=x.device)
#     ones = npx.ones((1, 1, length), dtype=x.dtype, **device_kw)
#     no_weights = weights is None
#     if no_weights:
#         weights = ones
#         wt = template[:, None, :]
#     else:
#         assert weights.shape == (length,)
#         weights = weights[None, None]
#         wt = template[:, None, :] * weights

#     # conv1d valid rule:
#     # (B,1,L),(O,1,L)->(B,O,L)

#     # compute expectations
#     # how many points in each window? seems necessary to normalize
#     # for numerical stability.
#     N = conv1d(ones, weights, padding=padding)
#     if centered:
#         Et = conv1d(ones, wt, padding=padding)
#         Et /= N
#         Ex = conv1d(x[:, None, :], weights, padding=padding)
#         Ex /= N

#     # compute (weighted) covariance
#     # important: the formula E[XY] - EX EY is well-suited here,
#     # because the means are naturally subtracted correctly
#     # patch-wise. you couldn't pre-subtract them!
#     cov = conv1d(x[:, None, :], wt, padding=padding)
#     cov /= N
#     if centered:
#         cov -= Ex * Et

#     # compute variances for denominator, using var X = E[X^2] - (EX)^2
#     if normalized:
#         var_template = conv1d(ones, wt * template[:, None, :], padding=padding)
#         var_template /= N
#         var_x = conv1d(npx.square(x)[:, None, :], weights, padding=padding)
#         var_x /= N
#         if centered:
#             var_template -= npx.square(Et)
#             var_x -= npx.square(Ex)

#     # now find the final normxcorr
#     corr = cov  # renaming for clarity
#     if normalized:
#         corr /= npx.sqrt(var_x)
#         corr /= npx.sqrt(var_template)
#         # get rid of NaNs in zero-variance areas
#         corr[~npx.isfinite(corr)] = 0

#     return corr
