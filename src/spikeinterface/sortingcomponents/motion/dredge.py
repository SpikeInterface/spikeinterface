"""
Re-implementation or copy-paste of DREDge
https://github.com/evarol/dredge

For historical reason, some function from the DREDge package where implemeneted 
in spikeinterface in the motion_estimation.py before the DREDge package itself!

Here a copy/paste (and small rewriting) of some functions from DREDge.

The main entry for this function are still:

  * motion = estimate_motion((recording, ..., method='dredge_lfp')
  * motion = estimate_motion((recording, ..., method='dredge_ap')

but here the original functions from Charlie, Julien and Erdem have been ported for an
easier maintenance instead of making DREDge a dependency of spikeinterface.
"""

# TODO
# discuss the get_windows in dredge
# remove get_motion_estimate() ???


import numpy as np


def dredge_online_lfp(
    lfp_recording,
    rigid=True,
    chunk_len_s=10.0,
    max_disp_um=500,
    # nonrigid window construction arguments
    win_shape="gaussian",
    win_step_um=800,
    win_scale_um=850,
    win_margin_um=None,
    max_dt_s=None,
    # weighting arguments
    mincorr=0.8,
    mincorr_percentile=None,
    mincorr_percentile_nneighbs=20,
    soft=False,
    # low-level arguments
    thomas_kw=None,
    xcorr_kw=None,
    # misc
    save_full=False,
    device=None,
    pbar=True,
):
    """Online registration of a preprocessed LFP recording

    Arguments
    ---------
    lfp_recording : spikeinterface BaseRecording object
        Preprocessed LFP recording. The temporal resolution of this recording will
        be the target resolution of the registration, so definitely use SpikeInterface
        to resample your recording to, say, 250Hz (or a value you like) rather than
        estimating motion at the original frequency (which may be high).
    rigid : boolean, optional
        If True, window-related arguments are ignored and we do rigid registration
    chunk_len_s : float
        Length of chunks (in seconds) that the recording is broken into for online
        registration. The computational speed of the method is a function of the
        number of samples this corresponds to, and things can get slow if it is
        set high enough that the number of samples per chunk is bigger than ~10,000.
        But, it can't be set too low or the algorithm doesn't have enough data
        to work with. The default is set assuming sampling rate of 250Hz, leading
        to 2500 samples per chunk.
    max_dt_s : float
        Time-bins farther apart than this value in seconds will not be cross-correlated.
        Set this to at least `chunk_len_s`.
    max_disp_um : number, optional
        This is the ceiling on the possible displacement estimates. It should be
        set to a number which is larger than the allowed displacement in a single
        chunk. Setting it as small as possible (while following that rule) can speed
        things up and improve the result by making it impossible to estimate motion
        which is too big.
    win_shape, win_step_um, win_scale_um, win_margin_um : float
        Nonrigid window-related arguments
        The depth domain will be broken up into windows with shape controlled by win_shape,
        spaced by win_step_um at a margin of win_margin_um from the boundary, and with
        width controlled by win_scale_um.
    mincorr : float in [0,1]
        Minimum correlation between pairs of frames such that they will be included
        in the optimization of the displacement estimates.
    mincorr_percentile, mincorr_percentile_nneighbs
        If mincorr_percentile is set to a number in [0, 100], then mincorr will be replaced
        by this percentile of the correlations of neighbors within mincorr_percentile_nneighbs
        time bins of each other.
    device : string or torch.device
        Controls torch device

    Returns
    -------
    me : motion_util.MotionEstimate
        A motion estimate object. me.displacement is the displacement trace, but this object
        includes methods for getting the displacement at different times and depths; see
        the documentation in the motion_util.py file.
    extra : dict
        Dict containing extra info for debugging
    """
    geom = lfp_recording.get_channel_locations()
    fs = lfp_recording.get_sampling_frequency()
    T_total = lfp_recording.get_num_samples()
    T_chunk = min(int(np.floor(fs * chunk_len_s)), T_total)

    # kwarg defaults and handling
    # need lfp-specific defaults
    xcorr_kw = xcorr_kw if xcorr_kw is not None else {}
    thomas_kw = thomas_kw if thomas_kw is not None else {}
    full_xcorr_kw = dict(
        rigid=rigid,
        bin_um=np.median(np.diff(geom[:, 1])),
        max_disp_um=max_disp_um,
        pbar=False,
        device=device,
        **xcorr_kw,
    )
    threshold_kw = dict(
        mincorr_percentile_nneighbs=mincorr_percentile_nneighbs,
        in_place=True,
        soft=soft,
        # max_dt_s=weights_kw["max_dt_s"],  # max_dt not implemented for lfp at this point
        max_dt_s=max_dt_s,
        bin_s=1 / fs,  # only relevant for max_dt_s
    )

    # get windows
    windows, window_centers = get_windows(
        geom,
        win_step_um,
        win_scale_um,
        spatial_bin_centers=geom[:, 1],
        margin_um=win_margin_um,
        win_shape=win_shape,
        zero_threshold=1e-5,
        rigid=rigid,
    )
    B = len(windows)
    extra = dict(window_centers=window_centers, windows=windows)

    # -- allocate output and initialize first chunk
    P_online = np.empty((B, T_total), dtype=np.float32)
    # below, t0 is start of prev chunk, t1 start of cur chunk, t2 end of cur
    t0, t1 = 0, T_chunk
    traces0 = lfp_recording.get_traces(start_frame=t0, end_frame=t1)
    Ds0, Cs0, max_disp_um = xcorr_windows(
        traces0.T, windows, geom[:, 1], win_scale_um, **full_xcorr_kw
    )
    full_xcorr_kw["max_disp_um"] = max_disp_um
    Ss0, mincorr0 = threshold_correlation_matrix(
        Cs0,
        mincorr=mincorr,
        mincorr_percentile=mincorr_percentile,
        **threshold_kw,
    )
    if save_full:
        extra["D"] = [Ds0]
        extra["C"] = [Cs0]
        extra["S"] = [Ss0]
        extra["D01"] = []
        extra["C01"] = []
        extra["S01"] = []
    extra["mincorrs"] = [mincorr0]
    extra["max_disp_um"] = max_disp_um
    P_online[:, t0:t1], _ = thomas_solve(Ds0, Ss0, **thomas_kw)

    # -- loop through chunks
    chunk_starts = range(T_chunk, T_total, T_chunk)
    if pbar:
        chunk_starts = trange(
            T_chunk,
            T_total,
            T_chunk,
            desc=f"Online chunks [{chunk_len_s}s each]",
        )
    for t1 in chunk_starts:
        t2 = min(T_total, t1 + T_chunk)
        traces1 = lfp_recording.get_traces(start_frame=t1, end_frame=t2)

        # cross-correlations between prev/cur chunks
        # these are T1, T0 shaped
        Ds10, Cs10, _ = xcorr_windows(
            traces1.T,
            windows,
            geom[:, 1],
            win_scale_um,
            raster_b=traces0.T,
            **full_xcorr_kw,
        )

        # cross-correlation in current chunk
        Ds1, Cs1, _ = xcorr_windows(
            traces1.T, windows, geom[:, 1], win_scale_um, **full_xcorr_kw
        )
        Ss1, mincorr1 = threshold_correlation_matrix(
            Cs1,
            mincorr_percentile=mincorr_percentile,
            mincorr=mincorr,
            **threshold_kw,
        )
        Ss10, _ = threshold_correlation_matrix(
            Cs10, mincorr=mincorr1, t_offset_bins=T_chunk, **threshold_kw
        )
        extra["mincorrs"].append(mincorr1)

        if save_full:
            extra["D"].append(Ds1)
            extra["C"].append(Cs1)
            extra["S"].append(Ss1)
            extra["D01"].append(Ds10)
            extra["C01"].append(Cs10)
            extra["S01"].append(Ss10)

        # solve online problem
        P_online[:, t1:t2], _ = thomas_solve(
            Ds1,
            Ss1,
            P_prev=P_online[:, t0:t1],
            Ds_curprev=Ds10,
            Us_curprev=Ss10,
            Ds_prevcur=-Ds10.transpose(0, 2, 1),
            Us_prevcur=Ss10.transpose(0, 2, 1),
            **thomas_kw,
        )

        # update loop vars
        t0, t1 = t1, t2
        traces0 = traces1

    # -- convert to motion estimate and return
    me = get_motion_estimate(
        P_online,
        time_bin_centers_s=lfp_recording.get_times(0),
        spatial_bin_centers_um=window_centers,
    )
    return me, extra



# -- functions from dredgelib



def thomas_solve(
    Ds,
    Us,
    lambda_t=DEFAULT_LAMBDA_T,
    lambda_s=1.0,
    eps=DEFAULT_EPS,
    P_prev=None,
    Ds_prevcur=None,
    Us_prevcur=None,
    Ds_curprev=None,
    Us_curprev=None,
    pbar=False,
    bandwidth=None,
):
    """Block tridiagonal algorithm, special cased to our setting

    This code solves for the displacement estimates across the nonrigid windows,
    given blockwise, pairwise (BxTxT) displacement and weights arrays `Ds` and `Us`.

    If `lambda_t>0`, a temporal prior is applied to "fill the gaps", effectively
    interpolating through time to avoid artifacts in low-signal areas. Setting this
    to 0 can lead to numerical warnings and should be done with care.

    If `lambda_s>0`, a spatial prior is applied. This can help fill gaps more
    meaningfully in the nonrigid case, using information from the neighboring nonrigid
    windows to inform the estimate in an untrusted region of a given window.

    If arguments `P_prev,Ds_prevcur,Us_prevcur` are supplied, this code handles the
    online case. The return value will be the new chunk's displacement estimate,
    solving the online registration problem.
    """
    Ds = np.asarray(Ds, dtype=np.float64)
    Us = np.asarray(Us, dtype=np.float64)
    online = P_prev is not None
    online_kw_rhs = online_kw_hess = lambda b: {}
    if online:
        assert Ds_prevcur is not None
        assert Us_prevcur is not None
        online_kw_rhs = lambda b: dict(  # noqa
            Pb_prev=P_prev[b].astype(np.float64, copy=False),
            Db_prevcur=Ds_prevcur[b].astype(np.float64, copy=False),
            Ub_prevcur=Us_prevcur[b].astype(np.float64, copy=False),
            Db_curprev=Ds_curprev[b].astype(np.float64, copy=False),
            Ub_curprev=Us_curprev[b].astype(np.float64, copy=False),
        )
        online_kw_hess = lambda b: dict(  # noqa
            Ub_prevcur=Us_prevcur[b].astype(np.float64, copy=False),
            Ub_curprev=Us_curprev[b].astype(np.float64, copy=False),
        )

    B, T, T_ = Ds.shape
    assert T == T_
    assert Us.shape == Ds.shape

    # figure out which temporal bins are included in the problem
    # these are used to figure out where epsilon can be added
    # for numerical stability without changing the solution
    had_weights = (Us > 0).any(axis=2)
    had_weights[~had_weights.any(axis=1)] = 1

    # temporal prior matrix
    L_t = [laplacian(T, eps=eps, lambd=lambda_t, ridge_mask=w) for w in had_weights]
    extra = dict(L_t=L_t)

    # just solve independent problems when there's no spatial regularization
    # not that there's much overhead to the backward pass etc but might as well
    if B == 1 or lambda_s == 0:
        P = np.zeros((B, T))
        extra["HU"] = np.zeros((B, T, T))
        for b in range(B):
            P[b], extra["HU"][b] = newton_solve_rigid(
                Ds[b], Us[b], L_t[b], **online_kw_rhs(b)
            )
        return P, extra

    # spatial prior is a sparse, block tridiagonal kronecker product
    # the first and last diagonal blocks are
    Lambda_s_diagb = laplacian(T, eps=eps, lambd=lambda_s / 2, ridge_mask=had_weights[0])
    # and the off-diagonal blocks are
    Lambda_s_offdiag = laplacian(T, eps=0, lambd=-lambda_s / 2)

    # initialize block-LU stuff and forward variable
    alpha_hat_b = (
        L_t[0]
        + Lambda_s_diagb
        + neg_hessian_likelihood_term(Us[0], **online_kw_hess(0))
    )
    targets = np.c_[
        Lambda_s_offdiag, newton_rhs(Us[0], Ds[0], **online_kw_rhs(0))
    ]
    res = solve(alpha_hat_b, targets, assume_a="pos")
    assert res.shape == (T, T + 1)
    gamma_hats = [res[:, :T]]
    ys = [res[:, T]]

    # forward pass
    for b in (trange(1, B, desc="Solve") if pbar else range(1, B)):
        if b < B - 1:
            Lambda_s_diagb = laplacian(T, eps=eps, lambd=lambda_s, ridge_mask=had_weights[b])
        else:
            Lambda_s_diagb = laplacian(T, eps=eps, lambd=lambda_s / 2, ridge_mask=had_weights[b])

        Ab = (
            L_t[b]
            + Lambda_s_diagb
            + neg_hessian_likelihood_term(Us[b], **online_kw_hess(b))
        )
        alpha_hat_b = Ab - Lambda_s_offdiag @ gamma_hats[b - 1]
        targets[:, T] = newton_rhs(Us[b], Ds[b], **online_kw_rhs(b))
        targets[:, T] -= Lambda_s_offdiag @ ys[b - 1]
        res = solve(alpha_hat_b, targets)
        assert res.shape == (T, T + 1)
        gamma_hats.append(res[:, :T])
        ys.append(res[:, T])

    # back substitution
    xs = [None] * B
    xs[-1] = ys[-1]
    for b in range(B - 2, -1, -1):
        xs[b] = ys[b] - gamma_hats[b] @ xs[b + 1]

    # un-vectorize
    P = np.concatenate(xs).reshape(B, T)

    return P, extra



def threshold_correlation_matrix(
    Cs,
    mincorr=0.0,
    mincorr_percentile=None,
    mincorr_percentile_nneighbs=20,
    max_dt_s=0,
    in_place=False,
    bin_s=1,
    t_offset_bins=None,
    T=None,
    soft=True,
):
    if mincorr_percentile is not None:
        diags = [
            np.diagonal(Cs, offset=j, axis1=1, axis2=2).ravel()
            for j in range(1, mincorr_percentile_nneighbs)
        ]
        mincorr = np.percentile(
            np.concatenate(diags),
            mincorr_percentile,
        )

    # need abs to avoid -0.0s which cause numerical issues
    if in_place:
        Ss = Cs
        if soft:
            Ss[Ss < mincorr] = 0
        else:
            Ss = (Ss >= mincorr).astype(Cs.dtype)
        np.square(Ss, out=Ss)
    else:
        if soft:
            Ss = np.square((Cs >= mincorr) * Cs)
        else:
            Ss = (Cs >= mincorr).astype(Cs.dtype)
    if (
        max_dt_s is not None
        and max_dt_s > 0
        and T is not None
        and max_dt_s < T
    ):
        tt0 = bin_s * np.arange(T)
        tt1 = tt0
        if t_offset_bins:
            tt1 = tt0 + t_offset_bins
        dt = tt1[:, None] - tt0[None, :]
        mask = (np.abs(dt) <= max_dt_s).astype(Ss.dtype)
        Ss *= mask[None]
    return Ss, mincorr


def xcorr_windows(
    raster_a,
    windows,
    spatial_bin_edges_um,
    win_scale_um,
    raster_b=None,
    rigid=False,
    bin_um=1,
    max_disp_um=None,
    max_dt_bins=None,
    pbar=True,
    centered=True,
    normalized=True,
    masks=None,
    device=None,
):
    """Main computational function

    Compute pairwise (time x time) maximum cross-correlation and displacement
    matrices in each nonrigid window.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if max_disp_um is None:
        if rigid:
            max_disp_um = int(spatial_bin_edges_um.ptp() // 4)
        else:
            max_disp_um = int(win_scale_um // 4)

    max_disp_bins = int(max_disp_um // bin_um)
    slices = get_window_domains(windows)
    B, D = windows.shape
    D_, T0 = raster_a.shape
    assert D == D_

    # torch versions on device
    windows_ = torch.as_tensor(windows, dtype=torch.float, device=device)
    raster_a_ = torch.as_tensor(raster_a, dtype=torch.float, device=device)
    if raster_b is not None:
        assert raster_b.shape[0] == D
        T1 = raster_b.shape[1]
        raster_b_ = torch.as_tensor(raster_b, dtype=torch.float, device=device)
    else:
        T1 = T0
        raster_b_ = raster_a_
    if masks is not None:
        masks = torch.as_tensor(masks, dtype=torch.float, device=device)

    # estimate each window's displacement
    Ds = np.zeros((B, T0, T1), dtype=np.float32)
    Cs = np.zeros((B, T0, T1), dtype=np.float32)
    block_iter = trange(B, desc="Cross correlation") if pbar else range(B)
    for b in block_iter:
        window = windows_[b]

        # we search for the template (windowed part of raster a)
        # within a larger-than-the-window neighborhood in raster b
        targ_low = slices[b].start - max_disp_bins
        b_low = max(0, targ_low)
        targ_high = slices[b].stop + max_disp_bins
        b_high = min(D, targ_high)
        padding = max(b_low - targ_low, targ_high - b_high)

        # arithmetic to compute the lags in um corresponding to
        # corr argmaxes
        n_left = padding + slices[b].start - b_low
        n_right = padding + b_high - slices[b].stop
        poss_disp = -np.arange(-n_left, n_right + 1) * bin_um

        Ds[b], Cs[b] = calc_corr_decent_pair(
            raster_a_[slices[b]],
            raster_b_[b_low:b_high],
            weights=window[slices[b]],
            masks=None if masks is None else masks[slices[b]],
            xmasks=None if masks is None else masks[b_low:b_high],
            disp=padding,
            possible_displacement=poss_disp,
            device=device,
            centered=centered,
            normalized=normalized,
            max_dt_bins=max_dt_bins,
        )

    return Ds, Cs, max_disp_um


def calc_corr_decent_pair(
    raster_a,
    raster_b,
    weights=None,
    masks=None,
    xmasks=None,
    disp=None,
    batch_size=512,
    normalized=True,
    centered=True,
    possible_displacement=None,
    max_dt_bins=None,
    device=None,
):
    """Weighted pairwise cross-correlation

    Calculate TxT normalized xcorr and best displacement matrices
    Given a DxT raster, this computes normalized cross correlations for
    all pairs of time bins at offsets in the range [-disp, disp], by
    increments of step_size. Then it finds the best one and its
    corresponding displacement, resulting in two TxT matrices: one for
    the normxcorrs at the best displacement, and the matrix of the best
    displacements.

    Arguments
    ---------
    raster : DxT array
    batch_size : int
        How many raster rows to xcorr against the whole raster
        at once.
    step_size : int
        Displacement increment. Not implemented yet but easy to do.
    disp : int
        Maximum displacement
    device : torch device
    Returns: D, C: TxT arrays
    """
    D, Ta = raster_a.shape
    D_, Tb = raster_b.shape

    # sensible default: at most half the domain.
    if disp is None:
        disp == D // 2

    # range of displacements
    if D == D_:
        if possible_displacement is None:
            possible_displacement = np.arange(-disp, disp + 1)
    else:
        assert possible_displacement is not None
        assert disp is not None

    # pick torch device if unset
    if device is None:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    # process rasters into the tensors we need for conv2ds below
    # convert to TxD device floats
    raster_a = torch.as_tensor(raster_a.T, dtype=torch.float32, device=device)
    # normalize over depth for normalized (uncentered) xcorrs
    raster_b = torch.as_tensor(raster_b.T, dtype=torch.float32, device=device)

    D = np.zeros((Ta, Tb), dtype=np.float32)
    C = np.zeros((Ta, Tb), dtype=np.float32)
    for i in range(0, Ta, batch_size):
        for j in range(0, Tb, batch_size):
            dt_bins = min(
                abs(i - j), abs(i + batch_size - j), abs(i - j - batch_size)
            )
            if max_dt_bins and dt_bins > max_dt_bins:
                continue
            weights_ = weights
            if masks is not None:
                weights_ = masks.T[i : i + batch_size] * weights
            corr = normxcorr1d(
                raster_a[i : i + batch_size],
                raster_b[j : j + batch_size],
                weights=weights_,
                xmasks=None if xmasks is None else xmasks.T[j : j + batch_size],
                padding=disp,
                normalized=normalized,
                centered=centered,
            )
            max_corr, best_disp_inds = torch.max(corr, dim=2)
            best_disp = possible_displacement[best_disp_inds.cpu()]
            D[i : i + batch_size, j : j + batch_size] = best_disp.T
            C[i : i + batch_size, j : j + batch_size] = max_corr.cpu().T

    return D, C