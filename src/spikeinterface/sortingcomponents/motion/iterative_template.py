import numpy as np

from .motion_utils import Motion, get_spatial_windows, get_spatial_bin_edges, make_3d_motion_histograms


class IterativeTemplateRegistration:
    """
    Alignment function implemented by Kilosort2.5 and ported from pykilosort:
    https://github.com/int-brain-lab/pykilosort/blob/ibl_prod/pykilosort/datashift2.py#L166

    The main difference with respect to the original implementation are:
     * scipy is used for gaussian smoothing
     * windowing is implemented as gaussian tapering (instead of rectangular blocks)
     * the 3d histogram is constructed in less cryptic way
     * peak_locations are computed outside and so can either center fo mass or monopolar trianglation
       contrary to kilosort2.5 use exclusively center of mass

    See https://www.science.org/doi/abs/10.1126/science.abf4588?cookieSet=1

    Ported by Alessio Buccino into SpikeInterface
    """

    name = "iterative_template"
    need_peak_location = True
    params_doc = """
    bin_um: float, default: 10
        Spatial bin size in micrometers
    hist_margin_um: float, default: 0
        Margin in um from histogram estimation.
        Positive margin extrapolate out of the probe the motion.
        Negative margin crop the motion on the border
    bin_s: float, default: 2.0
        Bin duration in second
    num_amp_bins: int, default: 20
        number ob bins in the histogram on the log amplitues dimension
    num_shifts_global: int, default: 15
        Number of spatial bin shifts to consider for global alignment
    num_iterations: int, default: 10
        Number of iterations for global alignment procedure
    num_shifts_block: int, default: 5
        Number of spatial bin shifts to consider for non-rigid alignment
    smoothing_sigma: float, default: 0.5
        Sigma of gaussian for covariance matrices smoothing
    kriging_sigma: float,
        sigma parameter for kriging_kernel function
    kriging_p: foat
        p parameter for kriging_kernel function
    kriging_d: float
        d parameter for kriging_kernel function
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
        bin_um=10.0,
        hist_margin_um=0.0,
        bin_s=2.0,
        num_amp_bins=20,
        num_shifts_global=15,
        num_iterations=10,
        num_shifts_block=5,
        smoothing_sigma=0.5,
        kriging_sigma=1,
        kriging_p=2,
        kriging_d=2,
    ):

        dim = ["x", "y", "z"].index(direction)
        contact_depths = recording.get_channel_locations()[:, dim]

        # spatial histogram bins
        spatial_bin_edges = get_spatial_bin_edges(recording, direction, hist_margin_um, bin_um)
        spatial_bin_centers = 0.5 * (spatial_bin_edges[1:] + spatial_bin_edges[:-1])

        # get spatial windows
        non_rigid_windows, non_rigid_window_centers = get_spatial_windows(
            contact_depths=contact_depths,
            spatial_bin_centers=spatial_bin_centers,
            rigid=rigid,
            win_margin_um=win_margin_um,
            win_step_um=win_step_um,
            win_scale_um=win_scale_um,
            win_shape=win_shape,
            zero_threshold=None,
        )

        # make a 3D histogram
        if verbose:
            print("Making 3D motion histograms")
        motion_histograms, temporal_hist_bin_edges, spatial_hist_bin_edges = make_3d_motion_histograms(
            recording,
            peaks,
            peak_locations,
            direction=direction,
            num_amp_bins=num_amp_bins,
            bin_s=bin_s,
            spatial_bin_edges=spatial_bin_edges,
        )
        # temporal bins are bin center
        temporal_bins = temporal_hist_bin_edges[:-1] + bin_s // 2.0

        # do alignment
        if verbose:
            print("Estimating alignment shifts")
        shift_indices, target_histogram, shift_covs_block = iterative_template_registration(
            motion_histograms,
            non_rigid_windows=non_rigid_windows,
            num_shifts_global=num_shifts_global,
            num_iterations=num_iterations,
            num_shifts_block=num_shifts_block,
            smoothing_sigma=smoothing_sigma,
            kriging_sigma=kriging_sigma,
            kriging_p=kriging_p,
            kriging_d=kriging_d,
        )

        # convert to um
        motion_array = -(shift_indices * bin_um)

        if extra:
            extra["non_rigid_windows"] = non_rigid_windows
            extra["motion_histograms"] = motion_histograms
            extra["target_histogram"] = target_histogram
            extra["shift_covs_block"] = shift_covs_block
            extra["temporal_hist_bin_edges"] = temporal_hist_bin_edges
            extra["spatial_hist_bin_edges"] = spatial_hist_bin_edges

        # replace nan by zeros
        np.nan_to_num(motion_array, copy=False)

        motion = Motion([motion_array], [temporal_bins], non_rigid_window_centers, direction=direction)

        return motion


def iterative_template_registration(
    spikecounts_hist_images,
    non_rigid_windows=None,
    num_shifts_global=15,
    num_iterations=10,
    num_shifts_block=5,
    smoothing_sigma=0.5,
    kriging_sigma=1,
    kriging_p=2,
    kriging_d=2,
):
    """

    Parameters
    ----------

    spikecounts_hist_images : np.ndarray
        Spike count histogram images (num_temporal_bins, num_spatial_bins, num_amps_bins)
    non_rigid_windows : list, default: None
        If num_non_rigid_windows > 1, this argument is required and it is a list of
        windows to taper spatial bins in different blocks
    num_shifts_global : int, default: 15
        Number of spatial bin shifts to consider for global alignment
    num_iterations : int, default: 10
        Number of iterations for global alignment procedure
    num_shifts_block : int, default: 5
        Number of spatial bin shifts to consider for non-rigid alignment
    smoothing_sigma : float, default: 0.5
        Sigma of gaussian for covariance matrices smoothing
    kriging_sigma : float, default: 1
        sigma parameter for kriging_kernel function
    kriging_p : float, default: 2
        p parameter for kriging_kernel function
    kriging_d : float, default: 2
        d parameter for kriging_kernel function

    Returns
    -------
    optimal_shift_indices
        Optimal shifts for each temporal and spatial bin (num_temporal_bins, num_non_rigid_windows)
    target_spikecount_hist
        Target histogram used for alignment (num_spatial_bins, num_amps_bins)
    """
    from scipy.ndimage import gaussian_filter, gaussian_filter1d

    # F is y bins by amp bins by batches
    # ysamp are the coordinates of the y bins in um
    spikecounts_hist_images = spikecounts_hist_images.swapaxes(0, 1).swapaxes(1, 2)
    num_temporal_bins = spikecounts_hist_images.shape[2]

    # look up and down this many y bins to find best alignment
    shift_covs = np.zeros((2 * num_shifts_global + 1, num_temporal_bins))
    shifts = np.arange(-num_shifts_global, num_shifts_global + 1)

    # mean subtraction to compute covariance
    F = spikecounts_hist_images
    Fg = F - np.mean(F, axis=0)

    # initialize the target "frame" for alignment with a single sample
    # here we removed min(299, ...)
    F0 = Fg[:, :, np.floor(num_temporal_bins / 2).astype("int") - 1]
    F0 = F0[:, :, np.newaxis]

    # first we do rigid registration by integer shifts
    # everything is iteratively aligned until most of the shifts become 0.
    best_shifts = np.zeros((num_iterations, num_temporal_bins))
    for iteration in range(num_iterations):
        for t, shift in enumerate(shifts):
            # for each NEW potential shift, estimate covariance
            Fs = np.roll(Fg, shift, axis=0)
            shift_covs[t, :] = np.mean(Fs * F0, axis=(0, 1))
        if iteration + 1 < num_iterations:
            # estimate the best shifts
            imax = np.argmax(shift_covs, axis=0)
            # align the data by these integer shifts
            for t, shift in enumerate(shifts):
                ibest = imax == t
                Fg[:, :, ibest] = np.roll(Fg[:, :, ibest], shift, axis=0)
                best_shifts[iteration, ibest] = shift
            # new target frame based on our current best alignment
            F0 = np.mean(Fg, axis=2)[:, :, np.newaxis]
    target_spikecount_hist = F0[:, :, 0]

    # now we figure out how to split the probe into nblocks pieces
    # if len(non_rigid_windows) = 1, then we're doing rigid registration
    num_non_rigid_windows = len(non_rigid_windows)

    # for each small block, we only look up and down this many samples to find
    # nonrigid shift
    shifts_block = np.arange(-num_shifts_block, num_shifts_block + 1)
    num_shifts = len(shifts_block)
    shift_covs_block = np.zeros((2 * num_shifts_block + 1, num_temporal_bins, num_non_rigid_windows))

    # this part determines the up/down covariance for each block without
    # shifting anything
    for window_index in range(num_non_rigid_windows):
        win = non_rigid_windows[window_index]
        window_slice = np.flatnonzero(win > 1e-5)
        window_slice = slice(window_slice[0], window_slice[-1])
        tiled_window = win[window_slice, np.newaxis, np.newaxis]
        Ftaper = Fg[window_slice] * np.tile(tiled_window, (1,) + Fg.shape[1:])
        for t, shift in enumerate(shifts_block):
            Fs = np.roll(Ftaper, shift, axis=0)
            F0taper = F0[window_slice] * np.tile(tiled_window, (1,) + F0.shape[1:])
            shift_covs_block[t, :, window_index] = np.mean(Fs * F0taper, axis=(0, 1))

    # gaussian smoothing:
    # here the original my_conv2_cpu is substituted with scipy gaussian_filters
    shift_covs_block_smooth = shift_covs_block.copy()
    shifts_block_up = np.linspace(-num_shifts_block, num_shifts_block, (2 * num_shifts_block * 10) + 1)
    # 1. 2d smoothing over time and blocks dimensions for each shift
    for shift_index in range(num_shifts):
        shift_covs_block_smooth[shift_index, :, :] = gaussian_filter(
            shift_covs_block_smooth[shift_index, :, :], smoothing_sigma
        )  # some additional smoothing for robustness, across all dimensions
    # 2. 1d smoothing over shift dimension for each spatial block
    for window_index in range(num_non_rigid_windows):
        shift_covs_block_smooth[:, :, window_index] = gaussian_filter1d(
            shift_covs_block_smooth[:, :, window_index], smoothing_sigma, axis=0
        )  # some additional smoothing for robustness, across all dimensions
    upsample_kernel = kriging_kernel(
        shifts_block[:, np.newaxis], shifts_block_up[:, np.newaxis], sigma=kriging_sigma, p=kriging_p, d=kriging_d
    )

    optimal_shift_indices = np.zeros((num_temporal_bins, num_non_rigid_windows))
    for window_index in range(num_non_rigid_windows):
        # using the upsampling kernel K, get the upsampled cross-correlation
        # curves
        upsampled_cov = upsample_kernel.T @ shift_covs_block_smooth[:, :, window_index]

        # find the max index of these curves
        imax = np.argmax(upsampled_cov, axis=0)

        # add the value of the shift to the last row of the matrix of shifts
        # (as if it was the last iteration of the main rigid loop )
        best_shifts[num_iterations - 1, :] = shifts_block_up[imax]

        # the sum of all the shifts equals the final shifts for this block
        optimal_shift_indices[:, window_index] = np.sum(best_shifts, axis=0)

    return optimal_shift_indices, target_spikecount_hist, shift_covs_block


def kriging_kernel(source_location, target_location, sigma=1, p=2, d=2):
    from scipy.spatial.distance import cdist

    dist_xy = cdist(source_location, target_location, metric="euclidean")
    K = np.exp(-((dist_xy / sigma) ** p) / d)
    return K
