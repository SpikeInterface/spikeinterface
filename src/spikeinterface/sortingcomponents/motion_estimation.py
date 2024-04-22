from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm, trange
import scipy.interpolate

try:
    import torch
    import torch.nn.functional as F

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from .tools import make_multi_method_doc


def estimate_motion(
    recording,
    peaks,
    peak_locations,
    direction="y",
    bin_duration_s=10.0,
    bin_um=10.0,
    margin_um=0.0,
    rigid=False,
    win_shape="gaussian",
    win_step_um=50.0,
    win_sigma_um=150.0,
    post_clean=False,
    speed_threshold=30,
    sigma_smooth_s=None,
    method="decentralized",
    output_extra_check=False,
    progress_bar=False,
    upsample_to_histogram_bin=False,
    verbose=False,
    **method_kwargs,
):
    """
    Estimate motion for given peaks and after their localization.

    Note that the way you detect peak locations (center of mass/monopolar triangulation) have an impact on the result.

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor
    peaks: numpy array
        Peak vector (complex dtype)
    peak_locations: numpy array
        Complex dtype with "x", "y", "z" fields

    {method_doc}

    **histogram section**

    direction: "x" | "y" | "z", default: "y"
        Dimension on which the motion is estimated
    bin_duration_s: float, default: 10
        Bin duration in second
    bin_um: float, default: 10
        Spatial bin size in micrometers
    margin_um: float, default: 0
        Margin in um to exclude from histogram estimation and
        non-rigid smoothing functions to avoid edge effects.
        Positive margin extrapolate out of the probe the motion.
        Negative margin crop the motion on the border

    **non-rigid section**

    rigid : bool, default: False
        Compute rigid (one motion for the entire probe) or non rigid motion
        Rigid computation is equivalent to non-rigid with only one window with rectangular shape.
    win_shape: "gaussian" | "rect" | "triangle", default: "gaussian"
        The shape of the windows for non rigid.
        When rigid this is force to "rect"
    win_step_um: float, default: 50
        Step deteween window
    win_sigma_um: float, default: 150
        Sigma of the gaussian window

    **motion cleaning section**

    post_clean: bool, default: False
        Apply some post cleaning to motion matrix or not
    speed_threshold: float default: 30.
        Detect to fast motion bump and remove then with interpolation
    sigma_smooth_s: None or float
        Optional smooting gaussian kernel when not None

    output_extra_check: bool, default: False
        If True then return an extra dict that contains variables
        to check intermediate steps (motion_histogram, non_rigid_windows, pairwise_displacement)
    upsample_to_histogram_bin: bool or None, default: False
        If True then upsample the returned motion array to the number of depth bins specified by bin_um.
        When None:
          * for non rigid case: then automatically True
          * for rigid (non_rigid_kwargs=None): automatically False
        This feature is in fact a bad idea and the interpolation should be done outside using better methods
    progress_bar: bool, default: False
        Display progress bar or not
    verbose: bool, default: False
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
    spatial_bins: numpy.array 1d
        Windows center.
        spatial_bins.shape[0] == motion.shape[1]
        If rigid then spatial_bins.shape[0] == 1
    extra_check: dict
        Optional output if `output_extra_check=True`
        This dict contain histogram, pairwise_displacement usefull for ploting.
    """
    # TODO handle multi segment one day
    assert recording.get_num_segments() == 1

    if output_extra_check:
        extra_check = {}
    else:
        extra_check = None

    # contact positions
    probe = recording.get_probe()
    dim = ["x", "y", "z"].index(direction)
    contact_pos = probe.contact_positions[:, dim]

    # spatial bins
    spatial_bin_edges = get_spatial_bin_edges(recording, direction, margin_um, bin_um)

    # get windows
    non_rigid_windows, non_rigid_window_centers = get_windows(
        rigid, bin_um, contact_pos, spatial_bin_edges, margin_um, win_step_um, win_sigma_um, win_shape
    )

    if output_extra_check:
        extra_check["non_rigid_windows"] = non_rigid_windows

    # run method
    method_class = estimate_motion_methods[method]
    motion, temporal_bins = method_class.run(
        recording,
        peaks,
        peak_locations,
        direction,
        bin_duration_s,
        bin_um,
        spatial_bin_edges,
        non_rigid_windows,
        verbose,
        progress_bar,
        extra_check,
        **method_kwargs,
    )

    # replace nan by zeros
    motion[np.isnan(motion)] = 0

    if post_clean:
        motion = clean_motion_vector(
            motion, temporal_bins, bin_duration_s, speed_threshold=speed_threshold, sigma_smooth_s=sigma_smooth_s
        )

    if upsample_to_histogram_bin is None:
        upsample_to_histogram_bin = not rigid
    if upsample_to_histogram_bin:
        extra_check["motion"] = motion
        extra_check["non_rigid_window_centers"] = non_rigid_window_centers
        non_rigid_windows = np.array(non_rigid_windows)
        non_rigid_windows /= non_rigid_windows.sum(axis=0, keepdims=True)
        non_rigid_window_centers = spatial_bin_edges[:-1] + bin_um / 2
        motion = motion @ non_rigid_windows

    if output_extra_check:
        return motion, temporal_bins, non_rigid_window_centers, extra_check
    else:
        return motion, temporal_bins, non_rigid_window_centers


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
    params_doc = """
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
    convergence_method: "lsmr" | "lsqr_robust" | "gradient_descent", default: "lsqr_robust"
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
        bin_duration_s,
        bin_um,
        spatial_bin_edges,
        non_rigid_windows,
        verbose,
        progress_bar,
        extra_check,
        histogram_depth_smooth_um=None,
        histogram_time_smooth_s=None,
        pairwise_displacement_method="conv",
        max_displacement_um=100.0,
        weight_scale="linear",
        error_sigma=0.2,
        conv_engine=None,
        torch_device=None,
        batch_size=1,
        corr_threshold=0.0,
        time_horizon_s=None,
        convergence_method="lsqr_robust",
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
        # use torch if installed
        if conv_engine is None:
            conv_engine = "torch" if HAVE_TORCH else "numpy"

        # make 2D histogram raster
        if verbose:
            print("Computing motion histogram")

        motion_histogram, temporal_hist_bin_edges, spatial_hist_bin_edges = make_2d_motion_histogram(
            recording,
            peaks,
            peak_locations,
            direction=direction,
            bin_duration_s=bin_duration_s,
            spatial_bin_edges=spatial_bin_edges,
            weight_with_amplitude=weight_with_amplitude,
        )

        if histogram_depth_smooth_um is not None:
            bins = np.arange(motion_histogram.shape[1]) * bin_um
            bins = bins - np.mean(bins)
            smooth_kernel = np.exp(-(bins**2) / (2 * histogram_depth_smooth_um**2))
            smooth_kernel /= np.sum(smooth_kernel)
            motion_histogram = scipy.signal.fftconvolve(motion_histogram, smooth_kernel[None, :], mode="same", axes=1)

        if histogram_time_smooth_s is not None:
            bins = np.arange(motion_histogram.shape[0]) * bin_duration_s
            bins = bins - np.mean(bins)
            smooth_kernel = np.exp(-(bins**2) / (2 * histogram_time_smooth_s**2))
            smooth_kernel /= np.sum(smooth_kernel)
            motion_histogram = scipy.signal.fftconvolve(motion_histogram, smooth_kernel[:, None], mode="same", axes=0)

        if extra_check is not None:
            extra_check["motion_histogram"] = motion_histogram
            extra_check["pairwise_displacement_list"] = []
            extra_check["temporal_hist_bin_edges"] = temporal_hist_bin_edges
            extra_check["spatial_hist_bin_edges"] = spatial_hist_bin_edges

        # temporal bins are bin center
        temporal_bins = temporal_hist_bin_edges[:-1] + bin_duration_s // 2.0

        motion = np.zeros((temporal_bins.size, len(non_rigid_windows)), dtype=np.float64)
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
                bin_duration_s=bin_duration_s,
                progress_bar=False,
            )

            if spatial_prior:
                all_pairwise_displacements[i] = pairwise_displacement
                all_pairwise_displacement_weights[i] = pairwise_displacement_weight

            if extra_check is not None:
                extra_check["pairwise_displacement_list"].append(pairwise_displacement)

            if verbose:
                print(f"Computing global displacement: {i + 1} / {len(non_rigid_windows)}")

            # TODO: if spatial_prior, do this after the loop
            if not spatial_prior:
                motion[:, i] = compute_global_displacement(
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
            motion = compute_global_displacement(
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
                    motion[:, i + 1] -= np.median(motion[:, i + 1] - motion[:, i])

        # try to avoid constant offset
        # let the user choose how to do this. here are some ideas.
        # (one can also -= their own number on the result of this function.)
        if reference_displacement == "mean":
            motion -= motion.mean()
        elif reference_displacement == "median":
            motion -= np.median(motion)
        elif reference_displacement == "time":
            # reference the motion to 0 at a specific time, independently in each window
            reference_displacement_bin = np.digitize(reference_displacement_time_s, temporal_hist_bin_edges) - 1
            motion -= motion[reference_displacement_bin, :]
        elif reference_displacement == "mode_search":
            # just a sketch of an idea
            # things might want to change, should have a configurable bin size,
            # should use a call to histogram instead of the loop, ...
            step_size = 0.1
            round_mode = np.round  # floor?
            best_ref = np.median(motion)
            max_zeros = np.sum(round_mode(motion - best_ref) == 0)
            for ref in np.arange(np.floor(motion.min()), np.ceil(motion.max()), step_size):
                n_zeros = np.sum(round_mode(motion - ref) == 0)
                if n_zeros > max_zeros:
                    max_zeros = n_zeros
                    best_ref = ref
            motion -= best_ref

        return motion, temporal_bins


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
    params_doc = """
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
        bin_duration_s,
        bin_um,
        spatial_bin_edges,
        non_rigid_windows,
        verbose,
        progress_bar,
        extra_check,
        num_amp_bins=20,
        num_shifts_global=15,
        num_iterations=10,
        num_shifts_block=5,
        smoothing_sigma=0.5,
        kriging_sigma=1,
        kriging_p=2,
        kriging_d=2,
    ):
        # make a 3D histogram
        motion_histograms, temporal_hist_bin_edges, spatial_hist_bin_edges = make_3d_motion_histograms(
            recording,
            peaks,
            peak_locations,
            direction=direction,
            num_amp_bins=num_amp_bins,
            bin_duration_s=bin_duration_s,
            spatial_bin_edges=spatial_bin_edges,
        )
        # temporal bins are bin center
        temporal_bins = temporal_hist_bin_edges[:-1] + bin_duration_s // 2.0

        # do alignment
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
        motion = -(shift_indices * bin_um)

        if extra_check:
            extra_check["motion_histograms"] = motion_histograms
            extra_check["target_histogram"] = target_histogram
            extra_check["shift_covs_block"] = shift_covs_block
            extra_check["temporal_hist_bin_edges"] = temporal_hist_bin_edges
            extra_check["spatial_hist_bin_edges"] = spatial_hist_bin_edges

        return motion, temporal_bins


_methods_list = [DecentralizedRegistration, IterativeTemplateRegistration]
estimate_motion_methods = {m.name: m for m in _methods_list}
method_doc = make_multi_method_doc(_methods_list)
estimate_motion.__doc__ = estimate_motion.__doc__.format(method_doc=method_doc)


def get_spatial_bin_edges(recording, direction, margin_um, bin_um):
    # contact along one axis
    probe = recording.get_probe()
    dim = ["x", "y", "z"].index(direction)
    contact_pos = probe.contact_positions[:, dim]

    min_ = np.min(contact_pos) - margin_um
    max_ = np.max(contact_pos) + margin_um
    spatial_bins = np.arange(min_, max_ + bin_um, bin_um)

    return spatial_bins


def get_windows(rigid, bin_um, contact_pos, spatial_bin_edges, margin_um, win_step_um, win_sigma_um, win_shape):
    """
    Generate spatial windows (taper) for non-rigid motion.
    For rigid motion, this is equivalent to have one unique rectangular window that covers the entire probe.
    The windowing can be gaussian or rectangular.

    Parameters
    ----------
    rigid : bool
        If True, returns a single rectangular window
    bin_um : float
        Spatial bin size in um
    contact_pos : np.ndarray
        Position of electrodes (num_channels, 2)
    spatial_bin_edges : np.array
        The pre-computed spatial bin edges
    margin_um : float
        The margin to extend (if positive) or shrink (if negative) the probe dimension to compute windows.=
    win_step_um : float
        The steps at which windows are defined
    win_sigma_um : float
        Sigma of gaussian window (if win_shape is gaussian)
    win_shape : float
        "gaussian" | "rect"

    Returns
    -------
    non_rigid_windows : list of 1D arrays
        The scaling for each window. Each element has num_spatial_bins values
    non_rigid_window_centers: 1D np.array
        The center of each window

    Notes
    -----
    Note that kilosort2.5 uses overlaping rectangular windows.
    Here by default we use gaussian window.

    """
    bin_centers = spatial_bin_edges[:-1] + bin_um / 2.0
    n = bin_centers.size

    if rigid:
        # win_shape = 'rect' is forced
        non_rigid_windows = [np.ones(n, dtype="float64")]
        middle = (spatial_bin_edges[0] + spatial_bin_edges[-1]) / 2.0
        non_rigid_window_centers = np.array([middle])
    else:
        assert win_sigma_um >= win_step_um, f"win_sigma_um too low {win_sigma_um} compared to win_step_um {win_step_um}"

        min_ = np.min(contact_pos) - margin_um
        max_ = np.max(contact_pos) + margin_um
        num_non_rigid_windows = int((max_ - min_) // win_step_um)
        border = ((max_ - min_) % win_step_um) / 2
        non_rigid_window_centers = np.arange(num_non_rigid_windows + 1) * win_step_um + min_ + border
        non_rigid_windows = []

        for win_center in non_rigid_window_centers:
            if win_shape == "gaussian":
                win = np.exp(-((bin_centers - win_center) ** 2) / (2 * win_sigma_um**2))
            elif win_shape == "rect":
                win = np.abs(bin_centers - win_center) < (win_sigma_um / 2.0)
                win = win.astype("float64")
            elif win_shape == "triangle":
                center_dist = np.abs(bin_centers - win_center)
                in_window = center_dist <= (win_sigma_um / 2.0)
                win = -center_dist
                win[~in_window] = 0
                win[in_window] -= win[in_window].min()
                win[in_window] /= win[in_window].max()

            non_rigid_windows.append(win)

    return non_rigid_windows, non_rigid_window_centers


def make_2d_motion_histogram(
    recording,
    peaks,
    peak_locations,
    weight_with_amplitude=False,
    direction="y",
    bin_duration_s=1.0,
    bin_um=2.0,
    margin_um=50,
    spatial_bin_edges=None,
):
    """
    Generate 2d motion histogram in depth and time.

    Parameters
    ----------
    recording : BaseRecording
        The input recording
    peaks : np.array
        The peaks array
    peak_locations : np.array
        Array with peak locations
    weight_with_amplitude : bool, default: False
        If True, motion histogram is weighted by amplitudes
    direction : "x" | "y" | "z", default: "y"
        The depth direction
    bin_duration_s : float, default: 1.0
        The temporal bin duration in s
    bin_um : float, default: 2.0
        The spatial bin size in um. Ignored if spatial_bin_edges is given.
    margin_um : float, default: 50
        The margin to add to the minimum and maximum positions before spatial binning.
        Ignored if spatial_bin_edges is given.
    spatial_bin_edges : np.array, default: None
        The pre-computed spatial bin edges

    Returns
    -------
    motion_histogram
        2d np.array with motion histogram (num_temporal_bins, num_spatial_bins)
    temporal_bin_edges
        1d array with temporal bin edges
    spatial_bin_edges
        1d array with spatial bin edges
    """
    fs = recording.get_sampling_frequency()
    num_samples = recording.get_num_samples(segment_index=0)
    bin_sample_size = int(bin_duration_s * fs)
    sample_bin_edges = np.arange(0, num_samples + bin_sample_size, bin_sample_size)
    temporal_bin_edges = sample_bin_edges / fs
    if spatial_bin_edges is None:
        spatial_bin_edges = get_spatial_bin_edges(recording, direction, margin_um, bin_um)

    arr = np.zeros((peaks.size, 2), dtype="float64")
    arr[:, 0] = peaks["sample_index"]
    arr[:, 1] = peak_locations[direction]

    if weight_with_amplitude:
        weights = np.abs(peaks["amplitude"])
    else:
        weights = None

    motion_histogram, edges = np.histogramdd(arr, bins=(sample_bin_edges, spatial_bin_edges), weights=weights)

    # average amplitude in each bin
    if weight_with_amplitude:
        bin_counts, _ = np.histogramdd(arr, bins=(sample_bin_edges, spatial_bin_edges))
        bin_counts[bin_counts == 0] = 1
        motion_histogram = motion_histogram / bin_counts

    return motion_histogram, temporal_bin_edges, spatial_bin_edges


def make_3d_motion_histograms(
    recording,
    peaks,
    peak_locations,
    direction="y",
    bin_duration_s=1.0,
    bin_um=2.0,
    margin_um=50,
    num_amp_bins=20,
    log_transform=True,
    spatial_bin_edges=None,
):
    """
    Generate 3d motion histograms in depth, amplitude, and time.
    This is used by the "iterative_template_registration" (Kilosort2.5) method.


    Parameters
    ----------
    recording : BaseRecording
        The input recording
    peaks : np.array
        The peaks array
    peak_locations : np.array
        Array with peak locations
    direction : "x" | "y" | "z", default: "y"
        The depth direction
    bin_duration_s : float, default: 1.0
        The temporal bin duration in s.
    bin_um : float, default: 2.0
        The spatial bin size in um. Ignored if spatial_bin_edges is given.
    margin_um : float, default: 50
        The margin to add to the minimum and maximum positions before spatial binning.
        Ignored if spatial_bin_edges is given.
    log_transform : bool, default: True
        If True, histograms are log-transformed
    spatial_bin_edges : np.array, default: None
        The pre-computed spatial bin edges

    Returns
    -------
    motion_histograms
        3d np.array with motion histogram (num_temporal_bins, num_spatial_bins, num_amp_bins)
    temporal_bin_edges
        1d array with temporal bin edges
    spatial_bin_edges
        1d array with spatial bin edges
    """
    fs = recording.get_sampling_frequency()
    num_samples = recording.get_num_samples(segment_index=0)
    bin_sample_size = int(bin_duration_s * fs)
    sample_bin_edges = np.arange(0, num_samples + bin_sample_size, bin_sample_size)
    temporal_bin_edges = sample_bin_edges / fs
    if spatial_bin_edges is None:
        spatial_bin_edges = get_spatial_bin_edges(recording, direction, margin_um, bin_um)

    # pre-compute abs amplitude and ranges for scaling
    amplitude_bin_edges = np.linspace(0, 1, num_amp_bins + 1)
    abs_peaks = np.abs(peaks["amplitude"])
    max_peak_amp = np.max(abs_peaks)
    min_peak_amp = np.min(abs_peaks)
    # log amplitudes and scale between 0-1
    abs_peaks_log_norm = (np.log10(abs_peaks) - np.log10(min_peak_amp)) / (
        np.log10(max_peak_amp) - np.log10(min_peak_amp)
    )

    arr = np.zeros((peaks.size, 3), dtype="float64")
    arr[:, 0] = peaks["sample_index"]
    arr[:, 1] = peak_locations[direction]
    arr[:, 2] = abs_peaks_log_norm

    motion_histograms, edges = np.histogramdd(
        arr,
        bins=(
            sample_bin_edges,
            spatial_bin_edges,
            amplitude_bin_edges,
        ),
    )

    if log_transform:
        motion_histograms = np.log2(1 + motion_histograms)

    return motion_histograms, temporal_bin_edges, spatial_bin_edges


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
    bin_duration_s=None,
    progress_bar=False,
    window=None,
):
    """
    Compute pairwise displacement
    """
    from scipy import sparse
    from scipy import linalg

    assert conv_engine in ("torch", "numpy"), f"'conv_engine' must be 'torch' or 'numpy'"
    size = motion_hist.shape[0]
    pairwise_displacement = np.zeros((size, size), dtype="float32")

    if time_horizon_s is not None:
        band_width = int(np.ceil(time_horizon_s / bin_duration_s))
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
    if convergence_method == "gradient_descent":
        size = pairwise_displacement.shape[0]
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
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import lsqr
        from scipy.stats import zscore

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
        from scipy.stats import zscore

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


def normxcorr1d(
    template,
    x,
    weights=None,
    centered=True,
    normalized=True,
    padding="same",
    conv_engine="torch",
):
    """normxcorr1d: Normalized cross-correlation, optionally weighted

    The API is like torch's F.conv1d, except I have accidentally
    changed the position of input/weights -- template acts like weights,
    and x acts like input.

    Returns the cross-correlation of `template` and `x` at spatial lags
    determined by `mode`. Useful for estimating the location of `template`
    within `x`.

    This might not be the most efficient implementation -- ideas welcome.
    It uses a direct convolutional translation of the formula
        corr = (E[XY] - EX EY) / sqrt(var X * var Y)

    This also supports weights! In that case, the usual adaptation of
    the above formula is made to the weighted case -- and all of the
    normalizations are done per block in the same way.

    Parameters
    ----------
    template : tensor, shape (num_templates, length)
        The reference template signal
    x : tensor, 1d shape (length,) or 2d shape (num_inputs, length)
        The signal in which to find `template`
    weights : tensor, shape (length,)
        Will use weighted means, variances, covariances if supplied.
    centered : bool
        If true, means will be subtracted (per weighted patch).
    normalized : bool
        If true, normalize by the variance (per weighted patch).
    padding : str
        How far to look? if unset, we'll use half the length
    conv_engine : string, one of "torch", "numpy"
        What library to use for computing cross-correlations.
        If numpy, falls back to the scipy correlate function.

    Returns
    -------
    corr : tensor
    """
    if conv_engine == "torch":
        assert HAVE_TORCH
        conv1d = F.conv1d
        npx = torch
    elif conv_engine == "numpy":
        conv1d = scipy_conv1d
        npx = np
    else:
        raise ValueError(f"Unknown conv_engine {conv_engine}. 'conv_engine' must be 'torch' or 'numpy'")

    x = npx.atleast_2d(x)
    num_templates, length = template.shape
    num_inputs, length_ = template.shape
    assert length == length_

    # generalize over weighted / unweighted case
    device_kw = {} if conv_engine == "numpy" else dict(device=x.device)
    ones = npx.ones((1, 1, length), dtype=x.dtype, **device_kw)
    no_weights = weights is None
    if no_weights:
        weights = ones
        wt = template[:, None, :]
    else:
        assert weights.shape == (length,)
        weights = weights[None, None]
        wt = template[:, None, :] * weights

    # conv1d valid rule:
    # (B,1,L),(O,1,L)->(B,O,L)

    # compute expectations
    # how many points in each window? seems necessary to normalize
    # for numerical stability.
    N = conv1d(ones, weights, padding=padding)
    if centered:
        Et = conv1d(ones, wt, padding=padding)
        Et /= N
        Ex = conv1d(x[:, None, :], weights, padding=padding)
        Ex /= N

    # compute (weighted) covariance
    # important: the formula E[XY] - EX EY is well-suited here,
    # because the means are naturally subtracted correctly
    # patch-wise. you couldn't pre-subtract them!
    cov = conv1d(x[:, None, :], wt, padding=padding)
    cov /= N
    if centered:
        cov -= Ex * Et

    # compute variances for denominator, using var X = E[X^2] - (EX)^2
    if normalized:
        var_template = conv1d(ones, wt * template[:, None, :], padding=padding)
        var_template /= N
        var_x = conv1d(npx.square(x)[:, None, :], weights, padding=padding)
        var_x /= N
        if centered:
            var_template -= npx.square(Et)
            var_x -= npx.square(Ex)

    # now find the final normxcorr
    corr = cov  # renaming for clarity
    if normalized:
        corr /= npx.sqrt(var_x)
        corr /= npx.sqrt(var_template)
        # get rid of NaNs in zero-variance areas
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
        input = np.pad(input, [*[(0, 0)] * (input.ndim - 1), (padding, padding)])
        length_out = length - (kernel_size - 1) + 2 * padding
    else:
        raise ValueError(f"Unknown 'padding' value of {padding}, 'padding' must be 'same', 'valid' or an integer")

    output = np.zeros((n, c_out, length_out), dtype=input.dtype)
    for m in range(n):
        for c in range(c_out):
            output[m, c] = correlate(input[m, 0], weights[c, 0], mode=mode)

    return output


def clean_motion_vector(motion, temporal_bins, bin_duration_s, speed_threshold=30, sigma_smooth_s=None):
    """
    Simple machinery to remove spurious fast bump in the motion vector.
    Also can applyt a smoothing.


    Arguments
    ---------
    motion: numpy array 2d
        Motion estimate in um.
    temporal_bins: numpy.array 1d
        temporal bins (bin center)
    bin_duration_s: float
        bin duration in second
    speed_threshold: float (units um/s)
        Maximum speed treshold between 2 bins allowed.
        Expressed in um/s
    sigma_smooth_s: None or float
        Optional smooting gaussian kernel.

    Returns
    -------
    corr : tensor


    """
    motion_clean = motion.copy()

    # STEP 1 :
    #  * detect long plateau or small peak corssing the speed thresh
    #  * mask the period and interpolate
    for i in range(motion.shape[1]):
        one_motion = motion_clean[:, i]
        speed = np.diff(one_motion, axis=0) / bin_duration_s
        (inds,) = np.nonzero(np.abs(speed) > speed_threshold)
        inds += 1
        if inds.size % 2 == 1:
            # more compicated case: number of of inds is odd must remove first or last
            # take the smallest duration sum
            inds0 = inds[:-1]
            inds1 = inds[1:]
            d0 = np.sum(inds0[1::2] - inds0[::2])
            d1 = np.sum(inds1[1::2] - inds1[::2])
            if d0 < d1:
                inds = inds0
        mask = np.ones(motion_clean.shape[0], dtype="bool")
        for i in range(inds.size // 2):
            mask[inds[i * 2] : inds[i * 2 + 1]] = False
        f = scipy.interpolate.interp1d(temporal_bins[mask], one_motion[mask])
        one_motion[~mask] = f(temporal_bins[~mask])

    # Step 2 : gaussian smooth
    if sigma_smooth_s is not None:
        half_size = motion_clean.shape[0] // 2
        if motion_clean.shape[0] % 2 == 0:
            # take care of the shift
            bins = (np.arange(motion_clean.shape[0]) - half_size + 1) * bin_duration_s
        else:
            bins = (np.arange(motion_clean.shape[0]) - half_size) * bin_duration_s
        smooth_kernel = np.exp(-(bins**2) / (2 * sigma_smooth_s**2))
        smooth_kernel /= np.sum(smooth_kernel)
        smooth_kernel = smooth_kernel[:, None]
        motion_clean = scipy.signal.fftconvolve(motion_clean, smooth_kernel, mode="same", axes=0)

    return motion_clean


def kriging_kernel(source_location, target_location, sigma=1, p=2, d=2):
    from scipy.spatial.distance import cdist

    dist_xy = cdist(source_location, target_location, metric="euclidean")
    K = np.exp(-((dist_xy / sigma) ** p) / d)
    return K
