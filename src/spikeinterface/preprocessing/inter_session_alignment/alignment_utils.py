from spikeinterface import BaseRecording
import numpy as np

from spikeinterface.preprocessing import center
from spikeinterface.sortingcomponents.motion.motion_utils import make_2d_motion_histogram
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from spikeinterface.sortingcomponents.motion.iterative_template import kriging_kernel

# #############################################################################
# Get Histograms
# #############################################################################


def get_activity_histogram(
    recording: BaseRecording,
    peaks: np.ndarray,
    peak_locations: np.ndarray,
    spatial_bin_edges: np.ndarray,
    log_scale: bool,
    bin_s: float | None,
    depth_smooth_um: float | None,
    scale_to_hz: bool = False,
):
    """
    Generate a 2D activity histogram for the session. Wraps the underlying
    spikeinterface function with some adjustments for scaling to time and
    log transform.

    Parameters
    ----------

    recording: BaseRecording,
        A SpikeInterface recording object.
    peaks: np.ndarray,
        A SpikeInterface `peaks` array.
    peak_locations: np.ndarray,
        A SpikeInterface `peak_locations` array.
    spatial_bin_edges: np.ndarray,
        A (1 x n_bins + 1) array of spatial (probe y dimension) bin edges.
    log_scale: bool,
        If `True`, histogram is log scaled.
    bin_s | None: float,
        If `None`, a single histogram will be generated from all session
        peaks. Otherwise, multiple histograms will be generated, one for
        each time bin.
    depth_smooth_um: float | None
        If not `None`, smooth the histogram across the spatial
        axis. see `make_2d_motion_histogram()` for details.

    TODO
    ----
    - assumes 1-segment recording
    - ask Sam whether it makes sense to integrate this function with `make_2d_motion_histogram`.
    """
    activity_histogram, temporal_bin_edges, generated_spatial_bin_edges = make_2d_motion_histogram(
        recording,
        peaks,
        peak_locations,
        weight_with_amplitude=False,
        direction="y",
        bin_s=(bin_s if bin_s is not None else recording.get_duration(segment_index=0)),
        bin_um=None,
        hist_margin_um=None,
        spatial_bin_edges=spatial_bin_edges,
        depth_smooth_um=depth_smooth_um,
    )
    assert np.array_equal(generated_spatial_bin_edges, spatial_bin_edges), "TODO: remove soon after testing"

    temporal_bin_centers = get_bin_centers(temporal_bin_edges)
    spatial_bin_centers = get_bin_centers(spatial_bin_edges)

    if scale_to_hz:
        if bin_s is None:
            scaler = 1 / recording.get_duration()
        else:
            scaler = 1 / np.diff(temporal_bin_edges)[:, np.newaxis]

        activity_histogram *= scaler

    if log_scale:
        activity_histogram = np.log10(1 + activity_histogram)  # TODO: make_2d_motion_histogram uses log2

    return activity_histogram, temporal_bin_centers, spatial_bin_centers


def get_bin_centers(bin_edges):
    return (bin_edges[1:] + bin_edges[:-1]) / 2


def estimate_chunk_size(scaled_activity_histogram):
    """
    Estimate a chunk size based on the firing rate. Intuitively, we
    want longer chunk size to better estimate low firing rates. The
    estimation computes a summary of the the firing rates for the session
    by taking the value 25% of the max of the activity histogram.

    Then, the chunk size that will accurately estimate this firing rate
    within 90% accuracy, 90% of the time based on assumption of Poisson
    firing (based on CLT) is computed.

    Parameters
    ----------

    scaled_activity_histogram: np.ndarray
        The activity histogram scaled to firing rate in Hz.

    TODO
    ----
    - make the details available.
    """
    print("scaled max", np.max(scaled_activity_histogram))

    firing_rate = np.max(scaled_activity_histogram) * 0.25

    lambda_hat_s = firing_rate
    range_percent = 0.1
    confidence_z = 1.645  # 90% of samples in the normal distribution
    e = lambda_hat_s * range_percent

    t = lambda_hat_s / (e / confidence_z) ** 2

    print(
        f"Chunked histogram window size of: {t}s estimated "
        f"for firing rate (25% of histogram peak) of {lambda_hat_s}"
    )

    return 10


# #############################################################################
# Chunked Histogram estimation methods
# #############################################################################
# Given a set off chunked_session_histograms (num time chunks x num spatial bins)
# take the summary statistic over the time axis.


def get_chunked_hist_mean(chunked_session_histograms):

    mean_hist = np.mean(chunked_session_histograms, axis=0)

    std = np.std(chunked_session_histograms, axis=0, ddof=0)

    return mean_hist, std


def get_chunked_hist_median(chunked_session_histograms):

    median_hist = np.median(chunked_session_histograms, axis=0)

    quartile_1 = np.percentile(chunked_session_histograms, 25, axis=0)
    quartile_3 = np.percentile(chunked_session_histograms, 75, axis=0)

    iqr = quartile_3 - quartile_1

    return median_hist, iqr


def get_chunked_hist_supremum(chunked_session_histograms):

    max_hist = np.max(chunked_session_histograms, axis=0)

    min_hist = np.min(chunked_session_histograms, axis=0)

    scaled_range = (max_hist - min_hist) / (max_hist + 1e-12)

    return max_hist, scaled_range


def get_chunked_hist_poisson_estimate(chunked_session_histograms):
    """
    Make a MLE estimate of the most likely value for each bin
    given the assumption of Poisson firing. Turns out this is
    basically identical to the mean :'D.

    Keeping for now as opportunity to add prior or do some outlier
    removal per bin. But if not useful, deprecate in future.
    """

    def obj_fun(lambda_, m, sum_k):
        return -(sum_k * np.log(lambda_) - m * lambda_)

    poisson_estimate = np.zeros(chunked_session_histograms.shape[1])
    std_devs = []
    for i in range(chunked_session_histograms.shape[1]):

        ks = chunked_session_histograms[:, i]

        std_devs.append(np.std(ks))
        m = ks.shape
        sum_k = np.sum(ks)

        poisson_estimate[i] = minimize(obj_fun, 0.5, (m, sum_k), bounds=((1e-10, np.inf),)).x

    raise NotImplementedError("This is the same as the mean, deprecate")

    return poisson_estimate


def get_chunked_hist_eigenvector(chunked_session_histograms):
    """
    TODO: a little messy with the 2D stuff. Will probably deprecate anyway.
    """
    if chunked_session_histograms.shape[0] == 1:
        return chunked_session_histograms.squeeze(), None

    is_2d = chunked_session_histograms.ndim == 3

    if is_2d:
        num_hist, num_spat_bin, num_amp_bin = chunked_session_histograms.shape
        chunked_session_histograms = np.reshape(chunked_session_histograms, (num_hist, num_spat_bin * num_amp_bin))

    A = chunked_session_histograms
    S = (1 / A.shape[0]) * A.T @ A

    L, U = np.linalg.eigh(S)

    first_eigenvector = U[:, -1] * np.sqrt(L[-1])
    first_eigenvector = np.abs(first_eigenvector)  # sometimes the eigenvector is negative

    # Project all vectors (histograms) onto the principal component,
    # then take the standard deviation in each dimension (over bins)
    v1 = first_eigenvector[:, np.newaxis]
    projection_onto_v1 = (A @ v1 @ v1.T) / (v1.T @ v1)

    v1_std = np.std(projection_onto_v1, axis=0)

    if is_2d:  # TODO: double check this
        first_eigenvector = np.reshape(first_eigenvector, (num_spat_bin, num_amp_bin))
        v1_std = np.reshape(v1_std, (num_spat_bin, num_amp_bin))

    return first_eigenvector, v1_std


def get_chunked_gaussian_process_regression(chunked_session_histogram):
    """ """
    # TODO: this is currently a placeholder implementation where the
    # mean and variance over repeated samples is taken to run quickly.
    # It would be better to use sparse version with repeated measures
    # as done in pymc.
    # TODO: try https://github.com/cornellius-gp/gpytorch
    # even better : https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Heteroskedastic.html
    #

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    import GPy

    chunked_session_histogram = chunked_session_histogram.copy()
    chunked_session_histogram = chunked_session_histogram

    num_hist = chunked_session_histogram.shape[0]
    num_bins = chunked_session_histogram.shape[1]

    X = np.arange(num_bins)
    X_scaled = X

    Y = chunked_session_histogram

    bias_mean = False
    if bias_mean:
        # this is cool, bias the estimation towards the peak
        Y = Y + np.mean(Y, axis=0) - np.percentile(Y, 5, axis=0)  # TODO: avoid copy, also fix dims in case of square

    # normalise X and set lengthscale to 1 bin
    mu_x = np.mean(X)
    std_x = np.std(X)
    X_scaled = (X - mu_x) / std_x

    lengthscale = 1 / std_x  # 1 spatial bin

    mu_ystar = np.mean(Y)
    std_ystar = np.std(Y)

    # take the mean and variance of Y replicates. Scale to the mean / standard deviation of all y
    y_mean = np.mean(Y, axis=0)
    y_var = np.std(Y, axis=0)

    Y_mean_scaled = (y_mean - mu_ystar) / std_ystar  # standardise the normal way
    Y_var_scaled = (
        1 / std_ystar**2
    ) * y_var  # this is a variance so need to scale to the square (TODO: see overleaf notes)

    kernel = GPy.kern.RBF(input_dim=1, lengthscale=lengthscale, variance=np.mean(Y_var_scaled))  # TODO: check this

    output_index2 = np.arange(num_bins)
    Y_metadata2 = {"output_index": output_index2}

    likelihood = GPy.likelihoods.HeteroscedasticGaussian(
        Y_metadata2, variance=Y_var_scaled
    )  # one variance per y, but should be repeated for the same x

    gp = GPy.models.GPRegression(X_scaled.reshape(-1, 1), Y_mean_scaled.reshape(-1, 1), kernel, Y_metadata2)

    gp.likelihood = likelihood

    gp.optimize(messages=True)

    mean_pred, var_pred = gp.predict(X_scaled.reshape(-1, 1), Y_metadata=Y_metadata2)

    mean_pred = (mean_pred * std_ystar) + mu_ystar
    var_pred = var_pred * std_ystar**2

    std_pred = np.sqrt(var_pred)

    return mean_pred, std_pred, gp


# #############################################################################
# TODO: MOVE creating recordings
# #############################################################################


def compute_histogram_crosscorrelation(
    session_histogram_list: list[np.ndarray],
    non_rigid_windows: np.ndarray,
    num_shifts_block: int,
    interpolate: bool,
    interp_factor: int,
    kriging_sigma: float,
    kriging_p: float,
    kriging_d: float,
    smoothing_sigma_bin: float,
    smoothing_sigma_window: float,
):
    """
    Given a list of session activity histograms, cross-correlate
    all histograms returning the peak correlation shift (in indices)
    in a symmetric (num_session x num_session) matrix.

    Supports non-rigid estimation by windowing the activity histogram
    and performing separate cross-correlations on each window separately.

    Parameters
    ----------

    session_histogram_list : list[np.ndarray]
    non_rigid_windows : np.ndarray
        A (num windows x num_bins) binary of weights by which to window
        the activity histogram for non-rigid-registration. For example, if
        2 rectangular masks were used, there would be a two row binary mask
        the first row with mask of the first half of the probe and the second
        row a mask for the second half of the probe.
    num_shifts_block : int
        Number of indices by which to shift the histogram to find the maximum
        of the cross correlation. If `None`, the entire activity histograms
        are cross-correlated.
    interpolate : bool
        If `True`, the cross-correlation is interpolated before maximum is taken.
    interp_factor:
        Factor by which to interpolate the cross-correlation.
    kriging_sigma : float
        sigma parameter for kriging_kernel function. See `kriging_kernel`.
    kriging_p : float
        p parameter for kriging_kernel function. See `kriging_kernel`.
    kriging_d : float
        d parameter for kriging_kernel function. See `kriging_kernel`.
    smoothing_sigma_bin : float
        sigma parameter for the gaussian smoothing kernel over the
        spatial bins.
    smoothing_sigma_window : float
        sigma parameter for the gaussian smoothing kernel over the
        non-rigid windows.

    Returns
    -------

    shift_matrix : ndarray
        A (num_session x num_session) symmetric matrix of shifts
        (indices) between pairs of session activity histograms.

    Notes
    -----

    - This function is very similar to the IterativeTemplateRegistration
    function used in motion correct, though slightly difference in scope.
    It was not convenient to merge them at this time, but worth looking
    into in future.

    - Some obvious performances boosts, not done so because already fast
      1) the cross correlations for each session comparison are performed
         twice. They are slightly different due to interpolation, but
         still probably better to calculate once and flip.
      2) `num_shifts_block` is implemented by simply making the full
        cross correlation. Would probably be nicer to explicitly calculate
         only where needed. However, in general these cross correlations are
        only a few thousand datapoints and so are already extremely
        fast to cross correlate.

    Notes
    -----

    - The original kilosort method does not work in the inter-session
      context because it averages over time bins to form a template to
      align too. In this case, averaging over a small number of possibly
      quite different session histograms does not work well.

    - In the nonrigid case, this strategy can completely fail when the xcorr
        is very bad for a certain window. The smoothing and interpolation
        make it much worse, because bad xcorr are merged together. The x-corr
        can be bad when the recording is shifted a lot and so there are empty
        regions that are correlated with non-empty regions in the nonrigid
        approach. A different approach will need to be taken in this case.

    Note that kilosort method does not work because creating a
    mean does not make sense over sessions.
    """
    num_sessions = len(session_histogram_list)
    num_bins = session_histogram_list.shape[1]  # all hists are same length
    num_windows = non_rigid_windows.shape[0]

    shift_matrix = np.zeros((num_sessions, num_sessions, num_windows))

    center_bin = np.floor((num_bins * 2 - 1) / 2).astype(int)

    for i in range(num_sessions):
        for j in range(num_sessions):

            # Create the (num windows, num_bins) matrix for this pair of sessions

            import matplotlib.pyplot as plt

            # TODO: plot everything

            num_iter = (
                num_bins * 2 - 1 if not num_shifts_block else num_shifts_block * 2
            )  # TODO: make sure this is clearly defined, it is either side...
            xcorr_matrix = np.zeros((non_rigid_windows.shape[0], num_iter))

            # For each window, window the session histograms (`window` is binary)
            # and perform the cross correlations
            for win_idx, window in enumerate(non_rigid_windows):

                #   breakpoint()
                # TODO: track the 2d histogram through all steps to check everything is working okay

                # TOOD: gaussian window with crosscorr, won't it strongly bias zero shifts by maximising the signal at 0?
                # TODO: add weight option.
                # TODO: damn this is slow for 2D, speed up.
                if session_histogram_list.ndim == 3:
                    windowed_histogram_i = session_histogram_list[i, :] * window[:, np.newaxis]
                    windowed_histogram_j = session_histogram_list[j, :] * window[:, np.newaxis]

                    from scipy.signal import correlate2d

                    # carefully check indices
                    xcorr = correlate2d(
                        windowed_histogram_i - np.mean(windowed_histogram_i, axis=1)[:, np.newaxis],
                        windowed_histogram_j - np.mean(windowed_histogram_j, axis=1)[:, np.newaxis],
                    )  # TOOD: check speed, probs don't remove mean because we want zeros for unmasked version

                    mid_idx = windowed_histogram_j.shape[1] - 1
                    xcorr = xcorr[:, mid_idx]

                else:
                    windowed_histogram_i = session_histogram_list[i, :] * window

                    window_target = True  # this makes less sense now that things could be very far apart
                    if window_target:
                        windowed_histogram_j = session_histogram_list[j, :] * window
                    else:
                        windowed_histogram_j = session_histogram_list[j, :]

                    xcorr = np.correlate(windowed_histogram_i, windowed_histogram_j, mode="full")

                #            plt.plot(windowed_histogram_i)
                #            plt.plot(windowed_histogram_j)
                #            plt.show()

                if num_shifts_block:
                    window_indices = np.arange(center_bin - num_shifts_block, center_bin + num_shifts_block)
                    xcorr = xcorr[window_indices]
                    shift_center_bin = (
                        num_shifts_block  # np.floor(num_shifts_block / 2)  # TODO: CHECK! and move out of loop!
                    )
                else:
                    shift_center_bin = center_bin

                #          plt.plot(xcorr)
                #          plt.show()

                xcorr_matrix[win_idx, :] = xcorr

            # TODO: check absolute value of different bins, they are quite different (log scale, zero mean histograms)
            # TODO: print out a load of quality metrics from this!

            # Smooth the cross-correlations across the bins
            if smoothing_sigma_bin:
                xcorr_matrix = gaussian_filter(xcorr_matrix, smoothing_sigma_bin, axes=1)

            # Smooth the cross-correlations across the windows
            if num_windows > 1 and smoothing_sigma_window:
                xcorr_matrix = gaussian_filter(xcorr_matrix, smoothing_sigma_window, axes=0)

            # Upsample the cross-correlation
            if interpolate:
                shifts = np.arange(xcorr_matrix.shape[1])
                shifts_upsampled = np.linspace(shifts[0], shifts[-1], shifts.size * interp_factor)

                K = kriging_kernel(
                    np.c_[np.ones_like(shifts), shifts],
                    np.c_[np.ones_like(shifts_upsampled), shifts_upsampled],
                    kriging_sigma,
                    kriging_p,
                    kriging_d,
                )
                xcorr_matrix = np.matmul(xcorr_matrix, K, axes=[(-2, -1), (-2, -1), (-2, -1)])

                xcorr_peak = np.argmax(xcorr_matrix, axis=1) / interp_factor
            else:
                xcorr_peak = np.argmax(xcorr_matrix, axis=1)

            #           breakpoint()

            shift = xcorr_peak - shift_center_bin  # center_bin
            shift_matrix[i, j, :] = shift

    return shift_matrix


def shift_array_fill_zeros(array: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift an array by `shift` indices, padding with zero.
    Samples going out of bounds are dropped i,e, the array is not
    extended and samples are not wrapped around to the start of the array.

    Parameters
    ----------

    array : np.ndarray
        The array to pad.
    shift : int
        Number of indices why which to shift the array. If positive, the
       zeros are added from the end of the array. If negative, the zeros
       are added from the start of the array.

    Returns
    -------

    cut_padded_array : np.ndarray
        The `array` padded with zeros and cut down (i.e. out of bounds
        samples dropped).

    """
    abs_shift = np.abs(shift)
    pad_tuple = (0, abs_shift) if shift > 0 else (abs_shift, 0)

    if array.ndim == 2:
        pad_tuple = (pad_tuple, (0, 0))

    padded_hist = np.pad(array, pad_tuple, mode="constant")

    if padded_hist.ndim == 2:
        cut_padded_array = padded_hist[abs_shift:, :] if shift >= 0 else padded_hist[:-abs_shift, :]  # TOOD: tidy up
    else:
        cut_padded_array = padded_hist[abs_shift:] if shift >= 0 else padded_hist[:-abs_shift]

    return cut_padded_array


def akima_interpolate_nonrigid_shifts(
    non_rigid_shifts: np.ndarray,
    non_rigid_window_centers: np.ndarray,
    spatial_bin_centers: np.ndarray,
):
    """
    Perform Akima spline interpolation on a set of non-rigid shifts.
    The non-rigid shifts are per segment of the probe, each segment
    containing a number of channels. Interpolating these non-rigid
    shifts to the spatial bin centers gives a more accurate shift
    per channel.

    Parameters
    ----------
    non_rigid_shifts : np.ndarray
    non_rigid_window_centers : np.ndarray
    spatial_bin_centers : np.ndarray

    Returns
    -------
    interp_nonrigid_shifts : np.ndarray
        An array (length num_spatial_bins) of shifts
        interpolated from the non-rigid shifts.

    TODO
    ----
    requires scipy 14
    """
    from scipy.interpolate import Akima1DInterpolator

    x = non_rigid_window_centers
    xs = spatial_bin_centers

    num_sessions = non_rigid_shifts.shape[0]
    num_bins = spatial_bin_centers.shape[0]

    interp_nonrigid_shifts = np.zeros((num_sessions, num_bins))
    for ses_idx in range(num_sessions):

        y = non_rigid_shifts[ses_idx]
        y_new = Akima1DInterpolator(x, y, method="akima", extrapolate=True)(xs)
        interp_nonrigid_shifts[ses_idx, :] = y_new

    return interp_nonrigid_shifts


def get_shifts_from_session_matrix(alignment_order: str, session_offsets_matrix: np.ndarray):
    """
    Given a matrix of displacements between all sessions, find the
    shifts (one per session) to bring the sessions into alignment.

    Parameters
    ----------
    alignment_order : "to_middle" or "to_session_X" where
        "N" is the number of the session to align to.
    session_offsets_matrix : np.ndarray
        The num_sessions x num_sessions symmetric matrix
        of displacements between all sessions, generated by
        `_compute_session_alignment()`.

    Returns
    -------
    optimal_shift_indices : np.ndarray
        A 1 x num_sessions array of shifts to apply to
        each session in order to bring all sessions into
        alignment.
    """
    if alignment_order == "to_middle":
        optimal_shift_indices = -np.mean(session_offsets_matrix, axis=0)
    else:
        ses_idx = int(alignment_order.split("_")[-1]) - 1
        optimal_shift_indices = -session_offsets_matrix[ses_idx, :, :]

    return optimal_shift_indices
