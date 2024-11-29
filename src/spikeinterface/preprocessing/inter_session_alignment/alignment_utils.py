from signal import signal

from toolz import first
from torch.onnx.symbolic_opset11 import chunk

from spikeinterface import BaseRecording
import numpy as np
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
        bin_s=(
            bin_s if bin_s is not None else recording.get_duration(segment_index=0)
        ),  # TODO: doube cehck is this already scaling?
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
        activity_histogram = np.log10(1 + activity_histogram)

    return activity_histogram, temporal_bin_centers, spatial_bin_centers


def get_bin_centers(bin_edges):
    return (bin_edges[1:] + bin_edges[:-1]) / 2


def estimate_chunk_size(scaled_activity_histogram):
    """
    Get an estimate of chunk size such that
    the 80th percentile of the firing rate will be
    estimated within 10% 90% of the time,

    I think a better way is to take the peaks above half width and find the min.
    Or just to take the 50th percentile...? NO. Because all peaks might be similar heights

    corrected based on assumption
    of Poisson firing (based on CLT).

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

    print(f"estimated t: {t} for lambda {lambda_hat_s}")

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

    scaled_range = (max_hist - min_hist) / max_hist  # TODO: no idea if this is a good idea or not

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
    """ """
    if chunked_session_histograms.shape[0] == 1:  # TODO: handle elsewhere
        return chunked_session_histograms.squeeze(), None

    A = chunked_session_histograms
    S = (1 / A.shape[0]) * A.T @ A

    U, S, Vh = np.linalg.svd(S)  # TODO: this is already symmetric PSD so use eig

    first_eigenvector = U[:, 0] * np.sqrt(S[0])
    first_eigenvector = np.abs(first_eigenvector)  # sometimes the eigenvector can be negative

    v1 = first_eigenvector[:, np.newaxis]
    reconstruct = (A @ v1) @ v1.T
    v1_std = np.std(np.sqrt(reconstruct), axis=0, ddof=0)  # TODO: check sqrt, completel guess

    return first_eigenvector, v1_std


def get_chunked_gaussian_process_regression(chunked_session_histogram):
    """ """
    # TODO: try https://github.com/cornellius-gp/gpytorch

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    from sklearn.preprocessing import StandardScaler

    import GPy

    chunked_session_histogram = chunked_session_histogram.copy()
    chunked_session_histogram = chunked_session_histogram

    num_hist = chunked_session_histogram.shape[0]
    num_bins = chunked_session_histogram.shape[1]

    X = np.arange(num_bins)

    Y = chunked_session_histogram

    bias_mean = True
    if bias_mean:
        # this is cool, bias the estimation towards the peak
        Y = Y + np.mean(Y, axis=0) - np.percentile(Y, 5, axis=0)  # TODO: avoid copy, also fix dims in case of square

    # var = np.mean(np.std(Y, axis=0))
    Y = Y.flatten()

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X.reshape(-1, 1)).flatten()
    X_rep = np.tile(X_scaled, num_hist)

    scaler_y = StandardScaler()
    Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1)).flatten()

    ls = 1 / scaler_x.scale_  # 1 spatial bin

    # note variance of the KERNEL is amplitude (sigma^2 out front)
    # noise variance alpha is added to K(X,X) after computation, so is seaprate from kernel

    kernel = GPy.kern.RBF(input_dim=1, lengthscale=ls, variance=1.0)  # + GPy.kern.Constant(input_dim=1, variance=1.0)

    # https://docs.gpytorch.ai/en/v1.9.0/examples/02_Scalable_Exact_GPs/SGPR_Regression_CUDA.html
    sparse = True
    if sparse:
        # num_inducing = (X.shape[0] - 1)  #
        inducing_points = X_scaled  # X[np.random.choice(X.shape[0], num_inducing, replace=False)]
        gp = GPy.models.SparseGPRegression(
            X_rep.reshape(-1, 1), Y_scaled.reshape(-1, 1), kernel, Z=inducing_points.reshape(-1, 1)
        )
    else:
        gp = GPy.models.GPRegression(X_rep.reshape(-1, 1), Y_scaled.reshape(-1, 1), kernel)

    # gp.likelihood = t_distribution  # GPy.likelihoods.StudentT()

    optimise = False
    if optimise:
        kernel.lengthscale.fix()  # try unfixing but TBH looks goood already
        gp.optimize(messages=True)
    else:
        kernel.lengthscale.fix()
        kernel.variance.fix()

    mean_pred, var_pred = gp.predict(X_scaled.reshape(-1, 1))

    mean_pred = scaler_y.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
    std_pred = np.sqrt(var_pred * scaler_y.scale_).flatten()  # TODO: triple check this

    return mean_pred, std_pred, gp


# #############################################################################
# 2D VERSIONS
# #############################################################################

# dims are (time, depth, amplitude


def get_chunked_hist_mean_2d(chunked_histograms):

    mean_hist = np.mean(chunked_histograms, axis=0)

    std = np.std(chunked_histograms, axis=0)

    return mean_hist, std


def get_chunked_hist_median_2d(chunked_histograms):
    breakpoint()
    pass


def get_chunked_hist_supremum_2d(chunked_histograms):
    breakpoint()
    pass


def get_chunked_hist_eigenvector_2d(chunked_histograms):
    breakpoint()
    pass


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
    num_bins = session_histogram_list[0].size  # all hists are same length
    num_windows = non_rigid_windows.shape[0]

    shift_matrix = np.zeros((num_sessions, num_sessions, num_windows))

    center_bin = np.floor((num_bins * 2 - 1) / 2).astype(int)

    for i in range(num_sessions):
        for j in range(num_sessions):

            # Create the (num windows, num_bins) matrix for this pair of sessions
            xcorr_matrix = np.zeros((non_rigid_windows.shape[0], num_bins * 2 - 1))

            # For each window, window the session histograms (`window` is binary)
            # and perform the cross correlations
            for win_idx, window in enumerate(non_rigid_windows):
                windowed_histogram_i = session_histogram_list[i, :] * window
                windowed_histogram_j = session_histogram_list[j, :] * window

                xcorr = np.correlate(windowed_histogram_i, windowed_histogram_j, mode="full")

                if num_shifts_block:
                    window_indices = np.arange(center_bin - num_shifts_block, center_bin + num_shifts_block)
                    mask = np.zeros_like(xcorr)
                    mask[window_indices] = 1
                    xcorr *= mask

                xcorr_matrix[win_idx, :] = xcorr

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

            shift = xcorr_peak - center_bin
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
    padded_hist = np.pad(array, pad_tuple, mode="constant")
    cut_padded_array = padded_hist[abs_shift:] if shift >= 0 else padded_hist[:-abs_shift]

    return cut_padded_array
