import time

from spikeinterface import BaseRecording
import numpy as np

from spikeinterface.preprocessing import center
from spikeinterface.sortingcomponents.motion.motion_utils import make_2d_motion_histogram
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from spikeinterface.sortingcomponents.motion.iterative_template import kriging_kernel
from packaging.version import Version


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
    weight_with_amplitude: bool = False,
    avg_in_bin: bool = True,
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
        weight_with_amplitude=weight_with_amplitude,
        direction="y",
        bin_s=(bin_s if bin_s is not None else recording.get_duration(segment_index=0)),
        bin_um=None,
        hist_margin_um=None,
        spatial_bin_edges=spatial_bin_edges,
        depth_smooth_um=depth_smooth_um,
        avg_in_bin=avg_in_bin,
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
    """ """
    mean_hist = np.mean(chunked_session_histograms, axis=0)

    return mean_hist


def get_chunked_hist_median(chunked_session_histograms):
    """ """
    median_hist = np.median(chunked_session_histograms, axis=0)

    quartile_1 = np.percentile(chunked_session_histograms, 25, axis=0)
    quartile_3 = np.percentile(chunked_session_histograms, 75, axis=0)

    return median_hist


# #############################################################################
# TODO: MOVE creating recordings
# #############################################################################


# TODO: a good test here is to give zero shift for even and off numbered hist and check the output is zero!
def compute_histogram_crosscorrelation(
    session_histogram_list: list[np.ndarray],
    non_rigid_windows: np.ndarray,
    num_shifts: int,
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
    num_shifts : int
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
      2) `num_shifts` is implemented by simply making the full
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
    import matplotlib.pyplot as plt

    num_sessions = len(session_histogram_list)
    num_bins = session_histogram_list.shape[1]  # all hists are same length
    num_windows = non_rigid_windows.shape[0]

    shift_matrix = np.zeros((num_sessions, num_sessions, num_windows))

    center_bin = np.floor((num_bins * 2 - 1) / 2).astype(int)

    for i in range(num_sessions):
        for j in range(i, num_sessions):

            # Create the (num windows, num_bins) matrix for this pair of sessions
            num_iter = (
                num_bins * 2 - 1
                if not num_shifts
                else num_shifts * 2  # num_shift_block with iterative alignment is 2x, the same, make note!
            )
            xcorr_matrix = np.zeros((non_rigid_windows.shape[0], num_iter))

            # For each window, window the session histograms (`window` is binary)
            # and perform the cross correlations
            for win_idx, window in enumerate(non_rigid_windows):

                if session_histogram_list.ndim == 3:
                    # For 2D histogram (spatial, amplitude), manually loop through shifts along
                    # the spatial axis of the histogram. This is faster than using correlate2d
                    # because we are not shifting along the amplitude axis.

                    windowed_histogram_i = session_histogram_list[i, :] * window[:, np.newaxis]
                    windowed_histogram_j = session_histogram_list[j, :] * window[:, np.newaxis]

                    window_i = windowed_histogram_i - np.mean(windowed_histogram_i, axis=1)[:, np.newaxis]
                    window_j = windowed_histogram_j - np.mean(windowed_histogram_j, axis=1)[:, np.newaxis]

                    xcorr = np.zeros(num_iter)
                    for idx, shift in enumerate(range(-num_iter // 2, num_iter // 2)):
                        shifted_i = shift_array_fill_zeros(window_i, shift)

                        xcorr[idx] = np.correlate(shifted_i.flatten(), window_j.flatten())

                else:
                    # For a 1D histogram, compute the full cross-correlation and
                    # window the desired shifts ( this is faster than manual looping).
                    windowed_histogram_i = session_histogram_list[i, :] * window
                    windowed_histogram_j = session_histogram_list[j, :] * window

                    xcorr = np.correlate(
                        windowed_histogram_i - np.mean(windowed_histogram_i),
                        windowed_histogram_j - np.mean(windowed_histogram_i),
                        mode="full",
                    )
                    import os
                    if "hello_world" in os.environ:
                        plt.plot(windowed_histogram_i)
                        plt.plot(windowed_histogram_j)
                        plt.show()

                        plt.plot(xcorr)
                        plt.show()

                    if num_shifts:
                        window_indices = np.arange(center_bin - num_shifts, center_bin + num_shifts)
                        xcorr = xcorr[window_indices]

                xcorr_matrix[win_idx, :] = xcorr

            if num_shifts:
                shift_center_bin = num_shifts
            else:
                shift_center_bin = center_bin

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

            # Caclulate and save the shift for session i to j
            shift = xcorr_peak - shift_center_bin
            shift_matrix[i, j, :] = shift

    # As xcorr shifts are symmetric, the shift matrix is skew symmetric, so fill
    # the (empty) lower triangular with the negative (already computed) upper triangular to save computation
    for k in range(shift_matrix.shape[2]):
        lower_i, lower_j = np.tril_indices_from(shift_matrix[:, :, k], k=-1)
        upper_i, upper_j = np.triu_indices_from(shift_matrix[:, :, k], k=1)
        shift_matrix[lower_i, lower_j, k] = shift_matrix[upper_i, upper_j, k] * -1

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
        cut_padded_array = padded_hist[abs_shift:, :] if shift >= 0 else padded_hist[:-abs_shift, :]
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

    """
    if Version(scipy.__version__) >= Version("1.4.0"):
        raise ImportError("Scipy version 14 or higher is required fro Akima interpolation.")

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
    Assumes `session_offsets_matrix` is skew symmetric.

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
