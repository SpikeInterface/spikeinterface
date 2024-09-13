import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.full as si
from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion.motion_utils import make_2d_motion_histogram, make_3d_motion_histograms
from scipy.optimize import minimize
from pathlib import Path
from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
from spikeinterface.sortingcomponents.motion.motion_utils import get_spatial_windows, Motion
from spikeinterface.sortingcomponents.motion.iterative_template import iterative_template_registration
from spikeinterface.sortingcomponents.motion.motion_interpolation import correct_motion_on_peaks
from scipy.ndimage import gaussian_filter
from spikeinterface.sortingcomponents.motion.iterative_template import kriging_kernel

# -----------------------------------------------------------------------------
# Get Histograms
# -----------------------------------------------------------------------------


def get_activity_histogram(
    recording, peaks, peak_locations, spatial_bin_edges, log_scale, bin_s, depth_smooth_um
):
    """
    TODO: assumes 1-segment recording
    """
    activity_histogram, temporal_bin_edges, generated_spatial_bin_edges = make_2d_motion_histogram(
        recording,
        peaks,
        peak_locations,
        weight_with_amplitude=False,
        direction="y",
        bin_s=bin_s if bin_s is not None else recording.get_duration(segment_index=0),
        bin_um=None,
        hist_margin_um=None,
        spatial_bin_edges=spatial_bin_edges,
        depth_smooth_um=depth_smooth_um,
    )
    assert np.array_equal(generated_spatial_bin_edges, spatial_bin_edges), "TODO: remove soon after testing"

    temporal_bin_centers = get_bin_centers(temporal_bin_edges)
    spatial_bin_centers = get_bin_centers(spatial_bin_edges)

    if bin_s is None:
        scaler = 1 / recording.get_duration(segment_index=0)
    else:
        scaler = 1 / np.diff(temporal_bin_edges)[:, np.newaxis]

    activity_histogram *= scaler

    if log_scale:
        activity_histogram = np.log10(1 + activity_histogram)

    return activity_histogram, temporal_bin_centers, spatial_bin_centers


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def get_bin_centers(bin_edges):
    return (bin_edges[1:] + bin_edges[:-1]) / 2


def estimate_chunk_size(scaled_activity_histogram):
    """
    Get an estimate of chunk size such that
    the 85% percentile of the firing rate will be
    estimated within 10% 99% of the time,
    corrected based on assumption
    of Poisson firing (based on CLT).
    """
    firing_rate = np.percentile(scaled_activity_histogram, 98)

    lambda_ = firing_rate
    c = 0.5
    n_sd = 2
    n_draws = n_sd**2 * lambda_ / c**2

    t = n_draws

    return t, lambda_


# -----------------------------------------------------------------------------
# Chunked Histogram estimation methods
# -----------------------------------------------------------------------------


def get_chunked_hist_mean(chunked_session_histograms):
    """ """
    mean_hist = np.mean(chunked_session_histograms, axis=0)
    return mean_hist


def get_chunked_hist_median(chunked_session_histograms):
    """ """
    median_hist = np.median(chunked_session_histograms, axis=0)
    return median_hist


def get_chunked_hist_supremum(chunked_session_histograms):
    """ """
    max_hist = np.max(chunked_session_histograms, axis=0)
    return max_hist


def get_chunked_hist_poisson_estimate(chunked_session_histograms):
    """ """

    def obj_fun(lambda_, m, sum_k):
        return -(sum_k * np.log(lambda_) - m * lambda_)

    poisson_estimate = np.zeros(chunked_session_histograms.shape[1])
    std_devs = []
    for i in range(chunked_session_histograms.shape[1]):

        ks = chunked_session_histograms[:, i]

        std_devs.append(np.std(ks))
        m = ks.shape
        sum_k = np.sum(ks)

        # lol, this is painfully close to the mean, no meaningful
        # prior comes to mind to extend the method with.
        poisson_estimate[i] = minimize(obj_fun, 0.5, (m, sum_k), bounds=((1e-10, np.inf),)).x
    return poisson_estimate


# TODO: currently deprecated due to scaling issues between
# sessions. A much better (?) way will to make PCA from all
# sessions, then align based on projection
def get_chunked_hist_eigenvector(chunked_session_histograms):
    """ """
    if chunked_session_histograms.shape[0] == 1:  # TODO: handle elsewhere
        return chunked_session_histograms.squeeze()

    A = chunked_session_histograms - np.mean(chunked_session_histograms, axis=0)[np.newaxis, :]
    S = (1 / A.shape[0]) * A.T @ A  # (num hist, num_bins)

    U, S, Vh = np.linalg.svd(S)  # TODO: this is already symmetric PSD so use eig

    # TODO: check why this is flipped
    first_eigenvector = (
        U[:, 0] * -1 * S[0]
    )  # * np.sqrt(S[0]) # TODO: revise a little + consider another distance metric

    return first_eigenvector


# -----------------------------------------------------------------------------
# TODO: MOVE creating recordings
# -----------------------------------------------------------------------------


def compute_histogram_crosscorrelation(
    session_histogram_list,
    non_rigid_windows,
    num_shifts_block,
    interpolate,
    interp_factor,
    kriging_sigma,
    kriging_p,
    kriging_d,
    smoothing_sigma_bin,
    smoothing_sigma_window,
):
    """
    # TODO: what happens when this bigger than thing. Also rename about shifts
    # TODO: this is kind of wasteful, no optimisations made against redundant
    # session computation, but these in generate very fast.

    # The problem is this stratergy completely fails when thexcorr is very bad.
    # The smoothing and interpolation make it much worse, because bad xcorr are
    # merged together. The xcorr can be bad when the recording is shifted a lot
    # and so there are empty regions that are correlated with non-empty regions
    # in the nonrigid approach. A different approach will need to be taken in
    # this case.

    # Note that due to interpolation, ij vs ji are not exact duplicates.
    # dont improve for now.

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


def shift_array_fill_zeros(array, shift):
    abs_shift = np.abs(shift)
    pad_tuple = (0, abs_shift) if shift > 0 else (abs_shift, 0)
    padded_hist = np.pad(array, pad_tuple, mode="constant")
    cut_padded_hist = padded_hist[abs_shift:] if shift >= 0 else padded_hist[:-abs_shift]
    return cut_padded_hist


# TODO: deprecate
def prep_recording(recording, plot=False):
    """
    :param recording:
    :return:
    """
    peaks = detect_peaks(recording, method="locally_exclusive")

    peak_locations = localize_peaks(recording, peaks, method="grid_convolution")

    if plot:
        si.plot_drift_raster_map(
            peaks=peaks,
            peak_locations=peak_locations,
            recording=recording,
            clim=(-300, 0),  # fix clim for comparability across plots
        )
        plt.show()

    return peaks, peak_locations
