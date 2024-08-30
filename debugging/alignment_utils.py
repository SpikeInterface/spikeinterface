import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.full as si
from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion.motion_utils import \
    make_2d_motion_histogram, make_3d_motion_histograms
from scipy.optimize import minimize
from pathlib import Path
from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
from spikeinterface.sortingcomponents.motion.motion_utils import get_spatial_windows, Motion
from spikeinterface.sortingcomponents.motion.iterative_template import iterative_template_registration
from spikeinterface.sortingcomponents.motion.motion_interpolation import \
    correct_motion_on_peaks

# -----------------------------------------------------------------------------
# Get Histograms
# -----------------------------------------------------------------------------


def get_entire_session_hist(recording, peaks, peak_locations, spatial_bin_edges, log_scale, smooth_um=None):  # TODO: expose smooth_um
    """
    TODO: assumes 1-segment recording
    """
    entire_session_hist, temporal_bin_edges, generated_spatial_bin_edges = \
        make_2d_motion_histogram(
        recording,
        peaks,
        peak_locations,
        weight_with_amplitude=False,
        direction="y",
        bin_s=recording.get_duration(segment_index=0),
        bin_um=None,
        hist_margin_um=None,  # TODO: check all margins etc. are excluded and already passed in the spatial bins
        spatial_bin_edges=spatial_bin_edges,
        depth_smooth_um=smooth_um
    )
    assert np.array_equal(
        generated_spatial_bin_edges, spatial_bin_edges
    ), "TODO: remove soon after testing"

    entire_session_hist = entire_session_hist[0]

    entire_session_hist /= recording.get_duration(segment_index=0)

    spatial_centers = get_bin_centers(spatial_bin_edges)

    if log_scale:
        entire_session_hist = np.log10(1 + entire_session_hist)

    return entire_session_hist, temporal_bin_edges, spatial_centers


def get_chunked_histogram(
        recording, peaks, peak_locations, bin_s, spatial_bin_edges, log_scale, weight_with_amplitude=False, smooth_um=None,  # TODO: expose_um
):
    chunked_session_histograms, temporal_bin_edges, _ = \
        make_2d_motion_histogram(
        recording,
        peaks,
        peak_locations,
        weight_with_amplitude=weight_with_amplitude,
        direction="y",
        bin_s=bin_s,
        bin_um=None,
        hist_margin_um=None,  # TODO: check all margins etc. are excluded and already passed in the spatial bins
        spatial_bin_edges=spatial_bin_edges,
        depth_smooth_um=smooth_um
    )

    temporal_centers = get_bin_centers(temporal_bin_edges)
    spatial_centers = get_bin_centers(spatial_bin_edges)

    bin_times = np.diff(temporal_bin_edges)[:, np.newaxis]
    chunked_session_histograms /= bin_times

    if log_scale:
        chunked_session_histograms = np.log10(1 + chunked_session_histograms)

    return chunked_session_histograms, temporal_centers, spatial_centers

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
    n_draws = (n_sd ** 2 * lambda_ / c ** 2)

    t = n_draws

    return t, lambda_


# -----------------------------------------------------------------------------
# Chunked Histogram estimation methods
# -----------------------------------------------------------------------------


def get_chunked_hist_mean(chunked_session_histograms):
    """
    """
    mean_hist = np.mean(chunked_session_histograms, axis=0)
    return mean_hist


def get_chunked_hist_median(chunked_session_histograms):
    """
    """
    median_hist = np.median(chunked_session_histograms, axis=0)
    return median_hist


def get_chunked_hist_supremum(chunked_session_histograms):
    """
    """
    max_hist = np.max(chunked_session_histograms, axis=0)
    return max_hist


def get_chunked_hist_poisson_estimate(chunked_session_histograms):
    """
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

        # lol, this is painfully close to the mean, no meaningful
        # prior comes to mind to extend the method with.
        poisson_estimate[i] = minimize(obj_fun, 0.5, (m, sum_k),
                                       bounds=((1e-10, np.inf),)).x
    return poisson_estimate


# TODO: currently deprecated due to scaling issues between
# sessions. A much better way will to make PCA from all
# sessions, then align based on projection
def get_chunked_hist_eigenvector(chunked_session_histograms):
    """
    """
    if chunked_session_histograms.shape[0] == 1:  # TODO: handle elsewhere
        return chunked_session_histograms.squeeze()

    A = chunked_session_histograms - np.mean(chunked_session_histograms, axis=0)[np.newaxis, :]
    S = (1/A.shape[0]) * A.T @ A  # (num hist, num_bins)

    U, S, Vh = np.linalg.svd(S)  # TODO: this is already symmetric PSD so use eig

    # TODO: check why this is flipped
    first_eigenvector = U[:, 0] * -1 * S[0]  # * np.sqrt(S[0]) # TODO: revise a little + consider another distance metric

    return first_eigenvector


# -----------------------------------------------------------------------------
# TODO: MOVE creating recordings
# -----------------------------------------------------------------------------


def run_alignment_estimation(
    session_histogram_list, spatial_bin_centers, non_rigid_windows, non_rigid_window_centers, robust=False
):
    """
    """
    if isinstance(session_histogram_list, list):
        session_histogram_list = np.array(session_histogram_list)

    num_bins = spatial_bin_centers.size
    num_sessions = session_histogram_list.shape[0]

    # First, perform the rigid alignment.
    rigid_session_offsets_matrix = _compute_histogram_crosscorrelation(
        num_sessions, num_bins, session_histogram_list, np.ones(num_bins)[np.newaxis, :], robust
    )
    optimal_shift_indices = -np.mean(rigid_session_offsets_matrix, axis=1).T

    if non_rigid_window_centers.shape[0] == 1:  # TODO: Check - rigid case
        return optimal_shift_indices

    # Shift the histograms according to the rigid shift
    # TODO: could do this with a kriging interpolation.
    shifted_histograms = np.zeros_like(session_histogram_list)
    for i in range(session_histogram_list.shape[0]):

        shift = int(optimal_shift_indices[i, 0])
        abs_shift = np.abs(shift)
        pad_tuple = (0, abs_shift) if shift > 0 else (abs_shift, 0)  # TODO: check direction!

        padded_hist = np.pad(session_histogram_list[i, :], pad_tuple, mode="constant")
        cut_padded_hist = padded_hist[abs_shift:] if shift > 0 else padded_hist[:-abs_shift]
        shifted_histograms[i, :] = cut_padded_hist


    rigid_session_offsets_matrix = _compute_histogram_crosscorrelation(
        num_sessions, num_bins, shifted_histograms, non_rigid_windows, robust
    )

    non_rigid_shifts = -np.mean(rigid_session_offsets_matrix, axis=1).T  # fix the order in _compute_histogram_crosscorrelation to avoid this transpose

    # TODO: fix this
    akima = False  # TODO: decide whether to keep, factor to own function
    if akima:
        from scipy.interpolate import Akima1DInterpolator

        x = non_rigid_window_centers
        xs = spatial_bin_centers

        new_nonrigid_shifts = np.zeros((non_rigid_shifts.shape[0], num_bins))
        for ses_idx in range(non_rigid_shifts.shape[0]):

            y = non_rigid_shifts[ses_idx]
            y_new = Akima1DInterpolator(x, y, method="akima", extrapolate=True)(xs)  # requires scipy 14
            new_nonrigid_shifts[ses_idx, :] = y_new

        shifts = optimal_shift_indices + new_nonrigid_shifts
    else:
        shifts = optimal_shift_indices + non_rigid_shifts

    return shifts


def _compute_histogram_crosscorrelation(num_sessions, num_bins, session_histogram_list, non_rigid_windows, robust=False):
    """"""
    import time
    t = time.perf_counter()
    num_windows = non_rigid_windows.shape[0]

    shift_matrix = np.zeros((num_windows, num_sessions, num_sessions))

    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d, gaussian_filter

    # TODO: this is kind of wasteful, no optimisations made against redundant
    # session computation, but these in generate very fast.

    for i in range(num_sessions):
        for j in range(num_sessions):  # TODO: can make this much faster

            xcorr_matrix = np.zeros((non_rigid_windows.shape[0], num_bins * 2 - 1))

            for win_idx, window in enumerate(non_rigid_windows):
                windowed_histogram_i = session_histogram_list[i, :] * window
                windowed_histogram_j = session_histogram_list[j, :] * window

                if robust:
                    iterations = np.arange(-num_bins, num_bins)
                    # TODO: xcorr with weighted least squares
                else:
                    xcorr_matrix[win_idx, :] = np.correlate(windowed_histogram_i, windowed_histogram_j, mode="full")

            smooth_um = 0.5  # TODO: what are the physical interperation of this...
            if smooth_um is not None:  # RENAME
                xcorr_matrix = gaussian_filter(xcorr_matrix, smooth_um, axes=1)

            smooth_window = 1
            if num_windows > 1 and smooth_window:
                xcorr_matrix_ = gaussian_filter(xcorr_matrix, smooth_window, axes=0)

            interpolate = True
            upsample_n = 10  # TODO: fix this.

            if interpolate:

                shifts = np.arange(xcorr_matrix.shape[1])
                shifts_upsampled = np.linspace(shifts[0], shifts[-1], shifts.size * upsample_n)  # TODO: why arbitarily 10, its not stated in NP2 paper actually, ask Sam if he knows. The KS2.5 implementation (MATLAB) seems to differ from the paper?

                sigma = 1  # TODO: expose
                p = 2
                d = 2
                dists = np.repeat(shifts[:, np.newaxis], shifts_upsampled.size, axis=1) - shifts_upsampled[np.newaxis, :]
                K = np.exp(-((dists / sigma) ** p) / d)

                xcorr_matrix = np.matmul(xcorr_matrix, K, axes=[(-2, -1), (-2, -1), (-2, -1)])

                argmax = np.argmax(xcorr_matrix, axis=1) / upsample_n
            else:
                argmax = np.argmax(xcorr_matrix, axis=1)

            argmax = argmax.squeeze()  # TODO: fix this

            center_bin = np.floor((num_bins * 2 - 1)/2)
            shift = (argmax - center_bin)
            shift_matrix[:, i, j] = shift

    print("DONE", time.perf_counter() - t)
    return shift_matrix


# Kilosort-like registration
def run_kilosort_like_rigid_registration(all_hists, non_rigid_windows):
    histograms = np.array(all_hists)[:, :, np.newaxis]

    optimal_shift_indices, _, _ = iterative_template_registration(
        histograms, non_rigid_windows=non_rigid_windows
    )

    return -optimal_shift_indices  # TODO: these are reversed at this stage


# TODO: deprecate
def prep_recording(recording, plot=False):
    """
    :param recording:
    :return:
    """
    peaks = detect_peaks(recording, method="locally_exclusive")

    peak_locations = localize_peaks(recording, peaks,
                                    method="grid_convolution")

    if plot:
        si.plot_drift_raster_map(
            peaks=peaks,
            peak_locations=peak_locations,
            recording=recording,
            clim=(-300, 0)  # fix clim for comparability across plots
        )
        plt.show()

    return peaks, peak_locations
