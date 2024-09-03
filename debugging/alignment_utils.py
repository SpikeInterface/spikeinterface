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
from scipy.ndimage import gaussian_filter
from spikeinterface.sortingcomponents.motion.iterative_template import kriging_kernel

# -----------------------------------------------------------------------------
# Get Histograms
# -----------------------------------------------------------------------------


def get_activity_histogram(recording, peaks, peak_locations, spatial_bin_edges, log_scale, bin_s, smooth_um=None):  # TODO: expose smooth_um
    """
    TODO: assumes 1-segment recording
    """
    activity_histogram, temporal_bin_edges, generated_spatial_bin_edges = \
        make_2d_motion_histogram(
        recording,
        peaks,
        peak_locations,
        weight_with_amplitude=False,
        direction="y",
        bin_s=bin_s if bin_s is not None else recording.get_duration(segment_index=0),
        bin_um=None,
        hist_margin_um=None,
        spatial_bin_edges=spatial_bin_edges,
        depth_smooth_um=smooth_um
    )
    assert np.array_equal(
        generated_spatial_bin_edges, spatial_bin_edges
    ), "TODO: remove soon after testing"

    temporal_bin_centers = get_bin_centers(temporal_bin_edges)
    spatial_bin_centers = get_bin_centers(spatial_bin_edges)

    if bin_s is None:
        scaler = 1 / recording.get_duration(segment_index=0)
    else:
        scaler = np.diff(temporal_bin_edges)[:, np.newaxis]

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
# sessions. A much better (?) way will to make PCA from all
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
    session_histogram_list, spatial_bin_centers, non_rigid_windows, non_rigid_window_centers, alignment_order, robust=False
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

    optimal_shift_indices = get_shifts_from_session_matrix(alignment_order, rigid_session_offsets_matrix)

    if non_rigid_window_centers.shape[0] == 1:  # rigid case
        return optimal_shift_indices, non_rigid_window_centers  # TOOD: this is weird

    # For non-rigid, first shift the histograms according to the rigid shift
    shifted_histograms = np.zeros_like(session_histogram_list)
    for i in range(session_histogram_list.shape[0]):

        shift = int(optimal_shift_indices[i, 0])
        abs_shift = np.abs(shift)
        pad_tuple = (0, abs_shift) if shift > 0 else (abs_shift, 0)

        padded_hist = np.pad(session_histogram_list[i, :], pad_tuple, mode="constant")

        cut_padded_hist = padded_hist[abs_shift:] if shift >= 0 else padded_hist[:-abs_shift]
        shifted_histograms[i, :] = cut_padded_hist

    rigid_session_offsets_matrix = _compute_histogram_crosscorrelation(  # TODO: rename variable
        num_sessions, num_bins, shifted_histograms, non_rigid_windows, robust
    )
    non_rigid_shifts = get_shifts_from_session_matrix(alignment_order, rigid_session_offsets_matrix)

    akima = False  # TODO: expose this
    if akima:
        interp_nonrigid_shifts = akima_interpolate_nonrigid_shifts(
            non_rigid_shifts, non_rigid_window_centers, spatial_bin_centers
        )
        shifts = optimal_shift_indices + interp_nonrigid_shifts
        non_rigid_window_centers = spatial_bin_centers
    else:
        shifts = optimal_shift_indices + non_rigid_shifts

    return shifts, non_rigid_window_centers


def get_shifts_from_session_matrix(alignment_order, session_offsets_matrix):  # TODO: rename
    """
    """
    if alignment_order == "to_middle":  # TODO: do a lot of arg checks
        optimal_shift_indices = -np.mean(session_offsets_matrix, axis=0)  # TODO: these are not symmetrical because of interpolation?
    else:
        ses_idx = int(alignment_order.split("_")[-1]) - 1
        optimal_shift_indices = -session_offsets_matrix[ses_idx, :, :]

    return optimal_shift_indices


def akima_interpolate_nonrigid_shifts(non_rigid_shifts, non_rigid_window_centers, spatial_bin_centers):
    """
    """
    from scipy.interpolate import Akima1DInterpolator

    x = non_rigid_window_centers
    xs = spatial_bin_centers

    num_sessions = non_rigid_shifts.shape[0]
    num_bins = spatial_bin_centers.shape[0]

    interp_nonrigid_shifts = np.zeros((num_sessions, num_bins))
    for ses_idx in range(num_sessions):
        y = non_rigid_shifts[ses_idx]
        y_new = Akima1DInterpolator(x, y, method="akima", extrapolate=True)(xs) # TODO: requires scipy 14
        interp_nonrigid_shifts[ses_idx, :] = y_new

    return interp_nonrigid_shifts


def _compute_histogram_crosscorrelation(num_sessions, num_bins, session_histogram_list, non_rigid_windows, robust=False):
    """
    # TODO: this is kind of wasteful, no optimisations made against redundant
    # session computation, but these in generate very fast.
    """
    num_windows = non_rigid_windows.shape[0]

    shift_matrix = np.zeros((num_sessions, num_sessions, num_windows))

    for i in range(num_sessions):
        for j in range(num_sessions):

            # Create the (num windows, num_bins) matrix for this pair of sessions
            xcorr_matrix = np.zeros((non_rigid_windows.shape[0], num_bins * 2 - 1))

            # For each window, window the session histograms (`window` is binary)
            # and perform the cross correlations
            for win_idx, window in enumerate(non_rigid_windows):
                windowed_histogram_i = session_histogram_list[i, :] * window
                windowed_histogram_j = session_histogram_list[j, :] * window

                xcorr_matrix[win_idx, :] = np.correlate(windowed_histogram_i, windowed_histogram_j, mode="full")

            # Smooth the cross-correlations across the bins
            smooth_um = None # 0.5  # TODO: what are the physical interperation of this... also expose ... also rename
            if smooth_um is not None:
                xcorr_matrix = gaussian_filter(xcorr_matrix, smooth_um, axes=1)

            # Smooth the cross-correlations across the windows
            smooth_window = None # 1  # TODO: expose
            if num_windows > 1 and smooth_window:
                xcorr_matrix = gaussian_filter(xcorr_matrix, smooth_window, axes=0)

            # Upsample the cross-correlation by a factor of 10  # TODO: expose
            interpolate = False
            upsample_n = 10  # TODO: fix this. factor of 10 is arbitary (?), can combine args
            if interpolate:
                shifts = np.arange(xcorr_matrix.shape[1])
                shifts_upsampled = np.linspace(shifts[0], shifts[-1], shifts.size * upsample_n)

                sigma = 1; p = 2; d = 2 # TODO: expose
                K = kriging_kernel(np.c_[np.ones_like(shifts), shifts], np.c_[np.ones_like(shifts_upsampled), shifts_upsampled], sigma, p, d)

                xcorr_matrix = np.matmul(xcorr_matrix, K, axes=[(-2, -1), (-2, -1), (-2, -1)])

                argmax = np.argmax(xcorr_matrix, axis=1) / upsample_n
            else:
                argmax = np.argmax(xcorr_matrix, axis=1)


            import matplotlib.pyplot as plt
            plt.plot(xcorr_matrix.squeeze())
            plt.title(f"{i} and {j}")
            plt.show()

            center_bin = np.floor((num_bins * 2 - 1)/2)
            shift = (argmax - center_bin)
            shift_matrix[i, j, :] = shift

    return shift_matrix


# Kilosort-like registration
def run_kilosort_like_rigid_registration(all_hists, non_rigid_windows):
    """
    TODO: this doesn't really work in the inter-session context,
    just for testing. Remove soon. JZ 02/09/2024
    """
    histograms = np.array(all_hists)[:, :, np.newaxis]

    optimal_shift_indices, _, _ = iterative_template_registration(
        histograms, non_rigid_windows=non_rigid_windows
    )

    return -optimal_shift_indices


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
