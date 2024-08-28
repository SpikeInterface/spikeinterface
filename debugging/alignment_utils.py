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


def get_entire_session_hist(recording, peaks, peak_locations, spatial_bin_edges, log_scale):
    """
    TODO: assumes 1-segment recording
    """
    # TODO: this is weird, don't return spatial_bin_edges here... amybe assert..
    entire_session_hist, temporal_bin_edges, spatial_bin_edges = \
        make_2d_motion_histogram(
        recording,
        peaks,
        peak_locations,
        weight_with_amplitude=False,
        direction="y",
        bin_s=recording.get_duration(segment_index=0),
        bin_um=None,
        hist_margin_um=None,
        spatial_bin_edges=spatial_bin_edges,
    )

    entire_session_hist = entire_session_hist[0]

    entire_session_hist /= recording.get_duration(segment_index=0)

    spatial_centers = get_bin_centers(spatial_bin_edges)

    if log_scale:
        entire_session_hist = np.log10(1 + entire_session_hist)

    return entire_session_hist, temporal_bin_edges, spatial_centers


def get_chunked_histogram(
        recording, peaks, peak_locations, bin_s, spatial_bin_edges, log_scale, weight_with_amplitude=False
):
    chunked_session_hist, temporal_bin_edges, _ = \
        make_2d_motion_histogram(
        recording,
        peaks,
        peak_locations,
        weight_with_amplitude=weight_with_amplitude,
        direction="y",
        bin_s=bin_s,
        bin_um=None,
        hist_margin_um=None,  # TODO: ?
        spatial_bin_edges=spatial_bin_edges,
    )

    temporal_centers = get_bin_centers(temporal_bin_edges)
    spatial_centers = get_bin_centers(spatial_bin_edges)

    bin_times = np.diff(temporal_bin_edges)[:, np.newaxis]
    chunked_session_hist /= bin_times

    if log_scale:
        chunked_session_hist = np.log10(1 + chunked_session_hist)

    return chunked_session_hist, temporal_centers, spatial_centers

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def get_bin_centers(bin_edges):
    return (bin_edges[1:] + bin_edges[:-1]) / 2


def estimate_chunk_size(activity_histogram, recording):
    """
    Get an estimate of chunk size such that
    the 85% percentile of the firing rate will be
    estimated within 10% 99% of the time,
    corrected based on assumption
    of Poisson firing (based on CLT).
    """
    firing_rate = np.percentile(activity_histogram, 98)

    lambda_ = firing_rate
    c = 0.5
    n_sd = 2
    n_draws = (n_sd ** 2 * lambda_ / c ** 2)

    t = n_draws

    return t, lambda_


# -----------------------------------------------------------------------------
# Chunked Histogram estimation methods
# -----------------------------------------------------------------------------


def get_chunked_hist_mean(chunked_session_hist):
    """
    """
    mean_hist = np.mean(chunked_session_hist, axis=0)
    return mean_hist


def get_chunked_hist_median(chunked_session_hist):
    """
    """
    median_hist = np.median(chunked_session_hist, axis=0)
    return median_hist


def get_chunked_hist_supremum(chunked_session_hist):
    """
    """
    max_hist = np.max(chunked_session_hist, axis=0)
    return max_hist


def get_chunked_hist_poisson_estimate(chunked_session_hist, trimmed_percentiles=False, weight_on_confidence=False):
    """
    Basically the mean, I guess with robust it becomes trimmed mean
    """
    def obj_fun(lambda_, m, sum_k):
        return -(sum_k * np.log(lambda_) - m * lambda_)

    poisson_estimate = np.zeros(chunked_session_hist.shape[1])  # TODO: var names
    std_devs = []
    for i in range(chunked_session_hist.shape[1]):

        ks = chunked_session_hist[:, i]

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
def get_chunked_hist_eigenvector(chunked_session_hist):
    """
    """
    if chunked_session_hist.shape[0] == 1:  # TODO: handle elsewhere
        return chunked_session_hist.squeeze()

    A = chunked_session_hist - np.mean(chunked_session_hist, axis=0)[np.newaxis, :]
    S = (1/A.shape[0]) * A.T @ A  # (num hist, num_bins)

    U, S, Vh = np.linalg.svd(S)  # TODO: this is already symmetric PSD so use eig

    # TODO: check why this is flipped
    first_eigenvector = U[:, 0] * -1 * S[0]  # * np.sqrt(S[0]) # TODO: revise a little + consider another distance metric

    return first_eigenvector


def plot_chunked_session_hist(est_dict, scale_to_max=False):

    legend = []
    for plot_name in [
        "entire_session_hist", "mean_hist", "median_hist", "max_hist", "poisson_hist"
    ]:
        histogram = est_dict[plot_name]
        if scale_to_max:
            histogram /= np.max(histogram)
        plt.plot(est_dict["chunked_spatial_bins"], histogram)
        legend.append(plot_name)

    plt.legend(legend)
    plt.show()


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


# -----------------------------------------------------------------------------
# TODO: MOVE creating recordings
# -----------------------------------------------------------------------------

def create_motion_recordings(all_recordings, motion_array, all_temporal_bins, non_rigid_window_centers):
    """
    """
    interpolate_motion_kwargs = dict(
        border_mode="remove_channels", spatial_interpolation_method="kriging", sigma_um=20.0, p=2
    )

    corrected_recordings = []
    all_motions = []
    for i in range(len(all_recordings)):

        ses_motion = motion_array[i][np.newaxis, :]

        # TODO: such a hack on the temporal bins
        temporal_bin = np.array([np.mean(all_temporal_bins[i])])
        motion = Motion([ses_motion], [temporal_bin], non_rigid_window_centers, direction="y")  # TODO: must have same probe

        all_motions.append(motion)

        corrected_recordings.append(InterpolateMotionRecording(
            all_recordings[i], motion, **interpolate_motion_kwargs
            )
        )


    return corrected_recordings, all_motions

# Kilosort-like registration
def run_kilosort_like_rigid_registration(all_hists, non_rigid_windows):
    histograms = np.array(all_hists)[:, :, np.newaxis]

    optimal_shift_indices, _, _ = iterative_template_registration(
        histograms, non_rigid_windows=non_rigid_windows
    )

    return -optimal_shift_indices  # TODO: these are reversed at this stage


# TODO: I wonder if it is better to estimate the hitsogram with finer bin size
# than try and interpolate the xcorr. What about smoothing the activity histograms directly?

# TOOD: the iterative_template seems a little different to the interpolation
# of nonrigid segments that is described in the NP2.0 paper. Oh, the KS
# implementation is different to that described in the paper/ where is the
# Akima spline interpolation?

# TODO: make sure that the num bins will always align.
# Apply the linear shifts, don't roll, as we don't want circular (why would the top of the probe appear at the bottom?)
# They roll the windowed version that is zeros, but here we want all done up front to simplify later code

# TODO: this is basically a re-implimentation of the nonrigid part
# of iterative template. Want to leave separate for now for prototyping
# but should combine the shared parts later.

# TOOD: important differenence, this does not roll, will need to test when new spikes are added...

# TODO: try out logarithmic scaling as some neurons fire too much...



def run_alignment_estimation(
    all_session_hists, spatial_bin_centers, rigid, num_nonrigid_bins, robust=False
):
    """
    """
    # TODO: figure out best way to represent this, should probably be
    # suffix _list instead of prefixed all_ for consistency
    if isinstance(all_session_hists, list):
        all_session_hists = np.array(all_session_hists)

    num_bins = spatial_bin_centers.size
    num_sessions = all_session_hists.shape[0]

    # TODO: rename
    hist_array = _compute_rigid_hist_crosscorr(
        num_sessions, num_bins, all_session_hists, robust
    )

    optimal_shift_indices = -np.mean(hist_array, axis=0)[:, np.newaxis]

    # First, perform the rigid alignment.

    if rigid:
        # TODO: used to get window center, for now just get them from the spatial bin
        #  centers and use no margin, which was applied earlier. Same below.
        non_rigid_windows, non_rigid_window_centers = get_spatial_windows(
            spatial_bin_centers,
            spatial_bin_centers,
            rigid=True,
            win_shape="gaussian",
            win_step_um=None,
            win_margin_um=0,
            # win_scale_um=win_scale_um,
            # zero_threshold=None,
        )

        # TODO: rename rigid, also this is weird to pass back bins in the rigid case
        return optimal_shift_indices, non_rigid_window_centers

    win_step_um = (np.max(spatial_bin_centers) - np.min(spatial_bin_centers)) / num_nonrigid_bins

    non_rigid_windows, non_rigid_window_centers = get_spatial_windows(
        spatial_bin_centers,
        spatial_bin_centers,
        rigid=False,
        win_shape="gaussian",
        win_step_um=win_step_um,  # TODO: expose!
        win_margin_um=0,
        # win_scale_um=win_scale_um,
        # zero_threshold=None,
    )

    # Shift the histograms according to the rigid shift
    shifted_histograms = np.zeros_like(all_session_hists)
    for i in range(all_session_hists.shape[0]):

        shift = int(optimal_shift_indices[i, 0])
        abs_shift = np.abs(shift)
        pad_tuple = (0, abs_shift) if shift > 0 else (abs_shift, 0)  # TODO: check direction!

        padded_hist = np.pad(all_session_hists[i, :], pad_tuple, mode="constant")
        cut_padded_hist = padded_hist[abs_shift:] if shift > 0 else padded_hist[:-abs_shift]
        shifted_histograms[i, :] = cut_padded_hist

    # For each nonrigid window, compute the shift
    non_rigid_shifts = np.zeros((num_sessions, non_rigid_windows.shape[0]))
    for i, window in enumerate(non_rigid_windows):

        windowed_histogram = shifted_histograms * window

        # NOTE: this method just xcorr the entire window,
        # does not provide subset of samples like kilosort_like
        window_hist_array = _compute_rigid_hist_crosscorr(
            num_sessions, num_bins, windowed_histogram, robust=False
        )
        non_rigid_shifts[:, i] = -np.mean(window_hist_array, axis=0)

    akima = False  # TODO: decide whether to keep, factor to own function
    if akima:
        from scipy.interpolate import Akima1DInterpolator
        x = win_step_um * np.arange(non_rigid_windows.shape[0])
        xs = spatial_bin_centers

        new_nonrigid_shifts = np.zeros((non_rigid_shifts.shape[0], num_bins))
        for ses_idx in range(non_rigid_shifts.shape[0]):

            y = non_rigid_shifts[ses_idx]
            y_new = Akima1DInterpolator(x, y, method="akima", extrapolate=True)(xs)  # requires scipy 14
            new_nonrigid_shifts[ses_idx, :] = y_new

        shifts = optimal_shift_indices + new_nonrigid_shifts
        non_rigid_window_centers = spatial_bin_centers
    else:
        shifts = optimal_shift_indices + non_rigid_shifts

    return shifts, non_rigid_window_centers


def _compute_rigid_hist_crosscorr(num_sessions, num_bins, all_session_hists, robust=False):
    """"""
    hist_array = np.zeros((num_sessions, num_sessions))
    for i in range(num_sessions):
        for j in range(num_sessions):  # TODO: can make this much faster

            if robust:
                iterations = np.arange(-num_bins, num_bins)
                # TODO: xcorr with weighted least squares
            else:
                argmax = np.argmax(np.correlate(all_session_hists[i, :], all_session_hists[j, :], mode="full"))

            center_bin = np.floor((num_bins * 2 - 1)/2)
            shift = (argmax - center_bin)
            hist_array[i, j] = shift

    return hist_array
