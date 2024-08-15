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

# TODO: this function might be pointless
def get_entire_session_hist(recording, peaks, peak_locations, bin_um):
    """
    TODO: assumes 1-segment recording
    """
    entire_session_hist, temporal_bin_edges, spatial_bin_edges = \
        make_2d_motion_histogram(
        recording,
        peaks,
        peak_locations,
        weight_with_amplitude=False,
        direction="y",
        bin_s=recording.get_duration(segment_index=0),
        bin_um=bin_um,
        hist_margin_um=50,  # TODO: ?
        spatial_bin_edges=None,
    )
    entire_session_hist = entire_session_hist[0]

    entire_session_hist /= recording.get_duration(segment_index=0)

    spatial_centers = get_bin_centers(spatial_bin_edges)

    return entire_session_hist, temporal_bin_edges, spatial_centers


def get_chunked_histogram(  # TODO: this function might be pointless
        recording, peaks, peak_locations, bin_s, bin_um, weight_with_amplitude=False
):
    chunked_session_hist, temporal_bin_edges, spatial_bin_edges = \
        make_2d_motion_histogram(
        recording,
        peaks,
        peak_locations,
        weight_with_amplitude=weight_with_amplitude,
        direction="y",
        bin_s=bin_s,
        bin_um=bin_um,
        hist_margin_um=50,  # TODO: ?
        spatial_bin_edges=None,
    )

    temporal_centers = get_bin_centers(temporal_bin_edges)
    spatial_centers = get_bin_centers(spatial_bin_edges)

    bin_times = np.diff(temporal_bin_edges)[:, np.newaxis]
    chunked_session_hist /= bin_times

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

    TODO: activity histogram must be scaled to spikes-per-second
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

        trimmed_percentiles = (20, 80)  # TODO: move this
        if trimmed_percentiles is not False:
            min, max = trimmed_percentiles
            min_percentile = np.percentile(ks, min)
            max_percentile = np.percentile(ks, max)

            ks = ks[
                np.logical_and(ks >= min_percentile, ks <= max_percentile)
            ]

        std_devs.append(np.std(ks))
        m = ks.shape
        sum_k = np.sum(ks)

        # lol, this is painfully close to the mean...
        poisson_estimate[i] = minimize(obj_fun, 0.5, (m, sum_k),
                                       bounds=((1e-10, np.inf),)).x
    weight_on_confidence = True
    # TODO: better handle single-time point estimation.
    if weight_on_confidence and np.any(std_devs):  # TODO: there is no reason this can be done just for poisson, can be done for all... maybe POisson has better variance estimate, do properly!
        # do exponential
        # this is a bad idea, we literally want to weight on height!
        stds = np.array(std_devs)
        stds = stds[~(stds==0)]
        stds = (stds - np.min(stds)) / (np.max(stds) - np.min(stds))

        # TODO: or weight by confidence?  this is basically the same as weighting by signal due to poisson variation
        stds = stds * (2 - np.exp(2 * stds))  # TODO: expose param, does this even make sense? does it scale?
        stds[np.where(stds<0)] = 0

    return poisson_estimate


def estimate_session_displacement_benchmarking(
        recordings_list, peaks_list, peak_locations_list, bin_um
):
    # Make histograms per session
    session_histogram_info = []
    for i in range(len(recordings_list)):

        ses_info = get_all_hist_estimation(recordings_list[i], peaks_list[i], peak_locations_list[i], bin_um)

        print(f"Session {i}\n-------------------")
        print("firing rate", ses_info["percentile_lambda"])
        print("Histogram STD:: ", ses_info["session_std"])
        print("bin_s", ses_info["bin_s"])
        print("recording duration", ses_info["recording_duration"])

        session_histogram_info.append(ses_info)

    # Check all the bins are the same, get the windows
    for i in range(len(session_histogram_info)):
        assert np.array_equal(session_histogram_info[0]["chunked_spatial_bins"],
                              session_histogram_info[i]["chunked_spatial_bins"]
                              )

    spatial_bins = session_histogram_info[0]["chunked_spatial_bins"]

    _, non_rigid_window_centers = get_spatial_windows_2(
        recordings_list[0], spatial_bins
    )

    # Compute the estimated alignment.
    alignment_results = {"histograms": {}, "motion_arrays": {}, "corrected_recordings": {}}

    for hist_name in ["mean_hist", "median_hist", "max_hist", "poisson_hist"]:
        all_ses_histogram = []
        for info in session_histogram_info:
            all_ses_histogram.append(info[hist_name])
        all_ses_histogram = np.array(all_ses_histogram)

        motion_array = run_alignment_estimation_rigid(
            all_ses_histogram, spatial_bins
        ) * bin_um

        alignment_results["histograms"][hist_name] = all_ses_histogram
        alignment_results["motion_arrays"][hist_name] = motion_array

    return alignment_results


def get_all_hist_estimation(recording, peaks, peak_locations, bin_um):
    """
    """
    entire_session_hist, _, _ = get_entire_session_hist(
        recording, peaks, peak_locations, bin_um
    )

    bin_s, percentile_lambda = estimate_chunk_size(entire_session_hist, recording)

    chunked_session_hist, chunked_temporal_bins, chunked_spatial_bins = get_chunked_histogram(
        recording, peaks, peak_locations, bin_s, bin_um
    )
    session_std = np.sum(np.std(chunked_session_hist, axis=0)) / chunked_session_hist.shape[1]

    mean_hist = get_chunked_hist_mean(chunked_session_hist)
    median_hist = get_chunked_hist_median(chunked_session_hist)
    max_hist = get_chunked_hist_supremum(chunked_session_hist)
    poisson_hist = get_chunked_hist_poisson_estimate(chunked_session_hist)

    return {
        "entire_session_hist": entire_session_hist,
        "chunked_session_hist": chunked_session_hist,
        "chunked_temporal_bins": chunked_temporal_bins,
        "chunked_spatial_bins": chunked_spatial_bins,
        "mean_hist": mean_hist,
        "median_hist": median_hist,
        "max_hist": max_hist,
        "poisson_hist": poisson_hist,
        "bin_s": bin_s,
        "recording_duration": recording.get_duration(0),
        "session_std": session_std,
        "percentile_lambda": percentile_lambda,
    }


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


def plot_all_hist_estimation(chunked_session_hist, chunked_spatial_bins):
    """
    """
    for i in range(chunked_session_hist.shape[0]):
        plt.plot(chunked_spatial_bins, chunked_session_hist[i, :])
    plt.show()


def prep_recording(recording, plot=False):
    """

    :param recording:
    :return:
    """
    peaks = detect_peaks(recording, method="by_channel")  # "locally_exclusive")

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

        ses_motion = motion_array[i][:, np.newaxis]

        # TODO: such a hack on the temporal bins
        temporal_bin = np.array([np.mean(all_temporal_bins[i])])
        motion = Motion([ses_motion], [temporal_bin], non_rigid_window_centers, direction="y")  # TODO: must have same probe

        all_motions.append(motion)

        corrected_recordings.append(InterpolateMotionRecording(
            all_recordings[i][0], motion, **interpolate_motion_kwargs
            )
        )

    return corrected_recordings, all_motions


def get_spatial_windows_2(recording, spatial_bin_centers):
    dim = 1  # "["x", "y", "z"].index(direction)
    contact_depths = recording.get_channel_locations()[:, dim]

    non_rigid_windows, non_rigid_window_centers = get_spatial_windows(
        contact_depths, spatial_bin_centers, rigid=True
    )
    return non_rigid_windows, non_rigid_window_centers


# Kilosort-like registration
def run_kilosort_like_rigid_registration(all_hists, non_rigid_windows):
    histograms = np.array(all_hists)[:, :, np.newaxis]

    optimal_shift_indices, _, _ = iterative_template_registration(
        histograms, non_rigid_windows=non_rigid_windows
    )

    return -optimal_shift_indices  # TODO: these are reversed at this stage


def run_alignment_estimation_rigid(
    all_hists, spatial_bin_centers, robust=False
):
    """
    """
    num_bins = spatial_bin_centers.size
    num_sessions = all_hists.shape[0]

    hist_array = np.zeros((num_sessions, num_sessions))
    for i in range(num_sessions):
        for j in range(num_sessions):  # TODO: can make this much faster

            if robust:
                iterations = np.arange(-num_bins, num_bins)
                # TODO: xcorr with weighted least squares
            else:
                argmax = np.argmax(np.correlate(all_hists[i, :], all_hists[j, :], mode="full"))

            center_bin = np.floor((num_bins * 2 - 1)/2)
            shift = (argmax - center_bin)
            hist_array[i, j] = shift

    # TODO: this just shifts everything to the center. It would be (better?)
    # to optmize so all shifts are about the same.
    optimal_shift_indices = -np.mean(hist_array, axis=0)[:, np.newaxis]

    return optimal_shift_indices


def correct_peaks_and_plot_histogram(
    corrected_recordings, all_recordings, all_motions, bin_um
):
    legend = []

    for i, recording in enumerate(corrected_recordings):
        corr_peak_locations = correct_motion_on_peaks(
            all_recordings[i][1],
            all_recordings[i][2],
            all_motions[i],
            recording
        )

        entire_session_hist, _, spatial_bins = get_entire_session_hist(
            recording, all_recordings[i][1], corr_peak_locations, bin_um
        )

        plt.plot(spatial_bins, entire_session_hist)
        legend.append(all_motions[i].displacement[0].squeeze())
    plt.legend(legend)
    plt.show()
