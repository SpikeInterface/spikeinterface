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
    entire_session_hist /= np.max(entire_session_hist)
    spatial_centers = get_bin_centers(spatial_bin_edges)

    return entire_session_hist, temporal_centers, spatial_centers


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

    return chunked_session_hist, temporal_centers, spatial_centers

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def get_bin_centers(bin_edges):
    return (spatial_bin_edges[1:] + bin_edges[:-1]) / 2


def estimate_chunk_size(activity_histogram):
    """
    Get an estimate of chunk size such that
    the 85% percentile of the firing rate will be
    estimated within 10% 99% of the time,
    corrected based on assumption
    of Poisson firing (based on CLT).
    """
    firing_rate = np.percentile(activity_histogram, 85)  # est lambda

    l_pois = firing_rate
    l_exp = 1 / l_pois
    perc = 0.1
    s = (l_exp * perc) / 2  # 99% of values (this is very conservative, based on CLT)
    n = 1 / (s ** 2 / l_exp ** 2)
    t = n / l_pois

    return t, n


def normalise_histogram(histogram):
    histogram /= np.max(histogram)
    return hist

# -----------------------------------------------------------------------------
# Chunked Histogram estimation methods
# -----------------------------------------------------------------------------


def get_chunked_hist_mean(chunked_session_hist, normalise=False):
    """
    """
    mean_hist = np.mean(chunked_session_hist, axis=0)
    if normalise:
        mean_hist = normalise_histogram(mean_hist)
    return mean_hist


def get_chunked_hist_median(chunked_session_hist, normalise=False):
    """
    """
    median_hist = np.median(chunked_session_hist, axis=0)
    if normalise:
        median_hist = normalise_histogram(median_hist)
    return median_hist


def get_chunked_hist_supremum(chunked_session_hist, normalise=False):
    """
    """
    max_hist = np.max(chunked_session_hist, axis=0)
    if normalise:
        max_hist = normalise_histogram(max_hist)
    return max_hist


def get_chunked_hist_eigenvector(chunked_session_hist, normalise=False):
    """
    """
    A = chunked_session_hist
    S = A.T @ A  # (num hist, num_bins)

    U,S, Vh = np.linalg.svd(S)

    # TODO: check why this is flipped
    first_eigenvalue = U[:, 0] * -1  # TODO: revise a little + consider another distance metric
    if normalise:
        first_eigenvalue = normalise_histogram(first_eigenvalue)

    return first_eigenvalue


def get_chunked_hist_poisson_estimate(chunked_session_hist, normalise=False):
    """
    """
    def obj_fun(lambda_, m, sum_k):
        return -(sum_k * np.log(lambda_) - m * lambda_)

    poisson_estimate = np.zeros(chunked_session_hist.shape[1])  # TODO: var names
    for i in range(chunked_session_hist.shape[1]):
        ks = chunked_session_hist[:, i]

        m = ks.shape
        sum_k = np.sum(ks)

        # lol, this is painfully close to the mean...
        poisson_estimate[i] = minimize(obj_fun, 0.5, (m, sum_k),
                                       bounds=((1e-10, np.inf),)).x
    if normalise:
        poisson_estimate = normalise_histogram(poisson_estimate)

    return poisson_estimate


def get_all_hist_estimation(recording, peaks, peak_locations):
    """
    """
    bin_um = 60

    entire_session_hist, _, _ = get_entire_session_hist(
        recording, peaks, peak_locations, bin_um
    )

    # need to time this, and estimate from a few chunks if necessary...
    bin_s = estimate_chunk_size(entire_session_hist)

    chunked_session_hist, chunked_temporal_bins, chunked_spatial_bins = get_chunked_histogram(
        recording, peaks, peak_locations, bin_s, bin_um
    )

    # TODO: own function
    n_bin = chunked_session_hist.shape[1]

    session_std = np.sum(np.std(chunked_session_hist, axis=0)) / n_bin
    print("Histogram STD:: ", session_std)

    mean_hist = get_chunked_hist_mean(chunked_session_hist, normalise=True)
    median_hist = get_chunked_hist_median(chunked_session_hist, normalise=True)
    max_hist = get_chunked_hist_supremum(chunked_session_hist, normalise=True)
    eigenvector_hist = get_chunked_hist_eigenvector(chunked_session_hist, normalise=True)
    poisson_hist = get_chunked_hist_poisson_estimate(chunked_session_hist, normalise=True)

    return {
        "entire_session_hist": entire_session_hist,
        "chunked_session_hist": chunked_session_hist,
        "chunked_temporal_bins": chunked_temporal_bins,
        "chunked_spatial_bins": chunked_spatial_bins,
        "mean_hist": mean_hist,
        "median_hist": median_hist,
        "max_hist": max_hist,
        "eigenvector_hist": eigenvector_hist,
        "poisson_hist": poisson_hist,
    }


def plot_chunked_session_hist(est_dict):
    plt.plot(est_dict["entire_session_hist"])  # obs this is equal to mean hist
    plt.plot(est_dict["mean_hist"])
    plt.plot(est_dict["median_hist"])
    plt.plot(est_dict["max_hist"])
    plt.plot(est_dict["first_eigenvalue"])
    plt.plot(est_dict["poisson_estimate"])
    plt.legend(
        ["entire", "chunk mean", "chunk median",
         "chunk_max", "chunk eigenvalue", "Poisson estimate"
         ])
    plt.show()


def plot_all_hist_estimation(chunked_session_hist, chunked_spatial_bins):
    """
    """
    for i in range(chunked_session_hist.shape[0]):
        plt.plot(chunked_spatial_bins, chunked_session_hist[i, :])
    plt.show()
