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

import spikeinterface.full as si
from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion.motion_utils import \
    make_2d_motion_histogram, make_3d_motion_histograms
from scipy.optimize import minimize
from pathlib import Path
import alignment_utils
"""
Weight on confidence
two problems:
- how to measure 'confidence' (peak height? std?) larger peaks may have
  higher std, but we care about them more, so I think this is largely pointless.

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

trimmed_percentiles = (20, 80)  # TODO: this is originally in the context of Poisson estimation
if trimmed_percentiles is not False:
    min, max = trimmed_percentiles
    min_percentile = np.percentile(ks, min)
    max_percentile = np.percentile(ks, max)

    ks = ks[
        np.logical_and(ks >= min_percentile, ks <= max_percentile)
    ]
"""

# -----------------------------------------------------------------------------
# Get Histograms
# -----------------------------------------------------------------------------

# 1) for now, assume it is a multi-session recording. Then,
#    add the peak-detection stuff if not later
# 2) for now, estimate entire session. But, soon make this chunked.
#    If peaks are already detected, might as well estimate from
#    entire session. Otherwise, we will want to add chunking as part of above.

def run_inter_session_displacement_correction(
        recordings_list, peaks_list, peak_locations_list, bin_um, histogram_estimation_method, alignment_method
):  # TOOD: rename
    """
    """
    motion_estimates_list, all_temporal_bin_centers, all_spatial_bin_centers, histogram_info = estimate_inter_session_displacement(
        recordings_list, peaks_list, peak_locations_list, bin_um, histogram_estimation_method, alignment_method
    )

    _, non_ridgid_spatial_windows = alignment_utils.get_spatial_windows_alignment(
        recordings_list[0], all_spatial_bin_centers[0]
    )

    all_corrected_recordings, motion_objects_list = alignment_utils.create_motion_recordings(
        recordings_list, motion_estimates_list, all_temporal_bin_centers, non_ridgid_spatial_windows
    )

    extra_outputs_dict = {
        "motion_estimates_list": motion_estimates_list,
        "all_temporal_bin_centers": all_temporal_bin_centers,
        "all_spatial_bin_centers": all_spatial_bin_centers,
        "histogram_info": histogram_info,
    }
    return all_corrected_recordings, motion_objects_list, extra_outputs_dict

def estimate_inter_session_displacement(
    recordings_list, peaks_list, peak_locations_list, bin_um, histogram_estimation_method, alignment_method
):
    """
    """
    assert alignment_method in ["kilosort_like", "mean_crosscorr"], "TODO"

    # Estimate an activity histogram per-session
    all_session_hists = []  # TODO: probably better as a dict
    all_temporal_bin_centers = []
    all_spatial_bin_centers = []
    all_session_chunked_hists = []
    all_chunked_hist_stdevs = []

    for recording, peaks, peak_locations in zip(recordings_list, peaks_list, peak_locations_list):

        session_hist, temporal_bin_centers, spatial_bin_centers, session_chunked_hists, chunked_hist_stdevs = _get_single_session_activity_histogram(
                recording, peaks, peak_locations, bin_um, histogram_estimation_method
        )

        all_session_hists.append(session_hist)
        all_temporal_bin_centers.append(temporal_bin_centers)
        all_spatial_bin_centers.append(spatial_bin_centers)
        all_session_chunked_hists.append(session_chunked_hists)
        all_chunked_hist_stdevs.append(chunked_hist_stdevs)

        # TODO: decide what to do with session_chunked_hists, chunked_hist_stdevs
        all_motion_arrays = []

    # Check all spatial bins are the same.
    for i in range(len(all_spatial_bin_centers)):
        assert np.array_equal(all_spatial_bin_centers[0],
                              all_spatial_bin_centers[i],
                              )

    # Estimate the alignment based on the activity histograms
    if alignment_method == "kilosort_like":
        non_rigid_windows, non_rigid_window_centers = alignment_utils.get_spatial_windows_alignment(
            recordings_list[0], all_spatial_bin_centers[0]
            # TODO: double check these assumptions (first session) are good
        )
        all_motion_arrays = alignment_utils.run_kilosort_like_rigid_registration(  # TODO: check sign
            all_session_hists, non_rigid_windows
        ) * bin_um
    else:
        all_motion_arrays = alignment_utils.run_alignment_estimation_rigid(
            all_session_hists, all_spatial_bin_centers[0]
        ) * bin_um  # TODO: here the motion arrays are made negative initially. In motion correction they are done later. Discuss with others and make consistent.

    extra_outputs_dict = {
        "all_session_hists": all_session_hists,
        "all_session_chunked_hists": all_session_chunked_hists,
        "all_chunked_hist_stdevs": all_chunked_hist_stdevs
    }

    return all_motion_arrays, all_temporal_bin_centers, all_spatial_bin_centers, extra_outputs_dict


def _get_single_session_activity_histogram(recording, peaks, peak_locations, bin_um, method):
    """
    """
    accepted_methods = ["entire_session", "chunked_mean", "chunked_median", "chunked_supremum", "chunked_poisson"]
    assert method in ["entire_session", "chunked_mean", "chunked_median", "chunked_supremum", "chunked_poisson"], (
        f"`method` option must be one of: {accepted_methods}"

    )
    # First, get the histogram across the entire session
    entire_session_hist, temporal_bin_centers, spatial_bin_centers = alignment_utils.get_entire_session_hist(
        recording, peaks, peak_locations, bin_um
    )

    if method == "entire_session":
        return entire_session_hist, temporal_bin_centers, spatial_bin_centers, None, None

    # If method is not "entire_session", estimate the session
    # histogram based on histograms calculated from chunks.
    bin_s, percentile_lambda = alignment_utils.estimate_chunk_size(entire_session_hist, recording)

    chunked_session_hist, chunked_temporal_bin_centers, chunked_spatial_bins_centers = alignment_utils.get_chunked_histogram(
        recording, peaks, peak_locations, bin_s, bin_um
    )
    session_std = np.sum(np.std(chunked_session_hist, axis=0)) / chunked_session_hist.shape[1]

    # TODO: think about how to trim binsize. We can set to NaN and then use
    # Nan methods below. Otherwise, we can incorporate into every method e.g. as
    # above

    # TOOD: think about confidence weighting here.

    if method == "chunked_mean":
        summary_chunked_hist = alignment_utils.get_chunked_hist_mean(chunked_session_hist)

    elif method == "chunked_median":
        summary_chunked_hist = alignment_utils.get_chunked_hist_median(chunked_session_hist)

    elif method == "chunked_supremum":
        summary_chunked_hist = alignment_utils.get_chunked_hist_supremum(chunked_session_hist)

    elif method == "chunked_poisson":
        summary_chunked_hist = alignment_utils.get_chunked_hist_poisson_estimate(chunked_session_hist)

    return summary_chunked_hist, chunked_temporal_bin_centers, chunked_spatial_bins_centers, chunked_session_hist, session_std
