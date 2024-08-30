import numpy as np
from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
from spikeinterface.sortingcomponents.motion.motion_utils import get_spatial_windows, Motion, get_spatial_bins
from spikeinterface.sortingcomponents.motion.motion_interpolation import \
    correct_motion_on_peaks

import alignment_utils
from spikeinterface.preprocessing.motion import run_peak_detection_pipeline_node
import copy

"""
Trimmed_percentiles = (20, 80)  # TODO: this is originally in the context of Poisson estimation
if trimmed_percentiles is not False:
    min, max = trimmed_percentiles
    min_percentile = np.percentile(ks, min)
    max_percentile = np.percentile(ks, max)

    ks = ks[
        np.logical_and(ks >= min_percentile, ks <= max_percentile)
    ]

Easiest way is to NaN them out and skip in the compute methods.

"""

# -----------------------------------------------------------------------------
# Get Histograms
# -----------------------------------------------------------------------------

# 1) tidy up and check
# 2) expose trimmed versions, robust xcorr, akima interpolation, kriging interpolation / smoothing of histograms
# 3) write a final version that can take motion corrected or not.
# 4) write a list of things I'd like to do but dont have time to do now. Finish the generation function.


# 1) for now, assume it is a multi-session recording. Then,
#    add the peak-detection stuff if not later
# 2) for now, estimate entire session. But, soon make this chunked.
#    If peaks are already detected, might as well estimate from
#    entire session. Otherwise, we will want to add chunking as part of above.
# Expose a margin on the histogram?
# TODO: add a lot of confidence checks, e.g. that all sessions have the same contact positions.
"""
accepted_methods = ["entire_session", "chunked_mean", "chunked_median", "chunked_supremum", "chunked_poisson"]
assert method in ["entire_session", "chunked_mean", "chunked_median", "chunked_supremum", "chunked_poisson"], (
    f"`method` option must be one of: {accepted_methods}"
)
also
assert alignment_method in ["kilosort_like", "mean_crosscorr"], "TODO"
if peaks_list None, check peak_locations_list is none, etc, all same length etch
"""
"""
TODO: in this case, it is not necessary to run peak detection across
      the entire recording, would probably be sufficient to
      take a few chunks, of size determined by firing frequency of
      the neurons in the recording (or just take user defined size).
      For now, run on the entire recording and discuss the best way to
      run on chunked sections with Sam.
"""
# TOOD: add alignemnt method: "to_session_x"
# check if use_existing_spatial_bins that all are interpolate motion and all are the same bins

#    if not peaks_list:  TODO: expose as convenience function,arguments becomming overloaded
#        peaks_list, peak_locations_list = compute_peaks_and_locations_list(
#           recordings_list, "memory", {"method": "locally_exclusive"}, {"method": "monopolar_triangulation"}
#        )


def align_sessions(
        recordings_list,
        peaks_list,
        peak_locations_list,
        bin_um=2,
        histogram_estimation_method="chunked_mean",
        alignment_method="mean_crosscorr",
        chunked_bin_size_s="estimate",
        log_scale=True,
        rigid=True,
        apply_to_existing_motion_recording=False,  # TODO: name too long
        non_rigid_window_kwargs={},  # TODO: dict
):
    """
    """
    motion_estimates_list, temporal_bin_centers_list, spatial_bin_centers, spatial_bin_edges, non_rigid_window_centers, histogram_info = estimate_inter_session_displacement(
        recordings_list, peaks_list, peak_locations_list, bin_um, histogram_estimation_method, alignment_method, chunked_bin_size_s, log_scale, rigid, non_rigid_window_kwargs
    )

    corrected_recordings_list, motion_objects_list = _create_motion_recordings(
        recordings_list, motion_estimates_list, temporal_bin_centers_list, non_rigid_window_centers, apply_to_existing_motion_recording
    )

    corrected_peak_locations_list, corrected_session_histogram_list = _correct_session_displacement(
        corrected_recordings_list, peaks_list, peak_locations_list, spatial_bin_edges, log_scale
    )

    extra_outputs_dict = {
        "motion_estimates_list": motion_estimates_list,
        "temporal_bin_centers_list": temporal_bin_centers_list,
        "spatial_bin_centers": spatial_bin_centers,
        "spatial_bin_edges": spatial_bin_edges,  # TODO: sort out
        "non_rigid_window_centers": non_rigid_window_centers,
        "histogram_info": histogram_info,
        "corrected": {
            "corrected_peak_locations_list": corrected_peak_locations_list,
            "corrected_session_histogram_list": corrected_session_histogram_list,
        }
    }

    return corrected_recordings_list, motion_objects_list, extra_outputs_dict


# CHECKS


def align_sessions_after_motion_correction(recordings_list, motion_info_list, rigid, override_nonrigid_window_kwargs=False, **align_sessions_kwargs):
    """
    """
    # if both are nonrigid, but override is provided, error
    # if rigid is False but the orig recording is nonrigid, error
    # TOOD: assert the kwargs are all the same for all motion_info
    if "non_rigid_window_kwargs" in align_sessions_kwargs:
        raise ValueError("explain")

    if "apply_to_existing_motion_recording" in align_sessions_kwargs:
        if align_sessions_kwargs["apply_to_existing_motion_recording"] is False:
            raise ValueError("Explain")

        align_sessions_kwargs.pop("apply_to_existing_motion_recording")

    if override_nonrigid_window_kwargs:
        non_rigid_window_kwargs = override_nonrigid_window_kwargs
    else:
        non_rigid_window_kwargs = motion_info_list[0]["parameters"]["estimate_motion_kwargs"]
        non_rigid_window_kwargs.pop("method")
        non_rigid_window_kwargs.pop("rigid")
        assert non_rigid_window_kwargs.pop("direction") == "y"

    return align_sessions(
        recordings_list,
        [info["peaks"] for info in motion_info_list],
        [info["peak_locations"] for info in motion_info_list],
        rigid=rigid,
        non_rigid_window_kwargs=non_rigid_window_kwargs,
        apply_to_existing_motion_recording=True,
        **align_sessions_kwargs
    )

# CHECKS


def compute_peaks_and_locations_list(recording_list, gather_mode, detect_kwargs, localize_peaks_kwargs, job_kwargs):  # TODO: handle empty
    """
    """
    peaks_list = []
    peak_locations_list = []

    for recording in recording_list:
        peaks, peak_locations = run_peak_detection_pipeline_node(
            recording, gather_mode, detect_kwargs, localize_peaks_kwargs, job_kwargs
        )
        peaks_list.append(peaks)
        peak_locations_list.append(peak_locations)

    return peaks_list, peak_locations_list


def estimate_inter_session_displacement(
    recordings_list, peaks_list, peak_locations_list, bin_um, histogram_estimation_method, alignment_method, chunked_bin_size_s, log_scale, rigid, non_rigid_window_kwargs
):
    """
    # Next, estimate the alignment based on the activity histograms
    # TODO: "kilosort_like" does not work well in this context, can
    # probably remove after some benchmarking.
    """
    # Get spatial windows and estimate the session histograms
    spatial_bin_centers, spatial_bin_edges, contact_depths = get_spatial_bins(
        recordings_list[0], direction="y", hist_margin_um=0, bin_um=bin_um
    )

    non_rigid_windows, non_rigid_window_centers = get_spatial_windows(
        contact_depths,
        spatial_bin_centers,
        rigid,
        **non_rigid_window_kwargs,
    )

    temporal_bin_centers_list = []
    extra_outputs_dict = {
        "session_histogram_list": [],
        "chunked_histogram_list": [],
        "chunked_temporal_bin_centers": [],
        "chunked_histogram_stdevs_list": []
    }
    for recording, peaks, peak_locations in zip(recordings_list, peaks_list, peak_locations_list):

        session_hist, temporal_bin_centers, chunked_temporal_bin_centers, chunked_session_histogramsograms, chunked_histogram_stdevs = _get_single_session_activity_histogram(
                recording, peaks, peak_locations, histogram_estimation_method, spatial_bin_edges, log_scale, chunked_bin_size_s,
        )
        temporal_bin_centers_list.append(temporal_bin_centers)
        extra_outputs_dict["session_histogram_list"].append(session_hist)
        extra_outputs_dict["chunked_histogram_list"].append(chunked_session_histogramsograms)
        extra_outputs_dict["chunked_temporal_bin_centers"] = chunked_temporal_bin_centers
        extra_outputs_dict["chunked_histogram_stdevs_list"].append(chunked_histogram_stdevs)

    # Estimate the displacement from the session histograms
    if alignment_method == "kilosort_like":
        all_motion_arrays = alignment_utils.run_kilosort_like_rigid_registration(
            extra_outputs_dict["session_histogram_list"], non_rigid_windows
        ) * bin_um
    else:
        all_motion_arrays = alignment_utils.run_alignment_estimation(
            extra_outputs_dict["session_histogram_list"], spatial_bin_centers, non_rigid_windows, non_rigid_window_centers
        )
        all_motion_arrays *= bin_um

    return all_motion_arrays, temporal_bin_centers_list, spatial_bin_centers, spatial_bin_edges, non_rigid_window_centers, extra_outputs_dict


def _get_single_session_activity_histogram(recording, peaks, peak_locations, method, spatial_bin_edges, log_scale, chunked_bin_size_s):
    """
    """
    times = recording.get_times()
    temporal_bin_centers = np.atleast_1d((times[-1] + times[0]) / 2)

    if method == "entire_session" or chunked_bin_size_s == "estimate":

        entire_session_hist, chunked_temporal_bin_centers, _ = alignment_utils.get_entire_session_hist(
            recording, peaks, peak_locations, spatial_bin_edges,
            log_scale=False
        )
        if method == "entire_session":
            if log_scale:
                entire_session_hist = np.log10(1 + entire_session_hist)

            return entire_session_hist, temporal_bin_centers, chunked_temporal_bin_centers, None, None

    if chunked_bin_size_s == "estimate":
        # It is important that the passed histogram is scaled to firing rate
        chunked_bin_size_s, _ = alignment_utils.estimate_chunk_size(
            entire_session_hist
        )

    chunked_session_histograms, chunked_temporal_bin_centers, _ = alignment_utils.get_chunked_histogram(
        recording, peaks, peak_locations, chunked_bin_size_s, spatial_bin_edges, log_scale
    )
    session_std = np.sum(np.std(chunked_session_histograms, axis=0)) / chunked_session_histograms.shape[1]

    # TODO: handle_trim

    if method == "chunked_mean":
        summary_chunked_hist = alignment_utils.get_chunked_hist_mean(chunked_session_histograms)

    elif method == "chunked_median":
        summary_chunked_hist = alignment_utils.get_chunked_hist_median(chunked_session_histograms)

    elif method == "chunked_supremum":
        summary_chunked_hist = alignment_utils.get_chunked_hist_supremum(chunked_session_histograms)

    elif method == "chunked_poisson":
        summary_chunked_hist = alignment_utils.get_chunked_hist_poisson_estimate(chunked_session_histograms)

    return summary_chunked_hist, temporal_bin_centers, chunked_temporal_bin_centers, chunked_session_histograms, session_std


def _create_motion_recordings(all_recordings, motion_array, temporal_bin_centers_list, non_rigid_window_centers, apply_to_existing_motion_recording):
    """
    """
    interpolate_motion_kwargs = dict(
        border_mode="remove_channels", spatial_interpolation_method="kriging", sigma_um=20.0, p=2
    )
    corrected_recordings_list = []
    all_motions = []
    for i in range(len(all_recordings)):

        recording = all_recordings[i]
        ses_displacement = motion_array[i][np.newaxis, :]

        assert ses_displacement.shape[0] == 1, "time dimension should be 1 for session displacement"
        assert recording.get_num_segments() == 1, "TOOD: only support 1 segment"

        if apply_to_existing_motion_recording:

            if not isinstance(recording, InterpolateMotionRecording):
                raise ValueError("Explain")

            corrected_recording = _add_displacement_to_interpolate_recording(
                recording, ses_displacement, non_rigid_window_centers
            )
            # assert motion object is only 1 segment
            # TODO: need to handle the number of bins not being the same!
            # TODO: add an annotation to the corrected recording
            all_motions.append(None)
            corrected_recordings_list.append(corrected_recording)
        else:
            motion = Motion(
                [ses_displacement],
                [temporal_bin_centers_list[i]],
                non_rigid_window_centers,
                direction="y"
            )

            all_motions.append(motion)

            corrected_recording = InterpolateMotionRecording(
                recording, motion, **interpolate_motion_kwargs
            )
            corrected_recordings_list.append(corrected_recording)

    return corrected_recordings_list, all_motions


# TODO: this could probably go in motion_utils.py
def _add_displacement_to_interpolate_recording(recording, new_displacement, new_non_rigid_window_centers):
    """
    """
    corrected_recording = copy.deepcopy(recording)

    corrected_recording_motion = corrected_recording._recording_segments[0].motion
    recording_bins = corrected_recording_motion.displacement[0].shape[1]

    # If the new displacement is a scalar (i.e. rigid),
    # just add it to the existing displacements
    if new_displacement.shape[1] == 1:
        corrected_recording_motion.displacement[0] += new_displacement[0, 0]

    else:
        if recording_bins == 1:
            # If the new displacement is nonrigid (multiple windows) but the motion
            # recording is rigid, we update the displacement at all time bins
            # with the new, nonrigid displacement added to the old, rigid displacement.
            # TODO: check + ask Sam if any other fields need to be chagned. This is a little
            # hairy so test thoroughly.

            num_time_bins = corrected_recording_motion.displacement[0].shape[0]
            tiled_nonrigid_displacement = np.repeat(new_displacement, num_time_bins, axis=0)
            new_displacement = tiled_nonrigid_displacement + corrected_recording_motion.displacement

            corrected_recording_motion.displacement = new_displacement
            corrected_recording_motion.spatial_bins_um = new_non_rigid_window_centers
        else:
            # Otherwise, if both the motion and new displacement are
            # nonrigid, we need to make sure the nonrigid windows
            # match exactly.
            assert np.array_equal(
                corrected_recording_motion.spatial_bins_um,
                new_non_rigid_window_centers
            )
            assert corrected_recording_motion.displacement[0].shape[1] == new_displacement.shape[1]

            corrected_recording_motion.displacement[0] += new_displacement

    return corrected_recording


def _correct_session_displacement(
        recordings_list, peaks_list, peak_locations_list, spatial_bin_edges, log_scale
):
    """
    """
    num_sessions = len(recordings_list)

    # Correct the peak locations
    corrected_peak_locations_list = []
    for ses_idx in range(num_sessions):

        assert recordings_list[ses_idx].get_num_segments() == 1, "TODO"

        corrected_peaks = correct_motion_on_peaks(
            peaks_list[ses_idx],
            peak_locations_list[ses_idx],
            recordings_list[ses_idx]._recording_segments[0].motion,
            recordings_list[ses_idx],
        )
        corrected_peak_locations_list.append(corrected_peaks)

    # Create a corrected histogram based on corrected peak locations
    corrected_session_histogram_list = []
    for ses_idx in range(num_sessions):

        corrected_histogram = alignment_utils.get_entire_session_hist(
            recordings_list[ses_idx],
            peaks_list[ses_idx],
            corrected_peak_locations_list[ses_idx],
            spatial_bin_edges,
            log_scale
        )[0]
        corrected_session_histogram_list.append(corrected_histogram)

    return corrected_peak_locations_list, corrected_session_histogram_list
