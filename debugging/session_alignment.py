import numpy as np
from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
from spikeinterface.sortingcomponents.motion.motion_utils import get_spatial_windows, Motion, get_spatial_bins
from spikeinterface.sortingcomponents.motion.motion_interpolation import \
    correct_motion_on_peaks

import alignment_utils
from spikeinterface.preprocessing.motion import run_peak_detection_pipeline_node
import copy


# TODO: need to plot out the entire call tree and
# make sure it is optimal as its quite complex but
# there is quite a lot of weird stuff here.


_estimate_histogram_kwargs = {
    "bin_um": 2,
    "method": "chunked_mean",
    "chunked_bin_size_s": "estimate",
    "log_scale": False,
    "smooth_um": None,
}


_compute_alignment_kwargs = {
    "num_shifts_block": 5,
    "interpolate": False,
    "interp_factor": 10,
    "kriging_sigma": 1,
    "kriging_p": 2,
    "kriging_d": 2,
    "smoothing_sigma_bin": 0.5,
    "smoothing_sigma_window": 0.5,

    "non_rigid_window_kwargs": {
        "win_shape": "gaussian",
        "win_step_um": 50,
        "win_scale_um": 50,
        "win_margin_um": None,
        "zero_threshold": None,
    },
}


_interpolate_motion_kwargs = {
    "border_mode": "remove_channels",
    "spatial_interpolation_method": "kriging",
    "sigma_um": 20.0,
    "p": 2
}

# -----------------------------------------------------------------------------
# Public Entry Level Functions
# -----------------------------------------------------------------------------

# TODO: add some print statements for progress

def align_sessions(
        recordings_list,
        peaks_list,
        peak_locations_list,
        alignment_order="to_middle",
        rigid=True,
        estimate_histogram_kwargs=_estimate_histogram_kwargs,
        compute_alignment_kwargs=_compute_alignment_kwargs,
        interpolate_motion_kwargs=_interpolate_motion_kwargs,
):
    """
    print what happens with the update! it is automaticlally added
    to existing motion if it exists.
    """
    estimate_histogram_kwargs = copy.deepcopy(estimate_histogram_kwargs)
    compute_alignment_kwargs = copy.deepcopy(compute_alignment_kwargs)
    interpolate_motion_kwargs = copy.deepcopy(interpolate_motion_kwargs)

    _check_align_sesssions_inpus(recordings_list, peaks_list, peak_locations_list,
                                 alignment_order, estimate_histogram_kwargs)

    # Compute a single activity histogram from each session
    (session_histogram_list, temporal_bin_centers_list,
     spatial_bin_centers, spatial_bin_edges, histogram_info_list) = _compute_session_histograms(
        recordings_list, peaks_list, peak_locations_list, **estimate_histogram_kwargs
    )

    # Align the activity histograms across sessions
    contact_depths = recordings_list[0].get_channel_locations()[:, 1]  # "y" dim.

    shifts_array, non_rigid_windows, non_rigid_window_centers = _compute_session_alignment(
        session_histogram_list, contact_depths, spatial_bin_centers,
        alignment_order, rigid, compute_alignment_kwargs,
    )
    shifts_array *= estimate_histogram_kwargs["bin_um"]

    # Apply the motion correct, either generating new recordings or applying to
    # existing recording if InterpolateMotionRecording
    corrected_recordings_list, motion_objects_list = _create_motion_recordings(
        recordings_list, shifts_array, temporal_bin_centers_list, non_rigid_window_centers, interpolate_motion_kwargs
    )

    # Finally, create corrected peak locations and histogram for assessment.
    corrected_peak_locations_list, corrected_session_histogram_list = _correct_session_displacement(
        corrected_recordings_list, peaks_list, peak_locations_list,
        spatial_bin_edges, estimate_histogram_kwargs
    )

    extra_outputs_dict = {
        "shifts_array": shifts_array,
        "session_histogram_list": session_histogram_list,
        "spatial_bin_centers": spatial_bin_centers,
        "temporal_bin_centers_list": temporal_bin_centers_list,
        "non_rigid_window_centers": non_rigid_window_centers,
        "non_rigid_windows": non_rigid_windows,
        "histogram_info_list": histogram_info_list,
        "corrected": {
            "corrected_peak_locations_list": corrected_peak_locations_list,
            "corrected_session_histogram_list": corrected_session_histogram_list,
        }
    }
    return corrected_recordings_list, motion_objects_list, extra_outputs_dict


def align_sessions_after_motion_correction(
        recordings_list, motion_info_list, rigid, **align_sessions_kwargs
):
    """
    # if both are nonrigid, but override is provided, error
    # if rigid is False but the orig recording is nonrigid, error
    # TOOD: assert the kwargs are all the same for all motion_info
    """
    non_rigid_window_kwargs = motion_info_list[0]["parameters"]["estimate_motion_kwargs"]
    non_rigid_window_kwargs.pop("method")
    non_rigid_window_kwargs.pop("rigid")
    assert non_rigid_window_kwargs.pop("direction") == "y"

    return align_sessions(
        recordings_list,
        [info["peaks"] for info in motion_info_list],
        [info["peak_locations"] for info in motion_info_list],
        rigid=rigid,
        **align_sessions_kwargs
    )


def compute_peaks_for_session_alignment(
    recording_list, gather_mode, detect_kwargs, localize_peaks_kwargs, job_kwargs
):
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


# -----------------------------------------------------------------------------
# Private Functions
# -----------------------------------------------------------------------------


def _compute_session_histograms(
    recordings_list,
    peaks_list,
    peak_locations_list,
    bin_um,
    method,
    chunked_bin_size_s,
    smooth_um,
    log_scale,
):
    """
    """
    # Get spatial windows and estimate the session histograms
    temporal_bin_centers_list = []

    spatial_bin_centers, spatial_bin_edges, _ = get_spatial_bins(
        recordings_list[0], direction="y", hist_margin_um=0, bin_um=bin_um
    )

    session_histogram_list = []
    histogram_info_list = []

    for recording, peaks, peak_locations in zip(recordings_list, peaks_list, peak_locations_list):

        session_hist, temporal_bin_centers, histogram_info = _get_single_session_activity_histogram(
            recording, peaks, peak_locations, spatial_bin_edges,
            method, log_scale, chunked_bin_size_s, smooth_um,
        )
        temporal_bin_centers_list.append(temporal_bin_centers)
        session_histogram_list.append(session_hist)
        histogram_info_list.append(histogram_info)

    return session_histogram_list, temporal_bin_centers_list, spatial_bin_centers, spatial_bin_edges, histogram_info_list


def _get_single_session_activity_histogram(
        recording, peaks, peak_locations, spatial_bin_edges, method, log_scale, chunked_bin_size_s, smooth_um
):
    """
    """
    times = recording.get_times()
    temporal_bin_centers = np.atleast_1d((times[-1] + times[0]) / 2)

    # Estimate a entire session histogram if requested or doing
    # full estimation for chunked bin size
    if method == "entire_session" or chunked_bin_size_s == "estimate":

        one_bin_histogram, _, _ = alignment_utils.get_activity_histogram(
            recording, peaks, peak_locations, spatial_bin_edges,
            log_scale=False, bin_s=None, smooth_um=smooth_um
        )
        if method == "entire_session":
            if log_scale:
                one_bin_histogram = np.log10(1 + one_bin_histogram)

            return one_bin_histogram.squeeze(), temporal_bin_centers, None

    # Compute summary histogram based on histograms
    # calculated on session chunks
    if chunked_bin_size_s == "estimate":
        # It is important that the passed histogram is scaled to firing rate
        chunked_bin_size_s, _ = alignment_utils.estimate_chunk_size(
            one_bin_histogram
        )

    chunked_histograms, chunked_temporal_bin_centers, _ = alignment_utils.get_activity_histogram(
        recording, peaks, peak_locations, spatial_bin_edges, log_scale, bin_s=chunked_bin_size_s, smooth_um=smooth_um,
    )
    session_std = np.sum(np.std(chunked_histograms, axis=0)) / chunked_histograms.shape[1]

    if method == "chunked_mean":
        session_histogram = alignment_utils.get_chunked_hist_mean(chunked_histograms)

    elif method == "chunked_median":
        session_histogram = alignment_utils.get_chunked_hist_median(chunked_histograms)

    elif method == "chunked_supremum":
        session_histogram = alignment_utils.get_chunked_hist_supremum(chunked_histograms)

    elif method == "chunked_poisson":
        session_histogram = alignment_utils.get_chunked_hist_poisson_estimate(chunked_histograms)

    histogram_info = {
        "chunked_histograms": chunked_histograms,
        "chunked_temporal_bin_centers": chunked_temporal_bin_centers,
        "session_std": session_std,
        "chunked_bin_size_s": chunked_bin_size_s,
    }

    return session_histogram, temporal_bin_centers, histogram_info


def _create_motion_recordings(
        all_recordings, motion_array, temporal_bin_centers_list, non_rigid_window_centers, interpolate_motion_kwargs
):
    """
    # TODO: add an annotation to the corrected recording
    """
    assert all(array.ndim == 1 for array in motion_array), (
        "time dimension should be 1 for session displacement"
    )

    corrected_recordings_list = []
    all_motions = []
    for i in range(len(all_recordings)):

        recording = all_recordings[i]
        ses_displacement = motion_array[i][np.newaxis, :]

        if isinstance(recording, InterpolateMotionRecording):

            corrected_recording = _add_displacement_to_interpolate_recording(
                recording, ses_displacement, non_rigid_window_centers
            )
            all_motions.append(None)
        else:
            motion = Motion(
                [ses_displacement],
                [temporal_bin_centers_list[i]],
                non_rigid_window_centers,
                direction="y"
            )
            corrected_recording = InterpolateMotionRecording(
                recording, motion, **interpolate_motion_kwargs
            )
            all_motions.append(motion)

        corrected_recordings_list.append(corrected_recording)

    return corrected_recordings_list, all_motions


def _add_displacement_to_interpolate_recording(recording, new_displacement, new_non_rigid_window_centers):
    """
    # TODO: check + ask Sam if any other fields need to be chagned. This is a little
    # hairy (4 possible combinations of new and
    # old displacement shapes, rigid or nonrigid, so test thoroughly.
    """
    # Everything is done in place, so keep a short variable
    # name reference to the new recordings `motion` object
    # and update it.okay
    corrected_recording = copy.deepcopy(recording)

    motion_ref = corrected_recording._recording_segments[0].motion
    recording_bins = motion_ref.displacement[0].shape[1]

    # If the new displacement is a scalar (i.e. rigid),
    # just add it to the existing displacements
    if new_displacement.shape[1] == 1:
        motion_ref.displacement[0] += new_displacement[0, 0]

    else:
        if recording_bins == 1:
            # If the new displacement is nonrigid (multiple windows) but the motion
            # recording is rigid, we update the displacement at all time bins
            # with the new, nonrigid displacement added to the old, rigid displacement.
            num_time_bins = motion_ref.displacement[0].shape[0]
            tiled_nonrigid_displacement = np.repeat(new_displacement, num_time_bins, axis=0)
            new_displacement = tiled_nonrigid_displacement + motion_ref.displacement

            motion_ref.displacement = new_displacement
            motion_ref.spatial_bins_um = new_non_rigid_window_centers
        else:
            # Otherwise, if both the motion and new displacement are
            # nonrigid, we need to make sure the nonrigid windows
            # match exactly.
            assert np.array_equal(
                motion_ref.spatial_bins_um,
                new_non_rigid_window_centers
            )
            assert motion_ref.displacement[0].shape[1] == new_displacement.shape[1]

            motion_ref.displacement[0] += new_displacement

    return corrected_recording


def _correct_session_displacement(
        recordings_list, peaks_list, peak_locations_list, spatial_bin_edges, estimate_histogram_kwargs,
):
    """
    """
    # Correct the peak locations
    corrected_peak_locations_list = []

    for recording, peaks, peak_locations in zip(recordings_list, peaks_list, peak_locations_list):

        corrected_peak_locs = correct_motion_on_peaks(
            peaks,
            peak_locations,
            recording._recording_segments[0].motion,
            recording,
        )
        corrected_peak_locations_list.append(corrected_peak_locs)

    corrected_session_histogram_list = []

    for recording, peaks, corrected_locations in zip(
            recordings_list, peaks_list, corrected_peak_locations_list
    ):
        session_hist, _, _ = _get_single_session_activity_histogram(
            recording,
            peaks,
            corrected_locations,
            spatial_bin_edges,
            estimate_histogram_kwargs["method"],
            estimate_histogram_kwargs["log_scale"],
            estimate_histogram_kwargs["chunked_bin_size_s"],
            estimate_histogram_kwargs["smooth_um"],
        )
        corrected_session_histogram_list.append(session_hist)

    return corrected_peak_locations_list, corrected_session_histogram_list


def _compute_session_alignment(
    session_histogram_list, contact_depths, spatial_bin_centers, alignment_order, rigid, compute_alignment_kwargs,
):
    session_histogram_array = np.array(session_histogram_list)

    non_rigid_window_kwargs = compute_alignment_kwargs.pop("non_rigid_window_kwargs")  # TODO: copy this somewhere
    akima_interp_nonrigid = compute_alignment_kwargs.pop("akima_interp_nonrigid")

    non_rigid_windows, non_rigid_window_centers = get_spatial_windows(
        contact_depths,
        spatial_bin_centers,
        rigid,
        **non_rigid_window_kwargs
    )

    rigid_shifts = _estimate_rigid_alignment(
        session_histogram_array,
        alignment_order,
        compute_alignment_kwargs,
    )

    if rigid:
        return rigid_shifts, non_rigid_windows, non_rigid_window_centers

    # For non-rigid, first shift the histograms according to the rigid shift
    shifted_histograms = np.zeros_like(session_histogram_array)
    for ses_idx in range(session_histogram_array.shape[0]):

        shifted_histogram = alignment_utils.shift_array_fill_zeros(
            array=session_histogram_array[ses_idx, :],
            shift=int(rigid_shifts[ses_idx, 0])
        )
        shifted_histograms[ses_idx, :] = shifted_histogram

    # Then compute the nonrigid shifts
    nonrigid_session_offsets_matrix = alignment_utils.compute_histogram_crosscorrelation(
        shifted_histograms,
        non_rigid_windows,
        **compute_alignment_kwargs
    )
    non_rigid_shifts = _get_shifts_from_session_matrix(
        alignment_order, nonrigid_session_offsets_matrix
    )

    # Akima interpolate the nonrigid bins if required.
    if akima_interp_nonrigid:
        interp_nonrigid_shifts = _akima_interpolate_nonrigid_shifts(
            non_rigid_shifts, non_rigid_window_centers, spatial_bin_centers
        )
        shifts = rigid_shifts + interp_nonrigid_shifts
        non_rigid_window_centers = spatial_bin_centers
    else:
        shifts = rigid_shifts + non_rigid_shifts
        non_rigid_window_centers = non_rigid_window_centers

    return shifts, non_rigid_windows, non_rigid_window_centers


def _estimate_rigid_alignment(
    session_histogram_array,
    alignment_order,
    compute_alignment_kwargs,
):
    """
    """
    compute_alignment_kwargs = copy.deepcopy(compute_alignment_kwargs)
    compute_alignment_kwargs["num_shifts_block"] = False

    rigid_window = np.ones_like(session_histogram_array[0, :])[np.newaxis, :]

    rigid_session_offsets_matrix = alignment_utils.compute_histogram_crosscorrelation(
        session_histogram_array,
        rigid_window,
        **compute_alignment_kwargs,
    )
    optimal_shift_indices = _get_shifts_from_session_matrix(
        alignment_order, rigid_session_offsets_matrix
    )
    return optimal_shift_indices


def _akima_interpolate_nonrigid_shifts(non_rigid_shifts, non_rigid_window_centers, spatial_bin_centers):
    """
    TODO: requires scipy 14
    """
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


def _get_shifts_from_session_matrix(alignment_order, session_offsets_matrix):
    """
    """
    if alignment_order == "to_middle":
        optimal_shift_indices = -np.mean(
            session_offsets_matrix, axis=0
        )
    else:
        ses_idx = int(alignment_order.split("_")[-1]) - 1
        optimal_shift_indices = -session_offsets_matrix[ses_idx, :, :]

    return optimal_shift_indices


# -----------------------------------------------------------------------------
# Checkers
# -----------------------------------------------------------------------------

def _check_align_sesssions_inpus(
        recordings_list, peaks_list, peak_locations_list, alignment_order, estimate_histogram_kwargs
):
    """"""
    num_sessions = len(recordings_list)

    if len(peaks_list) != num_sessions or len(peak_locations_list) != num_sessions:
        raise ValueError("`recordings_list`, `peaks_list` and `peak_locations_list` "
                         "be the same length. They must contains list of corresponding "
                         "recordings, peak and peak location objects.")

    if not all(rec.get_num_segments() == 1 for rec in recordings_list):
        raise ValueError("Multi-segment recordings not supported. All recordings "
                         "in `recordings_list` but have only 1 segment.")

    channel_locs = [rec.get_channel_locations() for rec in recordings_list]
    if not all(np.array_equal(locs, channel_locs[0]) for locs in channel_locs):
        raise ValueError("The recordings in `recordings_list` do not all have "
                         "the same channel locations. All recordings must be "
                         "performed using the same probe.")

    accepted_hist_methods = ["entire_session", "chunked_mean", "chunked_median", "chunked_supremum", "chunked_poisson"]
    method = estimate_histogram_kwargs["method"]
    if not method in ["entire_session", "chunked_mean", "chunked_median", "chunked_supremum", "chunked_poisson"]:
        raise ValueError(f"`method` option must be one of: {accepted_hist_methods}")

    if alignment_order != "to_middle":

        split_name = alignment_order.split("_")
        if not "_".join(split_name[:2]) == "to_session":
            raise ValueError("`alignment_order` must take the form 'to_sesion_X'"
                             "where X is the session number to align to.")

        ses_num = int(split_name[-1])
        if ses_num > num_sessions:
            raise ValueError(f"`alignment_order` session {ses_num} is larger than"
                             f"the number of sessions in `recordings_list`.")

        if ses_num == 0:
            raise ValueError("`alignment_order` required the session number, "
                             "not session index.")
