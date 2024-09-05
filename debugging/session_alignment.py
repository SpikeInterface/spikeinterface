import numpy as np
from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
from spikeinterface.sortingcomponents.motion.motion_utils import get_spatial_windows, Motion, get_spatial_bins
from spikeinterface.sortingcomponents.motion.motion_interpolation import \
    correct_motion_on_peaks

import alignment_utils
from spikeinterface.preprocessing.motion import run_peak_detection_pipeline_node
import copy

# TODO: update based on the issue
# -----------------------------------------------------------------------------
# Default Settings
# -----------------------------------------------------------------------------


def get_estimate_histogram_kwargs():
    return {
        "bin_um": 2,
        "method": "chunked_mean",
        "chunked_bin_size_s": "estimate",
        "log_scale": False,
        "non_rigid_window_kwargs": {
                "win_shape": "gaussian",
                "win_step_um": 50.0,
                "win_scale_um": 150.0,
                "win_margin_um": None,
                "zero_threshold": None,
            },
        }


def get_alignment_method_kwargs():
    return {
        "num_shifts_block": 5,
        "interpolate": False,
        "interp_factor": 10,
        "kriging_sigma": 1,
        "kriging_p": 2,
        "kriging_d": 2,
        "smoothing_sigma_bin": 0.5,
        "smoothing_sigma_window": 0.5,
        "akima_interp_nonrigid": False,
    }


def get_interpolate_motion_kwargs():
    return {
        "border_mode": "remove_channels",
        "spatial_interpolation_method": "kriging",
        "sigma_um": 20.0,
        "p": 2
    }

# -----------------------------------------------------------------------------
# Public Entry Level Functions
# -----------------------------------------------------------------------------


def align_sessions(
        recordings_list,
        peaks_list,
        peak_locations_list,
        alignment_order,
        rigid,
        akima_interp_nonrigid=False,
        estimate_histogram_kwargs=None,
        alignment_method_kwargs=None,
        interpolate_motion_kwargs=None,
):
    """
    print what happens with the update! it is automaticlally added
    to existing motion if it exists.
    """
    if estimate_histogram_kwargs is None:
        estimate_histogram_kwargs = get_estimate_histogram_kwargs()

    if alignment_method_kwargs is None:
        alignment_method_kwargs = get_alignment_method_kwargs()

    if interpolate_motion_kwargs is None:
        interpolate_motion_kwargs = get_interpolate_motion_kwargs()

    session_histogram_list, bins, histogram_info = _compute_session_histograms(
        recordings_list, peaks_list, peak_locations_list, rigid, **estimate_histogram_kwargs
    )

    # Estimate the displacement from the session histograms
    if rigid:
        shifts_array = _compute_rigid_alignment(
            np.array(session_histogram_list), alignment_order, alignment_method_kwargs
        )
    else:
        shifts_array, bins["non_rigid_window_centers"] = _compute_nonrigid_alignment(
            np.array(session_histogram_list), bins, alignment_order, alignment_method_kwargs, akima_interp_nonrigid
        )
    shifts_array *= estimate_histogram_kwargs["bin_um"]

    # Apply the motion correct, either generating new recordings or applying to
    # existing recording if InterpolateMotionRecording
    corrected_recordings_list, motion_objects_list = _create_motion_recordings(
        recordings_list, shifts_array, bins, interpolate_motion_kwargs
    )

    # Finally,  create corrected peak locations and histogram for assessment.
    corrected_peak_locations_list, corrected_session_histogram_list = _correct_session_displacement(
        corrected_recordings_list, peaks_list, peak_locations_list,
        bins["spatial_bin_edges"],
        estimate_histogram_kwargs["log_scale"],
    )

    extra_outputs_dict = {
        "shifts_array": shifts_array,
        "session_histogram_list": session_histogram_list,
        "bins": bins,
        "histogram_info": histogram_info,
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
    rigid,
    non_rigid_window_kwargs,
    bin_um,
    method,
    chunked_bin_size_s,
    log_scale,
):
    """
    """
    # Get spatial windows and estimate the session histograms
    bins = {"temporal_bin_centers_list": []}
    bins["spatial_bin_centers"], bins["spatial_bin_edges"], contact_depths = get_spatial_bins(
        recordings_list[0], direction="y", hist_margin_um=0, bin_um=bin_um
    )

    bins["non_rigid_windows"], bins["non_rigid_window_centers"] = get_spatial_windows(
        contact_depths,
        bins["spatial_bin_centers"],
        rigid,
        **non_rigid_window_kwargs,
    )

    session_histogram_list = []
    histogram_info_list = []

    for recording, peaks, peak_locations in zip(recordings_list, peaks_list, peak_locations_list):

        session_hist, temporal_bin_centers, histogram_info = _get_single_session_activity_histogram(
            recording, peaks, peak_locations, bins["spatial_bin_edges"],
            method, log_scale, chunked_bin_size_s,
        )
        bins["temporal_bin_centers_list"].append(temporal_bin_centers)
        session_histogram_list.append(session_hist)
        histogram_info_list.append(histogram_info)

    return session_histogram_list, bins, histogram_info_list


def _get_single_session_activity_histogram(
        recording, peaks, peak_locations, spatial_bin_edges, method, log_scale, chunked_bin_size_s
):
    """
    TODO: fix this, just estimate from some chunks? hmmm
    """
    times = recording.get_times()
    temporal_bin_centers = np.atleast_1d((times[-1] + times[0]) / 2)

    if method == "entire_session" or chunked_bin_size_s == "estimate":

        one_bin_histogram, _, _ = alignment_utils.get_activity_histogram(
            recording, peaks, peak_locations, spatial_bin_edges,
            log_scale=False, bin_s=None
        )
        if method == "entire_session":
            if log_scale:
                one_bin_histogram = np.log10(1 + one_bin_histogram)

            return one_bin_histogram.squeeze(), temporal_bin_centers, None

    if chunked_bin_size_s == "estimate":
        # It is important that the passed histogram is scaled to firing rate
        chunked_bin_size_s, _ = alignment_utils.estimate_chunk_size(
            one_bin_histogram
        )

    chunked_histograms, chunked_temporal_bin_centers, _ = alignment_utils.get_activity_histogram(
        recording, peaks, peak_locations, spatial_bin_edges, log_scale, bin_s=chunked_bin_size_s,
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


def _create_motion_recordings(all_recordings, motion_array, bins, interpolate_motion_kwargs):
    """
    # assert motion object is only 1 segment
    # TODO: need to handle the number of bins not being the same!
    # TODO: add an annotation to the corrected recording
    """
    # assert ses_displacement.shape[0] == 1, "time dimension should be 1 for session displacement"  # TODO own checking function
    # assert recording.get_num_segments() == 1, "TOOD: only support 1 segment"

    corrected_recordings_list = []
    all_motions = []
    for i in range(len(all_recordings)):

        recording = all_recordings[i]
        ses_displacement = motion_array[i][np.newaxis, :]

        if isinstance(recording, InterpolateMotionRecording):

            corrected_recording = _add_displacement_to_interpolate_recording(
                recording, ses_displacement, bins["non_rigid_window_centers"]
            )
            all_motions.append(None)
        else:
            motion = Motion(
                [ses_displacement],
                [bins["temporal_bin_centers_list"][i]],
                bins["non_rigid_window_centers"],
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
        recordings_list, peaks_list, peak_locations_list, spatial_bin_edges, log_scale
):
    """
    """
    num_sessions = len(recordings_list)

    # Correct the peak locations
    corrected_peak_locations_list = []
    for ses_idx in range(num_sessions):

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

        corrected_histogram = alignment_utils.get_activity_histogram(
            recordings_list[ses_idx],
            peaks_list[ses_idx],
            corrected_peak_locations_list[ses_idx],
            spatial_bin_edges,
            log_scale,
            bin_s=None,
        )[0]

        corrected_session_histogram_list.append(
            corrected_histogram.squeeze()
        )

    return corrected_peak_locations_list, corrected_session_histogram_list


def _compute_rigid_alignment(
    session_histogram_array,
    alignment_order,
    alignment_method_kwargs,
):
    """
    # TODO: this interpolate, smooth for both rigid and non-rigid. Check this is ok
    # maybe we only want to apply the smoothings etc for nonrigid like KS motion correction
    """
    alignment_method_kwargs = copy.deepcopy(alignment_method_kwargs)
    alignment_method_kwargs["num_shifts_block"] = False

    rigid_window = np.ones_like(session_histogram_array[0, :])[np.newaxis, :]

    rigid_session_offsets_matrix = alignment_utils.compute_histogram_crosscorrelation(
        session_histogram_array,
        rigid_window,
        **alignment_method_kwargs,
    )
    optimal_shift_indices = _get_shifts_from_session_matrix(
        alignment_order, rigid_session_offsets_matrix
    )
    return optimal_shift_indices


def _compute_nonrigid_alignment(
    session_histogram_array,
    bins,
    alignment_order,
    alignment_method_kwargs,
    akima_interp_nonrigid,
):
    """ """
    rigid_shifts = _compute_rigid_alignment(
        session_histogram_array,
        alignment_order,
        alignment_method_kwargs,
    )

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
        bins["non_rigid_windows"],
        **alignment_method_kwargs
    )
    non_rigid_shifts = _get_shifts_from_session_matrix(
        alignment_order, nonrigid_session_offsets_matrix
    )

    # Interpolate the nonrigid bins if required.
    if akima_interp_nonrigid:
        interp_nonrigid_shifts = _akima_interpolate_nonrigid_shifts(
            non_rigid_shifts, bins["non_rigid_window_centers"], bins["spatial_bin_centers"]
        )
        shifts = rigid_shifts + interp_nonrigid_shifts
        non_rigid_window_centers = bins["spatial_bin_centers"]
    else:
        shifts = rigid_shifts + non_rigid_shifts
        non_rigid_window_centers = bins["non_rigid_window_centers"]

    return shifts, non_rigid_window_centers

# TODO: tidy this up
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


def _get_shifts_from_session_matrix(alignment_order, session_offsets_matrix):  # TODO: rename
    """
    # TODO: this doesn't do the right thing!!!
    # optimal_shift_indices = -np.mean(session_offsets_matrix, axis=0)  # TODO: these are not symmetrical because of interpolation?
    # TODO: carefully check this! this puts to the center of all points.
    # Maybe we want to optimise such that all shifts are similar or closed form...
    To middle is pretty rough here, can think of more ways to weight the center
    chosen (i.e. middle between min / max, mean position etc). ALso to be
    # robust in the true estimation of differences.
    """
    if alignment_order == "to_middle":  # TODO: do a lot of arg checks
        optimal_shift_indices = -np.mean(
            session_offsets_matrix, axis=0
        )  # TOOD: pretty sure this is not correct, go back and figure out what is optimal
    else:
        ses_idx = int(alignment_order.split("_")[-1]) - 1
        optimal_shift_indices = -session_offsets_matrix[ses_idx, :, :]

    return optimal_shift_indices
