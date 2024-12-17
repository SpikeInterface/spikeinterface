from __future__ import annotations

import copy

import numpy as np
import json
from pathlib import Path
import time

from spikeinterface.core.baserecording import BaseRecording
from spikeinterface.core import get_noise_levels, fix_job_kwargs, get_random_data_chunks
from spikeinterface.core.job_tools import _shared_job_kwargs_doc
from spikeinterface.core.core_tools import SIJsonEncoder
from spikeinterface.core.job_tools import _shared_job_kwargs_doc

# TODO: update motion docstrings around the 'select' step.


# TODO:
# 1) detect peaks and peak locations if not already provided.
#       - could use only a subset of data, for ease now just estimate
#         everything on the entire dataset
# 2) Calcualte the activity histogram across the entire session
#       - will be better ways to estimate this, i.e. from the end
#         of the session, from periods of stability, etc.
#         taking a weighted average of histograms
# 3) Optimise for drift correction for each session across
#    all histograms, minimising lost data at edges and keeping
#    shift similar for all sessions. Could alternatively shift
#    to the average histogram but this seems like a bad idea.
# 4) Store the motion vectors, ether adding to existing (of motion
#    objects passed) otherwise.


def correct_inter_session_displacement(
    recordings_list: list[BaseRecording],
    existing_motion_info: Optional[list[Dict]] = None,
    keep_channels_constant=False,
    detect_kwargs={},  # TODO: make non-mutable (same for motion.py)
    select_kwargs={},
    localize_peaks_kwargs={},
    job_kwargs={},
):
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks, detect_peak_methods
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks, detect_peak_methods
    from spikeinterface.sortingcomponents.peak_selection import select_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks, localize_peak_methods
    from spikeinterface.sortingcomponents.motion.motion_estimation import estimate_motion
    from spikeinterface.sortingcomponents.motion.motion_interpolation import InterpolateMotionRecording
    from spikeinterface.core.node_pipeline import ExtractDenseWaveforms, run_node_pipeline
    from spikeinterface.sortingcomponents.motion.motion_utils import Motion, get_spatial_windows

    # TODO: do not accept multi-segment recordings.
    # TODO: check all recordings have the same probe dimensions!
    # Check if exsting_motion_info is passed then the recordings have the motion vector (I guess this is stored somewhere? maybe it is on the motion object)
    if existing_motion_info is not None:
        if not isinstance(existing_motion_info, list) and len(recordings_list) != len(existing_motion_info):
            raise ValueError(
                "`estimate_motion_info` if provided, must be"
                "a list of `motion_info` with each associated with"
                "the corresponding recording in `recordings_list`."
            )

    # TODO: do not handle select peaks option yet as probably better to chunk
    # rather than select peaks? no sure can discuss.
    if existing_motion_info is None:

        peaks_list = []
        peak_locations_list = []

        for recording in recordings_list:
            # TODO: this is a direct copy from motion.detect_motion().
            # Factor into own function in motion.py
            gather_mode = "memory"
            # node detect
            method = detect_kwargs.pop("method", "locally_exclusive")
            method_class = detect_peak_methods[method]
            node0 = method_class(recording, **detect_kwargs)

            node1 = ExtractDenseWaveforms(recording, parents=[node0], ms_before=0.1, ms_after=0.3)

            # node detect + localize
            method = localize_peaks_kwargs.pop("method", "center_of_mass")
            method_class = localize_peak_methods[method]
            node2 = method_class(recording, parents=[node0, node1], return_output=True, **localize_peaks_kwargs)
            pipeline_nodes = [node0, node1, node2]

            peaks, peak_locations = run_node_pipeline(
                recording,
                pipeline_nodes,
                job_kwargs,
                job_name="detect and localize",
                gather_mode=gather_mode,
                gather_kwargs=None,
                squeeze_output=False,
                folder=None,
                names=None,
            )
            peaks_list.append(peaks)
            peak_locations_list.append(peak_locations)
    else:
        peaks_list = [info["peaks"] for info in existing_motion_info]
        peak_locations_list = [info["peak_locations"] for info in existing_motion_info]

    from spikeinterface.sortingcomponents.motion.motion_utils import make_2d_motion_histogram, make_3d_motion_histograms

    # make motion histogram
    motion_histogram_dim = "2D"  # "2D" or "3D", for now only handle 2D case

    motion_histogram_list = []
    all_temporal_bin_edges = []  # TODO: fix naming

    bin_um = 2  # TODO: critial paraneter. easier to take no binning and gaus smooth?

    # TODO: own function
    for recording, peaks, peak_locations in zip(
        recordings_list,
        peaks_list,
        peak_locations_list,  # TODO: this is overwriting above variable names. Own function!
    ):  # TODO: do a lot of checks to make sure these bin sizes make sesnese
        # Do some checks on temporal and spatial bin edges that they are all the same?

        if motion_histogram_dim == "2D":
            motion_histogram = make_2d_motion_histogram(
                recording,
                peaks,
                peak_locations,
                weight_with_amplitude=False,
                direction="y",
                bin_s=recording.get_duration(segment_index=0),  # 1.0,
                bin_um=bin_um,
                hist_margin_um=50,
                spatial_bin_edges=None,
            )
        else:
            assert NotImplementedError  # TODO: might be old API pre-dredge
            motion_histogram = make_3d_motion_histograms(
                recording,
                peaks,
                peak_locations,
                direction="y",
                bin_duration_s=recording.get_duration(segment_index=0),  # 1.0,
                bin_um=bin_um,
                margin_um=50,
                num_amp_bins=20,
                log_transform=True,
                spatial_bin_edges=None,
            )
        motion_histogram_list.append(motion_histogram[0].squeeze())
        # store bin edges
        all_temporal_bin_edges.append(motion_histogram[1])
        spatial_bin_edges_um = motion_histogram[2]  # should be same across all recordings

    # Do some checks on temporal and spatial bin edges that they are all the same?
    # TODO: do some smoothing? Try some other methds (e.g. NMI, KL divergence)
    # Let's do a very basic optimisation to find the best midpoint, just
    # align everything to the first session. This isn't great because
    # introduces some bias. Maybe align to all sessions and then take some
    # average. Certainly cannot optimise brute force over the whole space
    # which is (2P-1)^N where P is length of motion histogram and N is number of recordings.
    # TODO: double-check what is done in kilosort-like / DREDGE
    # put histograms into X and do X^T X then mean(U), det or eigs of covar mat
    # can try iterative template. Not sure it will work so well taking the mean
    # over only a few histograms that could be wildy different.
    # Displacemene
    num_recordings = len(recordings_list)

    shifts = np.zeros(num_recordings)

    # TODO: not checked any of the below properly
    first_hist = motion_histogram_list[0] / motion_histogram_list[0].sum()
    #  first_hist -= np.mean(first_hist)  # TODO: pretty sure not necessary

    for i in range(1, num_recordings):

        hist = motion_histogram_list[i] / motion_histogram_list[i].sum()
        #   hist -= np.mean(hist)  # TODO: pretty sure not necessary
        conv = np.correlate(first_hist, hist, mode="full")

        if conv.size % 2 == 0:
            midpoint = conv.size / 2
        else:
            midpoint = (conv.size - 1) / 2  # TODO: carefully double check!

        # TODO: think will need to make this negative
        shifts[i] = (midpoint - np.argmax(conv)) * bin_um  # # TODO: the bin spacing is super important for resoltuion

    # half
    # TODO: need to figure out interpolation to the center point, weird;y
    # the below does not work
    # shifts[0] = (shifts[1] / 2)
    #  shifts[1] = (shifts[1] / 2) * -1
    #   print("SHIFTS", shifts)
    # TODO: handle only the 2D case for now
    # TODO: do multi-session optimisation

    # Handle drift
    interpolate_motion_kwargs = {}

    # TODO: add motion to motion if exists otherwise create InterpolateMotionRecording object!
    # Will need the y-axis bins for this
    all_recording_corrected = []
    all_motion_info = []
    for i, recording in enumerate(recordings_list):

        # TODO: direct copy, use 'get_window' from motion machinery
        if False:
            bin_centers = spatial_bin_edges[:-1] + bin_um / 2.0
            n = bin_centers.size
            non_rigid_windows = [np.ones(n, dtype="float64")]
            middle = (spatial_bin_edges[0] + spatial_bin_edges[-1]) / 2.0
            non_rigid_window_centers = np.array([middle])

        dim = 1  # ["x", "y", "z"].index(direction)
        contact_depths = recording.get_channel_locations()[:, dim]
        spatial_bin_centers = 0.5 * (spatial_bin_edges_um[1:] + spatial_bin_edges_um[:-1])

        _, window_centers = get_spatial_windows(
            contact_depths, spatial_bin_centers, rigid=True  # TODO: handle non-rigid case
        )
        #        win_shape=win_shape,  TODO: handle defaults better
        #       win_step_um=win_step_um,
        #      win_scale_um=win_scale_um,
        #     win_margin_um=win_margin_um,
        #    zero_threshold=1e-5,

        #    if shifts[i] == 0:
        ##       all_recording_corrected.append(recording)  # TODO
        #     continue
        temporal_bin_edges = all_temporal_bin_edges[i]
        temporal_bins = 0.5 * (temporal_bin_edges[1:] + temporal_bin_edges[:-1])

        motion_array = np.zeros((temporal_bins.size, window_centers.size))  # TODO: check this is the expected shape
        motion_array[:, :] = shifts[i]  # TODO: this is the rigid case!

        motion = Motion(
            [motion_array], [temporal_bins], window_centers, direction="y"
        )  # will be same for all except for shifts
        all_motion_info.append(motion)  # not certain on this

        if isinstance(recording, InterpolateMotionRecording):
            raise NotImplementedError
            recording_corrected = copy.deepcopy(recording)
            # TODO: add interpolation to the existing one.
            # Not if inter-session motion correction already exists, but further
            # up the preprocessing chain, it will NOT be added and interpolation
            # will occur twice. Throw a warning here!
        else:
            recording_corrected = InterpolateMotionRecording(recording, motion, **interpolate_motion_kwargs)
        all_recording_corrected.append(recording_corrected)

    displacement_info = {
        "all_motion_info": all_motion_info,
        "all_motion_histograms": motion_histogram_list,  # TODO: naming
        "all_shifts": shifts,
    }

    if keep_channels_constant:
        # TODO: use set
        import functools

        common_channels = functools.reduce(
            np.intersect1d, [recording.channel_ids for recording in all_recording_corrected]
        )

        all_recording_corrected = [recording.channel_slice(common_channels) for recording in all_recording_corrected]

    return all_recording_corrected, displacement_info  # TODO: output more stuff later e.g. the Motion object