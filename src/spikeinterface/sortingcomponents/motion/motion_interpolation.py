from __future__ import annotations

import numpy as np
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.preprocessing import get_spatial_interpolation_kernel
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.preprocessing.filter import fix_dtype


def correct_motion_on_peaks(peaks, peak_locations, motion, recording):
    """
    Given the output of estimate_motion(), apply inverse motion on peak locations.

    Parameters
    ----------
    peaks : np.array
        peaks vector
    peak_locations : np.array
        peaks location vector
    motion : Motion
        The motion object.
    recording : Recording
        The recording object. This is used to convert sample indices to times.

    Returns
    -------
    corrected_peak_locations: np.array
        Motion-corrected peak locations
    """
    if recording is None:
        raise ValueError("correct_motion_on_peaks need recording to be not None")

    corrected_peak_locations = peak_locations.copy()

    for segment_index in range(motion.num_segments):
        times_s = recording.sample_index_to_time(peaks["sample_index"], segment_index=segment_index)
        i0, i1 = np.searchsorted(peaks["segment_index"], [segment_index, segment_index + 1])

        spike_times = times_s[i0:i1]
        spike_locs = peak_locations[motion.direction][i0:i1]
        spike_displacement = motion.get_displacement_at_time_and_depth(
            spike_times, spike_locs, segment_index=segment_index
        )
        corrected_peak_locations[i0:i1][motion.direction] -= spike_displacement

    return corrected_peak_locations


def interpolate_motion_on_traces(
    traces,
    times,
    channel_locations,
    motion,
    segment_index=None,
    channel_inds=None,
    interpolation_time_bin_centers_s=None,
    spatial_interpolation_method="kriging",
    spatial_interpolation_kwargs={},
    dtype=None,
):
    """
    Apply inverse motion with spatial interpolation on traces.

    Traces can be full traces, but also waveforms snippets.

    Parameters
    ----------
    traces : np.array
        Trace snippet (num_samples, num_channels)
    times : np.array
        Sample times in seconds for the frames of the traces snippet
    channel_location: np.array 2d
        Channel location with shape (n, 2) or (n, 3)
    motion: Motion
        The motion object.
    segment_index: int or None
        The segment index.
    channel_inds: None or list
        If not None, interpolate only a subset of channels.
    interpolation_time_bin_centers_s : None or np.array
        Manually specify the time bins which the interpolation happens
        in for this segment. If None, these are the motion estimate's time bins.
    spatial_interpolation_method: "idw" | "kriging", default: "kriging"
        The spatial interpolation method used to interpolate the channel locations:
            * idw : Inverse Distance Weighing
            * kriging : kilosort2.5 like
    spatial_interpolation_kwargs:
        * specific option for the interpolation method

    Returns
    -------
    traces_corrected: np.array
        Motion-corrected trace snippet, (num_samples, num_channels)
    """
    # assert HAVE_NUMBA
    assert times.shape[0] == traces.shape[0]

    if dtype is None:
        dtype = traces.dtype
    if dtype.kind != "f":
        raise ValueError(f"Can't interpolate_motion with dtype {dtype}.")
    if traces.dtype != dtype:
        traces = traces.astype(dtype)

    if segment_index is None:
        if motion.num_segments == 1:
            segment_index = 0
        else:
            raise ValueError("Several segment need segment_index=")

    if channel_inds is None:
        traces_corrected = np.zeros(traces.shape, dtype=traces.dtype)
    else:
        channel_inds = np.asarray(channel_inds)
        traces_corrected = np.zeros((traces.shape[0], channel_inds.size), dtype=traces.dtype)

    total_num_chans = channel_locations.shape[0]

    # -- determine the blocks of frames that will land in the same interpolation time bin
    time_bins = interpolation_time_bin_centers_s
    if time_bins is None:
        time_bins = motion.temporal_bins_s[segment_index]
    bin_s = time_bins[1] - time_bins[0]
    bins_start = time_bins[0] - 0.5 * bin_s
    # nearest bin center for each frame?
    bin_inds = (times - bins_start) // bin_s
    bin_inds = bin_inds.astype(int)
    # the time bins may not cover the whole set of times in the recording,
    # so we need to clip these indices to the valid range
    np.clip(bin_inds, 0, time_bins.size, out=bin_inds)

    # -- what are the possibilities here anyway?
    bins_here = np.arange(bin_inds[0], bin_inds[-1] + 1)

    # inperpolation kernel will be the same per temporal bin
    interp_times = np.empty(total_num_chans)
    current_start_index = 0
    for bin_ind in bins_here:
        bin_time = time_bins[bin_ind]
        interp_times.fill(bin_time)
        channel_motions = motion.get_displacement_at_time_and_depth(
            interp_times,
            channel_locations[:, motion.dim],
            segment_index=segment_index,
        )
        channel_locations_moved = channel_locations.copy()
        channel_locations_moved[:, motion.dim] += channel_motions

        if channel_inds is not None:
            channel_locations_moved = channel_locations_moved[channel_inds]

        drift_kernel = get_spatial_interpolation_kernel(
            channel_locations,
            channel_locations_moved,
            dtype=dtype,
            method=spatial_interpolation_method,
            **spatial_interpolation_kwargs,
        )

        # keep this for DEBUG
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.matshow(drift_kernel)
        # ax.set_title(f"bin_ind {bin_ind} - {bin_time}s - {spatial_interpolation_method}")
        # plt.show()

        # quickly find the end of this bin, which is also the start of the next
        next_start_index = current_start_index + np.searchsorted(
            bin_inds[current_start_index:], bin_ind + 1, side="left"
        )
        in_bin = slice(current_start_index, next_start_index)

        # here we use a simple np.matmul even if dirft_kernel can be super sparse.
        # because the speed for a sparse matmul is not so good when we disable multi threaad (due multi processing
        # in ChunkRecordingExecutor)
        np.matmul(traces[in_bin], drift_kernel, out=traces_corrected[in_bin])
        current_start_index = next_start_index

    return traces_corrected


# if HAVE_NUMBA:
#     # @numba.jit(parallel=False)
#     @numba.jit(parallel=True)
#     def my_sparse_dot(data_in, data_out, sparse_chans, weights):
#         """
#         Experimental home made sparse dot.
#         Faster when use prange but with multiprocessing it is not a good idea.
#         Custum sparse dot
#         data_in: num_sample, num_chan_in
#         data_out: num_sample, num_chan_out
#         sparse_chans: num_chan_out, num_sparse
#         weights: num_chan_out, num_sparse
#         """
#         num_samples = data_in.shape[0]
#         num_chan_out = data_out.shape[1]
#         num_sparse = sparse_chans.shape[1]
#         # for sample_index in range(num_samples):
#         for sample_index in numba.prange(num_samples):
#             for out_chan in range(num_chan_out):
#                 v = 0
#                 for i in range(num_sparse):
#                     in_chan = sparse_chans[out_chan, i]
#                     v +=  weights[out_chan, i] * data_in[sample_index, in_chan]
#                 data_out[sample_index, out_chan] = v


def _get_closest_ind(array, values):
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(array)) | (
        np.fabs(values - array[np.maximum(idxs - 1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array) - 1)])
    )
    idxs[prev_idx_is_less] -= 1

    return idxs


class InterpolateMotionRecording(BasePreprocessor):
    """
    Interpolate the input recording's traces to correct for motion, according to the
    motion estimate object `motion`. The interpolation is carried out "lazily" / on the fly
    by applying a spatial interpolation on the original traces to estimate their values
    at the positions of the probe's channels after being shifted inversely to the motion.

    To get a Motion object, use `interpolate_motion()`.

    By default, each frame is spatially interpolated by the motion at the nearest motion
    estimation time bin -- in other words, the temporal resolution of the motion correction
    is the same as the motion estimation's. However, this behavior can be changed by setting
    `interpolation_time_bin_centers_s` or `interpolation_time_bin_size_s` below. In that case,
    the motion estimate will be interpolated to match the interpolation time bins.

    Parameters
    ----------
    recording: Recording
        The parent recording.
    motion: Motion
        The motion object
    spatial_interpolation_method: "kriging" | "idw" | "nearest", default: "kriging"
        The spatial interpolation method used to interpolate the channel locations.
        See `spikeinterface.preprocessing.get_spatial_interpolation_kernel()` for more details.
        Choice of the method:

            * "kriging" : the same one used in kilosort
            * "idw" : inverse  distance weighted
            * "nearest" : use neareast channel
    sigma_um: float, default: 20.0
        Used in the "kriging" formula
    p: int, default: 1
        Used in the "kriging" formula
    num_closest: int, default: 3
        Number of closest channels used by "idw" method for interpolation.
    border_mode: "remove_channels" | "force_extrapolate" | "force_zeros", default: "remove_channels"
        Control how channels are handled on border:
        * "remove_channels": remove channels on the border, the recording has less channels
        * "force_extrapolate": keep all channel and force extrapolation (can lead to strange signal)
        * "force_zeros": keep all channel but set zeros when outside (force_extrapolate=False)
    interpolation_time_bin_centers_s: np.array or list of np.array, optional
        Spatially interpolate each frame according to the displacement estimate at its closest
        bin center in this array. If not supplied, this is set to the motion estimate's time bin
        centers. If it's supplied, the motion estimate is interpolated to these bin centers.
        If you have a multi-segment recording, pass a list of these, one per segment.
    interpolation_time_bin_size_s: float, optional
        Similar to the previous argument: interpolation_time_bin_centers_s will be constructed
        by bins spaced by interpolation_time_bin_size_s. This is ignored if interpolation_time_bin_centers_s
        is supplied.
    dtype : str or np.dtype, optional
        Interpolation needs to convert to a floating dtype. If dtype is supplied, that will be used.
        If the input recording is already floating and dtype=None, then its dtype is used by default.
        If the input recording is integer, then float32 is used by default.

    Returns
    -------
    corrected_recording: InterpolateMotionRecording
        Recording after motion correction
    """

    name = "interpolate_motion"

    def __init__(
        self,
        recording,
        motion,
        border_mode="remove_channels",
        spatial_interpolation_method="kriging",
        sigma_um=20.0,
        p=1,
        num_closest=3,
        interpolation_time_bin_centers_s=None,
        interpolation_time_bin_size_s=None,
        dtype=None,
        **spatial_interpolation_kwargs,
    ):
        # assert recording.get_num_segments() == 1, "correct_motion() is only available for single-segment recordings"

        channel_locations = recording.get_channel_locations()
        assert channel_locations.ndim >= motion.dim, (
            f"'direction' {motion.direction} not available. "
            f"Channel locations have {channel_locations.ndim} dimensions."
        )
        spatial_interpolation_kwargs = dict(
            sigma_um=sigma_um, p=p, num_closest=num_closest, **spatial_interpolation_kwargs
        )
        if border_mode == "remove_channels":
            locs = channel_locations[:, motion.dim]
            l0, l1 = np.min(locs), np.max(locs)

            # check if channels stay inside the probe extents for all segments
            channel_inside = np.ones(locs.shape[0], dtype="bool")
            for segment_index in range(recording.get_num_segments()):
                # evaluate the positions of all channels over all time bins
                channel_displacements = motion.get_displacement_at_time_and_depth(
                    times_s=motion.temporal_bins_s[segment_index],
                    locations_um=locs,
                    grid=True,
                )
                channel_locations_moved = locs[:, None] + channel_displacements
                # check if these remain inside of the probe
                seg_inside = channel_locations_moved.clip(l0, l1) == channel_locations_moved
                seg_inside = seg_inside.all(axis=1)
                channel_inside &= seg_inside

            channel_inds = np.flatnonzero(channel_inside)
            channel_ids = recording.channel_ids[channel_inds]
            spatial_interpolation_kwargs["force_extrapolate"] = False
        elif border_mode == "force_extrapolate":
            channel_inds = None
            channel_ids = recording.channel_ids
            spatial_interpolation_kwargs["force_extrapolate"] = True
        elif border_mode == "force_zeros":
            channel_inds = None
            channel_ids = recording.channel_ids
            spatial_interpolation_kwargs["force_extrapolate"] = False
        else:
            raise ValueError("Wrong border_mode")

        if dtype is None:
            if recording.dtype.kind == "f":
                dtype = recording.dtype
            else:
                raise ValueError(f"Can't interpolate traces of recording with non-floating dtype={recording.dtype=}.")

        dtype_ = fix_dtype(recording, dtype)
        BasePreprocessor.__init__(self, recording, channel_ids=channel_ids, dtype=dtype_)

        if border_mode == "remove_channels":
            # change the wiring of the probe
            # TODO this is also done in ChannelSliceRecording, this should be done in a common place
            contact_vector = self.get_property("contact_vector")
            if contact_vector is not None:
                contact_vector["device_channel_indices"] = np.arange(len(channel_ids), dtype="int64")
                self.set_property("contact_vector", contact_vector)

        # handle manual interpolation_time_bin_centers_s
        # the case where interpolation_time_bin_size_s is set is handled per-segment below
        if interpolation_time_bin_centers_s is None:
            if interpolation_time_bin_size_s is None:
                interpolation_time_bin_centers_s = motion.temporal_bins_s

        for segment_index, parent_segment in enumerate(recording._recording_segments):
            # finish the per-segment part of the time bin logic
            if interpolation_time_bin_centers_s is None:
                # in this case, interpolation_time_bin_size_s is set.
                s_end = parent_segment.get_num_samples()
                t_start, t_end = parent_segment.sample_index_to_time(np.array([0, s_end]))
                halfbin = interpolation_time_bin_size_s / 2.0
                segment_interpolation_time_bins_s = np.arange(t_start + halfbin, t_end, interpolation_time_bin_size_s)
            else:
                segment_interpolation_time_bins_s = interpolation_time_bin_centers_s[segment_index]

            rec_segment = InterpolateMotionRecordingSegment(
                parent_segment,
                channel_locations,
                motion,
                spatial_interpolation_method,
                spatial_interpolation_kwargs,
                channel_inds,
                segment_index,
                segment_interpolation_time_bins_s,
                dtype=dtype_,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            motion=motion,
            border_mode=border_mode,
            spatial_interpolation_method=spatial_interpolation_method,
            sigma_um=sigma_um,
            p=p,
            num_closest=num_closest,
            interpolation_time_bin_centers_s=interpolation_time_bin_centers_s,
            dtype=dtype_.str,
        )
        self._kwargs.update(spatial_interpolation_kwargs)


class InterpolateMotionRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        channel_locations,
        motion,
        spatial_interpolation_method,
        spatial_interpolation_kwargs,
        channel_inds,
        segment_index,
        interpolation_time_bin_centers_s,
        dtype="float32",
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.channel_locations = channel_locations
        self.spatial_interpolation_method = spatial_interpolation_method
        self.spatial_interpolation_kwargs = spatial_interpolation_kwargs
        self.channel_inds = channel_inds
        self.segment_index = segment_index
        self.interpolation_time_bin_centers_s = interpolation_time_bin_centers_s
        self.dtype = dtype
        self.motion = motion

    def get_traces(self, start_frame, end_frame, channel_indices):
        if self.time_vector is not None:
            raise NotImplementedError("InterpolateMotionRecording does not yet support recordings with time_vectors.")

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        times = self.parent_recording_segment.sample_index_to_time(np.arange(start_frame, end_frame))
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices=slice(None))
        traces = traces.astype(self.dtype)
        traces = interpolate_motion_on_traces(
            traces,
            times,
            self.channel_locations,
            self.motion,
            segment_index=self.segment_index,
            channel_inds=self.channel_inds,
            spatial_interpolation_method=self.spatial_interpolation_method,
            spatial_interpolation_kwargs=self.spatial_interpolation_kwargs,
            interpolation_time_bin_centers_s=self.interpolation_time_bin_centers_s,
        )

        if channel_indices is not None:
            traces = traces[:, channel_indices]

        return traces


interpolate_motion = define_function_from_class(source_class=InterpolateMotionRecording, name="interpolate_motion")
