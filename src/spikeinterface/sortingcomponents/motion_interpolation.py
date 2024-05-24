from __future__ import annotations

import numpy as np
import scipy.interpolate
from tqdm import tqdm

import scipy.spatial

from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.preprocessing import get_spatial_interpolation_kernel


# try:
#     import numba
#     HAVE_NUMBA = True
# except ImportError:
#     HAVE_NUMBA = False


def correct_motion_on_peaks(
    peaks,
    peak_locations,
    sampling_frequency,
    motion,
    temporal_bins,
    spatial_bins,
    direction="y",
):
    """
    Given the output of estimate_motion(), apply inverse motion on peak locations.

    Parameters
    ----------
    peaks: np.array
        peaks vector
    peak_locations: np.array
        peaks location vector
    sampling_frequency: np.array
        sampling_frequency of the recording
    motion: np.array 2D
        motion.shape[0] equal temporal_bins.shape[0]
        motion.shape[1] equal 1 when "rigid" motion equal temporal_bins.shape[0] when "non-rigid"
    temporal_bins: np.array
        Temporal bins in second.
    spatial_bins: np.array
        Bins for non-rigid motion. If spatial_bins.sahpe[0] == 1 then rigid motion is used.

    Returns
    -------
    corrected_peak_locations: np.array
        Motion-corrected peak locations
    """
    corrected_peak_locations = peak_locations.copy()

    spike_times = peaks["sample_index"] / sampling_frequency
    if spatial_bins.shape[0] == 1:
        # rigid motion interpolation 1D
        f = scipy.interpolate.interp1d(temporal_bins, motion[:, 0], bounds_error=False, fill_value="extrapolate")
        shift = f(spike_times)
        corrected_peak_locations[direction] -= shift
    else:
        # non rigid motion = interpolation 2D
        f = scipy.interpolate.RegularGridInterpolator(
            (temporal_bins, spatial_bins), motion, method="linear", bounds_error=False, fill_value=None
        )
        shift = f(np.c_[spike_times, peak_locations[direction]])
        corrected_peak_locations[direction] -= shift

    return corrected_peak_locations


def interpolate_motion_on_traces(
    traces,
    times,
    channel_locations,
    motion,
    temporal_bins,
    spatial_bins,
    direction=1,
    channel_inds=None,
    spatial_interpolation_method="kriging",
    spatial_interpolation_kwargs={},
):
    """
    Apply inverse motion with spatial interpolation on traces.

    Traces can be full traces, but also waveforms snippets.

    Parameters
    ----------
    traces : np.array
        Trace snippet (num_samples, num_channels)
    channel_location: np.array 2d
        Channel location with shape (n, 2) or (n, 3)
    motion: np.array 2D
        motion.shape[0] equal temporal_bins.shape[0]
        motion.shape[1] equal 1 when "rigid" motion
                        equal temporal_bins.shape[0] when "none rigid"
    temporal_bins: np.array
        Temporal bins in second.
    spatial_bins: None or np.array
        Bins for non-rigid motion. If None, rigid motion is used
    direction: int in (0, 1, 2)
        Dimension of shift in channel_locations.
    channel_inds: None or list
        If not None, interpolate only a subset of channels.
    spatial_interpolation_method: "idw" | "kriging", default: "kriging"
        The spatial interpolation method used to interpolate the channel locations:
            * idw : Inverse Distance Weighing
            * kriging : kilosort2.5 like
    spatial_interpolation_kwargs:
        * specific option for the interpolation method

    Returns
    -------
    channel_motions: np.array
        Shift over time by channel
        Shape (times.shape[0], channel_location.shape[0])
    """
    # assert HAVE_NUMBA
    assert times.shape[0] == traces.shape[0]

    if channel_inds is None:
        traces_corrected = np.zeros(traces.shape, dtype=traces.dtype)
    else:
        channel_inds = np.asarray(channel_inds)
        traces_corrected = np.zeros((traces.shape[0], channel_inds.size), dtype=traces.dtype)

    # regroup times by closet temporal_bins
    bin_inds = _get_closest_ind(temporal_bins, times)

    # inperpolation kernel will be the same per temporal bin
    for bin_ind in np.unique(bin_inds):
        # Step 1 : channel motion
        if spatial_bins.shape[0] == 1:
            # rigid motion : same motion for all channels
            channel_motions = motion[bin_ind, 0]
        else:
            # non rigid : interpolation channel motion for this temporal bin
            f = scipy.interpolate.interp1d(
                spatial_bins, motion[bin_ind, :], kind="linear", axis=0, bounds_error=False, fill_value="extrapolate"
            )
            locs = channel_locations[:, direction]
            channel_motions = f(locs)
        channel_locations_moved = channel_locations.copy()
        channel_locations_moved[:, direction] += channel_motions
        # channel_locations_moved[:, direction] -= channel_motions

        if channel_inds is not None:
            channel_locations_moved = channel_locations_moved[channel_inds]

        drift_kernel = get_spatial_interpolation_kernel(
            channel_locations,
            channel_locations_moved,
            dtype="float32",
            method=spatial_interpolation_method,
            **spatial_interpolation_kwargs,
        )

        i0 = np.searchsorted(bin_inds, bin_ind, side="left")
        i1 = np.searchsorted(bin_inds, bin_ind, side="right")

        # here we use a simple np.matmul even if dirft_kernel can be super sparse.
        # because the speed for a sparse matmul is not so good when we disable multi threaad (due multi processing
        # in ChunkRecordingExecutor)
        traces_corrected[i0:i1] = traces[i0:i1] @ drift_kernel

    return traces_corrected


# if HAVE_NUMBA:
#     # @numba.jit(parallel=False)
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
#         # for sample_index in range(num_samples):
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
    Recording that corrects motion on-the-fly given a motion vector estimation (rigid or non-rigid).
    This internally applies a spatial interpolation on the original traces after reversing the motion.
    `estimate_motion()` must be called before this to estimate the motion vector.

    Parameters
    ----------
    recording: Recording
        The parent recording.
    motion: np.array 2D
        The motion signal obtained with `estimate_motion()`
        motion.shape[0] must correspond to temporal_bins.shape[0]
        motion.shape[1] is 1 when "rigid" motion and spatial_bins.shape[0] when "non-rigid"
    temporal_bins: np.array
        Temporal bins in second.
    spatial_bins: None or np.array
        Bins for non-rigid motion. If None, rigid motion is used
    direction: 0 | 1 | 2, default: 1
        Dimension along which channel_locations are shifted (0 - x, 1 - y, 2 - z)
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

    Returns
    -------
    corrected_recording: InterpolateMotionRecording
        Recording after motion correction
    """

    name = "correct_motion"

    def __init__(
        self,
        recording,
        motion,
        temporal_bins,
        spatial_bins,
        direction=1,
        border_mode="remove_channels",
        spatial_interpolation_method="kriging",
        sigma_um=20.0,
        p=1,
        num_closest=3,
    ):
        assert recording.get_num_segments() == 1, "correct_motion() is only available for single-segment recordings"

        # force as arrays
        temporal_bins = np.asarray(temporal_bins)
        motion = np.asarray(motion)
        spatial_bins = np.asarray(spatial_bins)

        channel_locations = recording.get_channel_locations()
        assert channel_locations.ndim >= direction, (
            f"'direction' {direction} not available. " f"Channel locations have {channel_locations.ndim} dimensions."
        )
        spatial_interpolation_kwargs = dict(sigma_um=sigma_um, p=p, num_closest=num_closest)
        if border_mode == "remove_channels":
            locs = channel_locations[:, direction]
            l0, l1 = np.min(channel_locations[:, direction]), np.max(channel_locations[:, direction])

            # compute max and min motion (with interpolation)
            # and check if channels are inside
            channel_inside = np.ones(locs.shape[0], dtype="bool")
            for operator in (np.max, np.min):
                if spatial_bins.shape[0] == 1:
                    best_motions = operator(motion[:, 0])
                else:
                    # non rigid : interpolation channel motion for this temporal bin
                    f = scipy.interpolate.interp1d(
                        spatial_bins,
                        operator(motion[:, :], axis=0),
                        kind="linear",
                        axis=0,
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    best_motions = f(locs)
                channel_inside &= ((locs + best_motions) >= l0) & ((locs + best_motions) <= l1)

            (channel_inds,) = np.nonzero(channel_inside)
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

        BasePreprocessor.__init__(self, recording, channel_ids=channel_ids)

        if border_mode == "remove_channels":
            # change the wiring of the probe
            # TODO this is also done in ChannelSliceRecording, this should be done in a common place
            contact_vector = self.get_property("contact_vector")
            if contact_vector is not None:
                contact_vector["device_channel_indices"] = np.arange(len(channel_ids), dtype="int64")
                self.set_property("contact_vector", contact_vector)

        for parent_segment in recording._recording_segments:
            rec_segment = InterpolateMotionRecordingSegment(
                parent_segment,
                channel_locations,
                motion,
                temporal_bins,
                spatial_bins,
                direction,
                spatial_interpolation_method,
                spatial_interpolation_kwargs,
                channel_inds,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            motion=motion,
            temporal_bins=temporal_bins,
            spatial_bins=spatial_bins,
            direction=direction,
            border_mode=border_mode,
            spatial_interpolation_method=spatial_interpolation_method,
            sigma_um=sigma_um,
            p=p,
            num_closest=num_closest,
        )


class InterpolateMotionRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        channel_locations,
        motion,
        temporal_bins,
        spatial_bins,
        direction,
        spatial_interpolation_method,
        spatial_interpolation_kwargs,
        channel_inds,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.channel_locations = channel_locations
        self.motion = motion
        self.temporal_bins = temporal_bins
        self.spatial_bins = spatial_bins
        self.direction = direction
        self.spatial_interpolation_method = spatial_interpolation_method
        self.spatial_interpolation_kwargs = spatial_interpolation_kwargs
        self.channel_inds = channel_inds

    def get_traces(self, start_frame, end_frame, channel_indices):
        if self.time_vector is not None:
            raise NotImplementedError(
                "time_vector for InterpolateMotionRecording do not work because temporal_bins start from 0"
            )
            # times = np.asarray(self.time_vector[start_frame:end_frame])

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        times = np.arange(end_frame - start_frame, dtype="float64")
        times /= self.sampling_frequency
        t0 = start_frame / self.sampling_frequency
        # if self.t_start is not None:
        #     t0 = t0 + self.t_start
        times += t0

        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices=slice(None))

        trace2 = interpolate_motion_on_traces(
            traces,
            times,
            self.channel_locations,
            self.motion,
            self.temporal_bins,
            self.spatial_bins,
            direction=self.direction,
            channel_inds=self.channel_inds,
            spatial_interpolation_method=self.spatial_interpolation_method,
            spatial_interpolation_kwargs=self.spatial_interpolation_kwargs,
        )

        if channel_indices is not None:
            trace2 = trace2[:, channel_indices]

        return trace2


interpolate_motion = define_function_from_class(source_class=InterpolateMotionRecording, name="correct_motion")
