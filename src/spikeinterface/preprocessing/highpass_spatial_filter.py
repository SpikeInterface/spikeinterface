from __future__ import annotations

import numpy as np

from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment, BaseRecording
from spikeinterface.preprocessing.filter import fix_dtype
from spikeinterface.core import order_channels_by_depth, get_chunk_with_margin, get_noise_levels
from spikeinterface.core.core_tools import define_function_handling_dict_from_class


class HighpassSpatialFilterRecording(BasePreprocessor):
    """
    Perform destriping with high-pass spatial filtering. Uses
    the kfilt() function of the International Brain Laboratory.

    Median average filtering, by removing the median of signal across
    channels, assumes noise is constant across all channels. However,
    noise have exhibit low-frequency changes across nearby channels.

    Alternative to median filtering across channels, in which the cut-band is
    extended from 0 to the 0.01 Nyquist corner frequency using butterworth filter.
    This allows removal of contaminating stripes that are not constant across channels.

    Performs filtering on the 0 axis (across channels), with optional
    padding (mirrored) and tapering (cosine taper) prior to filtering.
    Applies a butterworth filter on the 0-axis with tapering / padding.

    Parameters
    ----------
    recording : BaseRecording
        The parent recording
    n_channel_pad : int, default: 60
        Number of channels to pad prior to filtering.
        Channels are padded with mirroring.
        If None, no padding is applied
    n_channel_taper : int, default: 0
        Number of channels to perform cosine tapering on
        prior to filtering. If None and n_channel_pad is set,
        n_channel_taper will be set to the number of padded channels.
        Otherwise, the passed value will be used
    direction : "x" | "y" | "z", default: "y"
        The direction in which the spatial filter is applied
    apply_agc : bool, default: True
        It True, Automatic Gain Control is applied
    agc_window_length_s : float, default: 0.1
        Window in seconds to compute Hanning window for AGC
    highpass_butter_order : int, default: 3
        Order of spatial butterworth filter
    highpass_butter_wn : float, default: 0.01
        Critical frequency (with respect to Nyquist) of spatial butterworth filter
    epsilon : float, default: 0.003
        Value multiplied to RMS values to avoid division by zero during AGC.
    random_slice_kwargs : dict | None, default: None
        If not None, dictionary of arguments to be passed to `get_noise_levels` when computing
        noise levels.
    dtype : dtype, default: None
        The dtype of the output traces. If None, the dtype is the same as the input traces
    rms_values : np.ndarray | None, default: None
        If not None, array of RMS values for each channel to be used during AGC. If None, RMS values are computed
        from the recording. This is used to cache pre-computed RMS values, which are only computed once at
        initialization.

    Returns
    -------
    highpass_recording : HighpassSpatialFilterRecording
        The recording with highpass spatial filtered traces

    References
    ----------
    Details of the high-pass spatial filter function (written by Olivier Winter)
    used in the IBL pipeline can be found at:
    International Brain Laboratory et al. (2022). Spike sorting pipeline for the International Brain Laboratory.
    https://www.internationalbrainlab.com/repro-ephys
    """

    def __init__(
        self,
        recording: BaseRecording,
        n_channel_pad=60,
        n_channel_taper=0,
        direction="y",
        apply_agc=True,
        agc_window_length_s=0.1,
        highpass_butter_order=3,
        highpass_butter_wn=0.01,
        epsilon=0.003,
        random_slice_kwargs=None,
        dtype=None,
        rms_values=None,
    ):
        BasePreprocessor.__init__(self, recording)

        import scipy.signal

        # Check single group
        channel_groups = recording.get_channel_groups()
        assert len(np.unique(channel_groups)) == 1, (
            "The recording contains multiple groups! It is recommended to apply spatial filtering "
            "separately for each group. You can split by group with: "
            ">>> recording_split = recording.split_by(property='group')"
        )
        # If location are not sorted, estimate forward and reverse sorting
        channel_locations = recording.get_channel_locations()
        dim = ["x", "y", "z"].index(direction)
        assert dim < channel_locations.ndim, f"Direction {direction} is wrong"
        locs_depth = channel_locations[:, dim]
        if np.array_equal(np.sort(locs_depth), locs_depth):
            order_f = None
            order_r = None
        else:
            # sort by x, y to avoid ambiguity
            order_f, order_r = order_channels_by_depth(recording=recording, dimensions=("x", "y"))

        # Fix channel padding and tapering
        n_channels = recording.get_num_channels()
        n_channel_pad = 0 if n_channel_pad is None else int(n_channel_pad)
        assert (
            n_channel_pad <= recording.get_num_channels()
        ), "'n_channel_pad' must be less than the number of channels in recording."
        n_channel_taper = n_channel_pad if n_channel_taper is None else int(n_channel_taper)
        assert (
            n_channel_taper <= recording.get_num_channels()
        ), "'n_channel_taper' must be less than the number of channels in recording."

        # Fix Automatic Gain Control options
        sampling_interval = 1 / recording.sampling_frequency
        if not apply_agc:
            agc_window_length_s = None

        # Compute or retrieve RMS values
        if rms_values is None:
            if "noise_level_rms_raw" in recording.get_property_keys():
                rms_values = recording.get_property("noise_level_rms_raw")
            else:
                random_slice_kwargs = {} if random_slice_kwargs is None else random_slice_kwargs
                rms_values = get_noise_levels(recording, method="rms", return_scaled=False, **random_slice_kwargs)

        # Pre-compute spatial filtering parameters
        butter_kwargs = dict(btype="highpass", N=highpass_butter_order, Wn=highpass_butter_wn)
        sos_filter = scipy.signal.butter(**butter_kwargs, output="sos")

        dtype = fix_dtype(recording, dtype)

        for parent_segment in recording._recording_segments:
            rec_segment = HighPassSpatialFilterSegment(
                parent_segment,
                n_channel_pad,
                n_channel_taper,
                n_channels,
                agc_window_length_s,
                sampling_interval,
                sos_filter,
                order_f,
                order_r,
                dtype=dtype,
                epsilon=epsilon,
                rms_values=rms_values,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            n_channel_pad=n_channel_pad,
            n_channel_taper=n_channel_taper,
            direction=direction,
            apply_agc=apply_agc,
            agc_window_length_s=agc_window_length_s,
            highpass_butter_order=highpass_butter_order,
            highpass_butter_wn=highpass_butter_wn,
            rms_values=rms_values,
        )


class HighPassSpatialFilterSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        n_channel_pad,
        n_channel_taper,
        n_channels,
        agc_window_length_s,
        sampling_interval,
        sos_filter,
        order_f,
        order_r,
        dtype,
        epsilon,
        rms_values,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.parent_recording_segment = parent_recording_segment
        self.n_channel_pad = n_channel_pad
        if n_channel_taper > 0:
            num_channels_padded = n_channels + n_channel_pad * 2
            self.taper = fcn_cosine([0, n_channel_taper])(np.arange(num_channels_padded))  # taper up
            self.taper *= 1 - fcn_cosine([num_channels_padded - n_channel_taper, num_channels_padded])(
                np.arange(num_channels_padded)
            )  # taper down
        else:
            self.taper = None
        if agc_window_length_s is not None:
            num_samples_window = int(np.round(agc_window_length_s / sampling_interval / 2) * 2 + 1)
            window = np.hanning(num_samples_window)
            window /= np.sum(window)
            self.window = window
        else:
            self.window = None
        self.order_f = order_f
        self.order_r = order_r
        # get filter params
        self.sos_filter = sos_filter
        self.dtype = dtype
        self.epsilon_values_for_agc = epsilon * np.array(rms_values)

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_indices = slice(None)
        if self.window is not None:
            margin = len(self.window) // 2
        else:
            margin = 0
        traces, left_margin, right_margin = get_chunk_with_margin(
            self.parent_recording_segment,
            start_frame=start_frame,
            end_frame=end_frame,
            channel_indices=slice(None),
            margin=margin,
        )
        # apply sorting by depth
        if self.order_f is not None:
            traces = traces[:, self.order_f]
        else:
            traces = traces.copy()

        # apply AGC and keep the gains
        traces = traces.astype(np.float32)
        if self.window is not None:
            traces, agc_gains = agc(traces, window=self.window, epsilons=self.epsilon_values_for_agc)
        else:
            agc_gains = None
        # pad the array with a mirrored version of itself and apply a cosine taper
        if self.n_channel_pad > 0:
            traces = np.c_[
                np.fliplr(traces[:, : self.n_channel_pad]), traces, np.fliplr(traces[:, -self.n_channel_pad :])
            ]
        # apply tapering
        if self.taper is not None:
            traces = traces * self.taper[np.newaxis, :]

        # apply actual HP filter
        import scipy.signal

        traces = scipy.signal.sosfiltfilt(self.sos_filter, traces, axis=1)

        # remove padding
        if self.n_channel_pad > 0:
            traces = traces[:, self.n_channel_pad : -self.n_channel_pad]

        # remove AGC gains
        if agc_gains is not None:
            traces = traces * agc_gains

        # reverse sorting by depth
        if self.order_r is not None:
            traces = traces[:, self.order_r]
        # remove margin and slice channels
        if right_margin > 0:
            traces = traces[left_margin:-right_margin, channel_indices]
        else:
            traces = traces[left_margin:, channel_indices]
        return traces.astype(self.dtype, copy=False)


# function for API
highpass_spatial_filter = define_function_handling_dict_from_class(
    source_class=HighpassSpatialFilterRecording, name="highpass_spatial_filter"
)


# -----------------------------------------------------------------------------------------------
# IBL Helper Functions
# -----------------------------------------------------------------------------------------------


def agc(traces, window, epsilons):
    """
    Automatic gain control
    w_agc, gain = agc(w, window_length=.5, si=.002, epsilon=1e-8)
    such as w_agc * gain = w

    Parameters
    ----------
    traces : np.ndarray
        Input traces
    window : np.ndarray
        Window to use for AGC (1D array)
    epsilons : np.ndarray[float]
        Epsilon values for each channel to avoid division by zero

    Returns
    -------
    agc_traces : np.ndarray
        AGC applied traces
    gain : np.ndarray
        Gain applied to the traces
    """
    import scipy.signal

    gain = scipy.signal.fftconvolve(np.abs(traces), window[:, None], mode="same", axes=0)

    dead_channels = np.sum(gain, axis=0) == 0

    traces[:, ~dead_channels] = traces[:, ~dead_channels] / np.maximum(epsilons, gain[:, ~dead_channels])

    return traces, gain


def fcn_extrap(x, f, bounds):
    """
    Extrapolates a flat value before and after bounds

    Parameters
    ----------
    x : np.ndarray
        Input array
    f : function
        Function to be applied between bounds
    bounds : list or np.ndarray
        2 elements list or array defining the bounds

    Returns
    -------
    y : np.ndarray
        Output array
    """
    y = f(x)
    y[x < bounds[0]] = f(bounds[0])
    y[x > bounds[1]] = f(bounds[1])
    return y


def fcn_cosine(bounds):
    """
    Returns a soft thresholding function with a cosine taper:
    values <= bounds[0]: values
    values < bounds[0] < bounds[1] : cosine taper
    values < bounds[1]: bounds[1]

    Parameters
    ----------
    bounds : list or np.ndarray
        2 elements list or array defining the bounds

    Returns
    -------
    func : function
        Lambda function implementing the soft thresholding with cosine taper
    """

    def _cos(x):
        return (1 - np.cos((x - bounds[0]) / (bounds[1] - bounds[0]) * np.pi)) / 2

    func = lambda x: fcn_extrap(x, _cos, bounds)  # noqa
    return func
