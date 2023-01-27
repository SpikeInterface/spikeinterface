import numpy as np
import scipy.signal
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class


class HighpassSpatialFilterRecording(BasePreprocessor):
    """
    Perform destriping with high-pass spatial filtering. Uses
    the kfilt() function of the International Brain Laboratory.

    Median average filtering, by removing the median of signal across
    channels, assumes noise is constant across all channels. However,
    noise have exhibit low-frequency changes across nearby channels.

    This function extends the cut-band from 0 to 0.01 Nyquist to
    remove these low-frequency changes during destriping.

    Parameters
    ----------
    recording : BaseRecording
        The parent recording
    n_channel_pad : int
        Number of channels to pad prior to filtering. 
        Channels are padded with mirroring. 
        If None, no padding is applied, by default 5
    n_channel_taper : int
        Number of channels to perform cosine tapering on
        prior to filtering. If None and n_channel_pad is set,
        n_channel_taper will be set to the number of padded channels.
        Otherwise, the passed value will be used, by default None
    direction : str
        The direction in which the spatial filter is applied, by default "y"
    agc_options: dict, str, or None
        Options for automatic gain control. By default, gain control
        is applied prior to filtering to improve filter performance.
        When the argument is "ibl", the function will use the IBL pipeline defaults (agc on).
        Setting to None will turn off gain control, by default None.
        To customize, pass a dictionary with the fields:
            * agc_options["window_length_s"] - window length in seconds
            * agc_options["sampling_interval"] - recording sampling interval
    butter_kwargs: dict
        Dictionary with fields "N" and "Wn" to be passed to
        scipy.signal.butter, by default N=3 and Wn=0.01
    """
    name = 'highpass_spatial_filter'

    def __init__(self, recording, n_channel_pad=None, n_channel_taper=5, direction="y",
                 agc_options=None, butter_kwargs=None):
        BasePreprocessor.__init__(self, recording)

        # Check single group
        channel_groups = recording.get_channel_groups()
        assert len(np.unique(channel_groups)) == 1, \
            ("The recording contains multiple groups! It is recommended to apply spatial filtering "
             "separately for each group. You can split by group with: "
             ">>> recording_split = recording.split_by(property='group')")
        # If location are not sorted, estimate forward and reverse sorting
        channel_locations = recording.get_channel_locations()
        dim = ["x", "y", "z"].index(direction)
        assert dim < channel_locations.ndim, f"Direction {direction} is wrong"
        locs_mono = channel_locations[:, dim]
        if np.array_equal(np.sort(locs_mono), locs_mono):
            order_f = None
            order_r = None
        else:
            # use stable sort (mergesort) to avoid randomness when non-unique values
            order_f = np.argsort(locs_mono, kind='mergesort')
            order_r = np.argsort(order_f)

        # Fix channel padding and tapering
        n_channels = recording.get_num_channels()
        n_channel_pad = 0 if n_channel_pad is None else int(n_channel_pad)
        assert n_channel_pad <= recording.get_num_channels(), \
            "'n_channel_pad' must be less than the number of channels in recording."
        n_channel_taper = n_channel_pad if n_channel_taper is None else int(n_channel_taper)
        assert n_channel_taper <= recording.get_num_channels(), \
            "'n_channel_taper' must be less than the number of channels in recording."

        # Fix Automatic Gain Control options
        if agc_options is not None:
            assert isinstance(agc_options, (str, dict)), \
                f"agc_options can be 'ibl', a dictionary, or None, not {type(agc_options)}"
            sampling_interval = 1 / recording.sampling_frequency
            if isinstance(agc_options, str):
                assert agc_options == "ibl", "agc_options can be 'ibl', a dictionary, or None"
                # default IBL value is 300 * sampling_interval
                default_window_length = 300 * sampling_interval 
                agc_options = {"window_length_s": default_window_length,
                               "sampling_interval": sampling_interval}
            elif isinstance(agc_options, dict):
                assert "window_length_s" in agc_options, \
                    "The agc_options dict must contain both the 'window_length_s' field"
                agc_options["sampling_interval"] = sampling_interval

        # Pre-compute spatial filtering parameters
        butter_kwargs_default = {'btype': 'highpass', 'N': 3, 'Wn': 0.01}
        if butter_kwargs is not None:
            assert all(k in ("N", "Wn") for k in butter_kwargs), \
                "butter_kwargs can only specify filter order 'N' and critical frequency 'Wn' (1 is Nyquist)"
            butter_kwargs_default.update(butter_kwargs)
        sos_filter = scipy.signal.butter(**butter_kwargs_default, output='sos')

        for parent_segment in recording._recording_segments:
            rec_segment = HighPassSpatialFilterSegment(parent_segment,
                                                       n_channel_pad,
                                                       n_channel_taper,
                                                       n_channels,
                                                       agc_options,
                                                       sos_filter,
                                                       order_f,
                                                       order_r)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(),
                            n_channel_pad=n_channel_pad,
                            n_channel_taper=n_channel_taper,
                            direction=direction,
                            agc_options=agc_options,
                            butter_kwargs=butter_kwargs)

class HighPassSpatialFilterSegment(BasePreprocessorSegment):

    def __init__(self,
                 parent_recording_segment,
                 n_channel_pad,
                 n_channel_taper,
                 n_channels,
                 agc_options,
                 sos_filter,
                 order_f,
                 order_r
                 ):
        self.parent_recording_segment = parent_recording_segment
        self.n_channel_pad = n_channel_pad
        if n_channel_taper > 0:
            num_channels_padded = n_channels + n_channel_pad * 2
            self.taper = fcn_cosine([0, n_channel_taper])(np.arange(num_channels_padded))  # taper up
            self.taper *= 1 - fcn_cosine([num_channels_padded - n_channel_taper, num_channels_padded])(np.arange(num_channels_padded))   # taper down
        else:
            self.taper = None
        self.agc_options = agc_options
        self.order_f = order_f
        self.order_r = order_r
        # get filter params
        self.sos_filter = sos_filter

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_indices = slice(None)
        traces = self.parent_recording_segment.get_traces(start_frame,
                                                          end_frame,
                                                          slice(None))
        if self.order_f is not None:
            traces = traces[:, self.order_f]
        else:
            traces = traces.copy()

        traces = kfilt(traces,
                       self.n_channel_pad,
                       self.taper,
                       self.agc_options,
                       self.sos_filter)

        if self.order_r is not None:
            traces = traces[:, self.order_r]

        return traces[:, channel_indices]

# function for API
highpass_spatial_filter = define_function_from_class(source_class=HighpassSpatialFilterRecording,
                                                     name="highpass_spatial_filter")

# -----------------------------------------------------------------------------------------------
# IBL KFilt Function
# -----------------------------------------------------------------------------------------------

def kfilt(traces, n_channel_pad, taper, agc_options, sos_filter):
    """
    Alternative to median filtering across channels, in which the cut-band is
    extended from 0 to the 0.01 Nyquist corner frequency using butterworth filter.
    This allows removal of contaminating stripes that are not constant across channels.

    Performs filtering on the 0 axis (across channels), with optional
    padding (mirrored) and tapering (cosine taper) prior to filtering.
    Applies a butterworth filter on the 0-axis with tapering / padding.

    Details of the high-pass spatial filter function (written by Olivier Winter)
    used in the IBL pipeline can be found at:

    International Brain Laboratory et al. (2022). Spike sorting pipeline for the
    International Brain Laboratory. https://www.internationalbrainlab.com/repro-ephys

    traces: (num_channels x num_samples) numpy array

    Parameters
    ----------
    traces : 2D np.array
        (num_channels x num_samples) numpy array with the signals
    n_channel_pad : int
        Number of channels to pad
    n_channel_taper : int
        Number of channels to taper
    agc_options : dict
        Automatic Gain Control options
    sos_filter : sos filter
        Butterworth filter in second-order sections format

    Returns
    -------
    traces_kfilt : 2D np.array
        The spatially filtered traces
    """
    num_channels = traces.shape[1]

    # lateral padding left and right
    num_channels_padded = num_channels + n_channel_pad * 2

    # apply agc and keep the gain in handy
    if not agc_options:
        gain = 1
    else:
        # pr-compute gains?
        traces, gain = agc(traces,
                           window_length=agc_options["window_length_s"],
                           sampling_interval=agc_options["sampling_interval"])

    if n_channel_pad > 0:
        # pad the array with a mirrored version of itself and apply a cosine taper
        traces = np.c_[np.fliplr(traces[:, :n_channel_pad]),
                       traces,
                       np.fliplr(traces[:, -n_channel_pad:])]

    if taper is not None:
        traces = traces * taper[np.newaxis, :]

    traces = scipy.signal.sosfiltfilt(sos_filter, traces, axis=1)

    if n_channel_pad > 0:
        traces = traces[:, n_channel_pad:-n_channel_pad]

    return traces * gain

# -----------------------------------------------------------------------------------------------
# IBL Helper Functions
# -----------------------------------------------------------------------------------------------

def agc(traces, window_length, sampling_interval, epsilon=1e-8):
    """
    Automatic gain control
    w_agc, gain = agc(w, window_length=.5, si=.002, epsilon=1e-8)
    such as w_agc * gain = w
    :param traces: seismic array (sample last dimension)
    :param window_length: window length (secs) (original default 0.5)
    :param si: sampling interval (secs) (original default 0.002)
    :param epsilon: whitening (useful mainly for synthetic data)
    :return: AGC data array, gain applied to data
    """
    num_samples_window = int(np.round(window_length / sampling_interval / 2) * 2 + 1)
    window = np.hanning(num_samples_window)
    window /= np.sum(window)

    gain = scipy.signal.fftconvolve(np.abs(traces), window[:, None], mode='same', axes=0)

    gain += (np.sum(gain, axis=0) * epsilon / traces.shape[0])[np.newaxis, :]

    dead_channels = np.sum(gain, axis=0) == 0

    traces[:, ~dead_channels] = traces[:, ~dead_channels] / gain[:, ~dead_channels]

    return traces, gain

def fcn_extrap(x, f, bounds):
    """
    Extrapolates a flat value before and after bounds
    x: array to be filtered
    f: function to be applied between bounds (cf. fcn_cosine below)
    bounds: 2 elements list or np.array
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
    :param bounds:
    :return: lambda function
    """
    def _cos(x):
        return (1 - np.cos((x - bounds[0]) / (bounds[1] - bounds[0]) * np.pi)) / 2
    func = lambda x: fcn_extrap(x, _cos, bounds)  # noqa
    return func
