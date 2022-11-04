import numpy as np
import scipy.signal
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class


class HighpassSpatialFilter(BasePreprocessor):
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

    n_channel_pad: Number of channels to pad prior to filtering. Channels
                   are padded with mirroring.

    n_channel_taper: Number of channels to perform cosine tapering on
                     prior to filtering. If None and n_channel_pad is set,
                     n_channel_taper will be set to the number of
                     padded channels. Otherwise, the passed value will be used.

    agc_options: Options for automatic gain control. By default, gain control
                 is applied prior to filtering to improve filter performance.
                 "default" will use the IBL pipeline defaults (agc on). Setting to
                 None will turn off gain control. To customise, pass a
                 dictionary with the fields:
                    agc_options["window_length_s"] - window length in seconds
                    agc_options["sampling_interval"] - recording sampling interval

    butter_kwargs: Dictionary with fields "N", "Wn", "btype" to be passed to
                   scipy.signal.butter
    """
    name = 'highpass_spatial_filter'

    def __init__(self, recording, n_channel_pad=None, n_channel_taper=None, agc_options="default", butter_kwargs=None):
        BasePreprocessor.__init__(self, recording)

        n_channel_pad, agc_options, butter_kwargs = self.handle_args(recording,
                                                                     n_channel_pad,
                                                                     agc_options,
                                                                     butter_kwargs)

        for parent_segment in recording._recording_segments:
            rec_segment = HighPassSpatialFilterSegment(parent_segment,
                                                       n_channel_pad,
                                                       n_channel_taper,
                                                       agc_options,
                                                       butter_kwargs)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(),
                            n_channel_pad=n_channel_pad,
                            n_channel_taper=n_channel_taper,
                            agc_options=agc_options,
                            butter_kwargs=butter_kwargs)

    def handle_args(self,
                    recording,
                    n_channel_pad,
                    agc_options,
                    butter_kwargs):
        """
        Make arguments well-defined before passing to kfilt.

        Use "default" argument for agc_options to make clear
        that they are on, then None / False / 0 are all clearly off.

        Default butter_kwargs are based on the IBL white paper .
        """
        if n_channel_pad in [None, False]:
            n_channel_pad = 0

        if agc_options == "default":
            sampling_interval = 1 / recording.get_sampling_frequency()
            agc_options = {"window_length_s": self.get_default_agc_window_length(sampling_interval),
                           "sampling_interval": sampling_interval}

        elif agc_options in [None, False, 0]:
            agc_options = None

        if butter_kwargs is None:
            butter_kwargs = {'N': 3, 'Wn': 0.01, 'btype': 'highpass'}

        return n_channel_pad, agc_options, butter_kwargs

    def get_default_agc_window_length(self, sampling_interval):
        """
        300 samples default based on the IBL implementation
        """
        return 300 * sampling_interval


class HighPassSpatialFilterSegment(BasePreprocessorSegment):

    def __init__(self,
                 parent_recording_segment,
                 n_channel_pad,
                 n_channel_taper,
                 agc_options,
                 butter_kwargs,
                 ):

        self.parent_recording_segment = parent_recording_segment
        self.n_channel_pad = n_channel_pad
        self.n_channel_taper = n_channel_taper
        self.agc_options = agc_options
        self.butter_kwargs = butter_kwargs

    def get_traces(self, start_frame, end_frame, channel_indices):

        traces = self.parent_recording_segment.get_traces(start_frame,
                                                          end_frame,
                                                          channel_indices)

        traces = traces.copy()

        traces = kfilt(traces.T,
                       self.n_channel_pad,
                       self.n_channel_taper,
                       self.agc_options,
                       self.butter_kwargs).T

        return traces

# function for API
highpass_spatial_filter = define_function_from_class(source_class=HighpassSpatialFilter, name="highpass_spatial_filter")

# -----------------------------------------------------------------------------------------------
# IBL KFilt Function
# -----------------------------------------------------------------------------------------------

def kfilt(traces, n_channel_pad, n_channel_taper, agc_options, butter_kwargs):
    """
    Alternative to median filtering across channels, in which the cut-band is
    extended from 0 to the 0.01 Nyquist corner frequency using butterworth filter.
    This allows removal of  contaminating stripes that are not constant across channels.

    Performs filtering on the 0 axis (across channels), with optional
    padding (mirrored) and tapering (cosine taper) prior to filtering.
    Applies a butterworth filter on the 0-axis with tapering / padding

    Details of the high-pass spatial filter function (Olivier Winter)
    used in the IBL pipeline can be found at:

    International Brain Laboratory et al. (2022). Spike sorting pipeline for the
    International Brain Laboratory. https://www.internationalbrainlab.com/repro-ephys

    traces: (num_channels x num_samples) numpy array
    """
    num_channels = traces.shape[0]

    # lateral padding left and right
    n_channel_pad = int(n_channel_pad)
    n_channel_taper = n_channel_pad if n_channel_taper is None else n_channel_taper
    num_channels_padded = num_channels + n_channel_pad * 2

    # apply agc and keep the gain in handy
    if not agc_options:
        gain = 1
    else:
        traces, gain = agc(traces,
                           window_length=agc_options["window_length_s"],
                           sampling_interval=agc_options["sampling_interval"])

    if n_channel_pad > 0:
        # pad the array with a mirrored version of itself and apply a cosine taper
        traces = np.r_[np.flipud(traces[:n_channel_pad]),
                       traces,
                       np.flipud(traces[-n_channel_pad:])]

    if n_channel_taper > 0:
        taper = fcn_cosine([0, n_channel_taper])(np.arange(num_channels_padded))  # taper up
        taper *= 1 - fcn_cosine([num_channels_padded - n_channel_taper, num_channels_padded])(np.arange(num_channels_padded))   # taper down
        traces = traces * taper[:, np.newaxis]

    sos = scipy.signal.butter(**butter_kwargs, output='sos')
    traces = scipy.signal.sosfiltfilt(sos, traces, axis=0)

    if n_channel_pad > 0:
        traces = traces[n_channel_pad:-n_channel_pad, :]

    return traces / gain

# -----------------------------------------------------------------------------------------------
# IBL Helper Functions
# -----------------------------------------------------------------------------------------------


def agc(traces, window_length, sampling_interval, epsilon=1e-8):
    """
    Automatic gain control
    w_agc, gain = agc(w, window_length=.5, si=.002, epsilon=1e-8)
    such as w_agc / gain = w
    :param traces: seismic array (sample last dimension)
    :param window_length: window length (secs) (original default 0.5)
    :param si: sampling interval (secs) (original default 0.002)
    :param epsilon: whitening (useful mainly for synthetic data)
    :return: AGC data array, gain applied to data
    """
    num_samples_window = int(np.round(window_length / sampling_interval / 2) * 2 + 1)
    window = np.hanning(num_samples_window)
    window /= np.sum(window)
    gain = convolve(np.abs(traces), window, mode='same')
    gain += (np.sum(gain, axis=1) * epsilon / traces.shape[-1])[:, np.newaxis]
    gain = 1 / gain

    return traces * gain, gain


def convolve(x, w, mode='full'):
    """
    Frequency domain convolution along the last dimension (2d arrays)
    Will broadcast if a matrix is convolved with a vector
    :param x:
    :param w:
    :param mode:
    :return: convolution
    """
    nsx = x.shape[-1]
    nsw = w.shape[-1]
    ns = ns_optim_fft(nsx + nsw)
    x_ = np.concatenate((x, np.zeros([*x.shape[:-1], ns - nsx], dtype=x.dtype)), axis=-1)
    w_ = np.concatenate((w, np.zeros([*w.shape[:-1], ns - nsw], dtype=w.dtype)), axis=-1)
    xw = np.real(np.fft.irfft(np.fft.rfft(x_, axis=-1) * np.fft.rfft(w_, axis=-1), axis=-1))
    xw = xw[..., :(nsx + nsw)]  # remove 0 padding
    if mode == 'full':
        return xw
    elif mode == 'same':
        first = int(np.floor(nsw / 2)) - ((nsw + 1) % 2)
        last = int(np.ceil(nsw / 2)) + ((nsw + 1) % 2)
        return xw[..., first:-last]


def ns_optim_fft(ns):
    """
    Gets the next higher combination of factors of 2 and 3 than ns to compute efficient ffts
    :param ns:
    :return: nsoptim
    """
    p2, p3 = np.meshgrid(2 ** np.arange(25), 3 ** np.arange(15))
    sz = np.unique((p2 * p3).flatten())
    return sz[np.searchsorted(sz, ns)]


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
