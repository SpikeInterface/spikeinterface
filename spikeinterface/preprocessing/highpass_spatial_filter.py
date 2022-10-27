import numpy as np
import scipy.signal
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment  # TODO: fix
from spikeinterface.core.core_tools import define_function_from_class
import matplotlib.pyplot as plt
# Tests: all options (taper, padding, agc_defaults)
# TODO: check that 300 samples is good default (is good for all time lengths?)
# TODO: check Wn
# TODO: operating on a copy of traces so can overwrite in
# TODO: check n_channel_pad and n_channel_taper argument behaviour
# TODO: check agc on first trace

class HighpassSpatialFilter(BasePreprocessor):
    """
    """
    name = 'highpass_spatial_filter'

    def __init__(self, recording, collection=None, n_channel_pad=0, n_channel_taper=None, agc_options="default", butter_kwargs=None):
        BasePreprocessor.__init__(self, recording)

        sampling_interval = 1 / recording.get_sampling_frequency()

        if agc_options == "default":
            agc_options = {"window_length_s": self.get_default_agc_window_length(sampling_interval),
                           "sampling_interval": sampling_interval}

        if butter_kwargs is None:
            butter_kwargs = {'N': 3, 'Wn': 0.01, 'btype': 'highpass'}

        for parent_segment in recording._recording_segments:
            rec_segment = HighPassSpatialFilterSegment(parent_segment,
                                                       collection,
                                                       n_channel_pad,
                                                       n_channel_taper,
                                                       agc_options,
                                                       butter_kwargs)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(),
                            collection=collection,
                            n_channel_pad=n_channel_pad,
                            n_channel_taper=n_channel_taper,
                            agc_options=agc_options,
                            butter_kwargs=butter_kwargs)

    def get_default_agc_window_length(self, sampling_interval):
        return 300 * sampling_interval

class HighPassSpatialFilterSegment(BasePreprocessorSegment):
    """

    """
    def __init__(self,
                 parent_recording_segment,
                 collection,
                 n_channel_pad,
                 n_channel_taper,
                 agc_options,
                 butter_kwargs):

        self.parent_recording_segment = parent_recording_segment
        self.collection = collection
        self.n_channel_pad = n_channel_pad
        self.n_channel_taper = n_channel_taper
        self.agc_options = agc_options
        self.butter_kwargs = butter_kwargs

    def get_traces(self, start_frame, end_frame, channel_indices):

        traces = self.parent_recording_segment.get_traces(start_frame,
                                                          end_frame,
                                                          slice(None))

        traces = traces.copy()

        traces = kfilt(traces.T,  # TODO: change dims in function
                       self.collection,
                       self.n_channel_pad,
                       self.n_channel_taper,
                       self.agc_options,
                       self.butter_kwargs)

        return traces

# function for API
highpass_spatial_filter = define_function_from_class(source_class=HighpassSpatialFilter, name="highpass_spatial_filter")

# -----------------------------------------------------------------------------------------------
# IBL Functions
# -----------------------------------------------------------------------------------------------

def kfilt(traces, collection, n_channel_pad, n_channel_taper, agc_options, butter_kwargs):
    """
    Applies a butterworth filter on the 0-axis with tapering / padding

    :param traces: the input array to be filtered. dimension, the filtering is considering
              axis=0: spatial dimension, axis=1 temporal dimension. (ntraces, ns)
    :param collection:
    :param n_channel_pad: traces added to each side (mirrored)
    :param n_channel_taper: n traces for apodizatin on each side
    :param agc_options: window size for time domain automatic gain control (no agc otherwise)
    :param butter_kwargs: filtering parameters: defaults: {'N': 3, 'Wn': 0.01, 'btype': 'highpass'}
    :return:
    """
    if collection is not None:
        xout = np.zeros_like(traces)
        for c in np.unique(collection):
            sel = collection == c
            xout[sel, :] = kfilt(traces=traces[sel, :], n_channel_pad=0, n_channel_taper=None, collection=None,
                                 butter_kwargs=butter_kwargs)
        return xout
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
        print(traces)
    if n_channel_pad > 0:
        # pad the array with a mirrored version of itself and apply a cosine taper
        traces = np.r_[np.flipud(traces[:n_channel_pad]),
                       traces,
                       np.flipud(traces[-n_channel_pad:])]
    if n_channel_taper > 0:
        taper = fcn_cosine([0, n_channel_taper])(np.arange(num_channels_padded))  # taper up
        taper *= 1 - fcn_cosine([num_channels_padded - n_channel_taper,
                                 num_channels_padded])(np.arange(num_channels_padded))   # taper down
        traces = traces * taper[:, np.newaxis]

    sos = scipy.signal.butter(**butter_kwargs, output='sos')
    traces = scipy.signal.sosfiltfilt(sos, traces, axis=0)

    if n_channel_pad > 0:
        traces = traces[n_channel_pad:-n_channel_pad, :]

    return traces / gain


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
