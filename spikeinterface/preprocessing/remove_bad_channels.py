import warnings

import numpy as np

from spikeinterface.core.channelslice import ChannelSliceRecording
from spikeinterface.core.core_tools import define_function_from_class

from .basepreprocessor import BasePreprocessor

from ..core import get_random_data_chunks

import scipy.stats

class RemoveBadChannelsRecording(BasePreprocessor, ChannelSliceRecording):
    """
    Remove bad channels from the recording extractor given a thershold
    on standard deviation.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    bad_threshold: float
        If automatic is used, the threshold for the standard deviation over which channels are removed
    **random_chunk_kwargs

    Returns
    -------
    remove_bad_channels_recording: RemoveBadChannelsRecording
        The recording extractor without bad channels
    """
    name = 'remove_bad_channels'

    def __init__(self, recording, bad_threshold=5, **random_chunk_kwargs):
        random_data = get_random_data_chunks(recording, **random_chunk_kwargs)

        stds = np.std(random_data, axis=0)
        thresh = bad_threshold * np.median(stds)
        keep_inds, = np.nonzero(stds < thresh)

        parents_chan_ids = recording.get_channel_ids()
        channel_ids = parents_chan_ids[keep_inds]
        self._parent_channel_indices = recording.ids_to_indices(channel_ids)

        BasePreprocessor.__init__(self, recording)
        ChannelSliceRecording.__init__(self, recording, channel_ids=channel_ids)

        self._kwargs = dict(recording=recording.to_dict(), bad_threshold=bad_threshold)
        self._kwargs.update(random_chunk_kwargs)


# function for API
remove_bad_channels = define_function_from_class(source_class=RemoveBadChannelsRecording, name="remove_bad_channels")

# ------------------------------------------------------------------------------------------
# SpikeInterface Detect Bad Channels
# ------------------------------------------------------------------------------------------

def detect_bad_channels(recording,
                        similarity_threshold=(-0.5, 1),
                        psd_hf_threshold=None,
                        random_chunk_kwargs=None):
    """
    Note psd_hf_threshold units
    Edge cases:
    """
    # assert recording.get_num_channels() != 384, "bad channel detection for channels < 384 is not currently supported"

    random_chunk_kwargs, scale_for_testing = handle_random_chunk_kwargs(recording,
                                                                        random_chunk_kwargs)
    random_data = get_random_data_chunks(recording, **random_chunk_kwargs)

    channel_labels = np.zeros((recording.get_num_channels(),
                               recording.get_num_segments() * random_chunk_kwargs["num_chunks_per_segment"]))

    for i, random_chunk in enumerate(random_data):

        channel_labels[:, i], __ = detect_bad_channels_ibl(random_chunk.T,
                                                           recording.get_sampling_frequency(),
                                                           similarity_threshold,
                                                           psd_hf_threshold,
                                                           scale_for_testing)

    channel_flags, __ = scipy.stats.mode(channel_labels, axis=1, keepdims=False)

    bad_inds, = np.where(channel_flags != 0)
    bad_channel_ids = recording.get_channel_ids()[bad_inds]

    if bad_channel_ids.size > recording.get_num_channels() * 0.333:
        warnings.warn("Over 1/3 of channels are detected as bad. In the precense of a high"
                      "number of dead / noisy channels, bad channel detection may fail "
                      "(erroneously label good channels as dead).")

    return bad_inds, bad_channel_ids, channel_flags


def handle_random_chunk_kwargs(recording, user_random_chunk_kwargs):
    """
    Make default random chunk kwargs and overwrite with any user-specified.
    """
    chunk_size = int(0.3 * recording.get_sampling_frequency())
    random_chunk_kwargs = {"return_scaled": True,
                           "num_chunks_per_segment": 10,
                           "chunk_size": chunk_size,
                           "concatenated": False,
                           "seed": 0}

    if user_random_chunk_kwargs is not None:
        random_chunk_kwargs.update(user_random_chunk_kwargs)

    scale_for_testing = handle_test_case(random_chunk_kwargs)

    if random_chunk_kwargs["concatenated"]:
        raise AttributeError("Custom random_chunk_kwargs cannot included data concatenation")

    return random_chunk_kwargs, scale_for_testing

def handle_test_case(scale_for_testing):
    """"""
    if "scale_for_testing" in scale_for_testing:
        scale_for_testing.pop("scale_for_testing")
        scale_for_testing = True
    else:
        scale_for_testing = False

    return scale_for_testing

# ----------------------------------------------------------------------------------------------
# IBL Detect Bad Channels
# ----------------------------------------------------------------------------------------------

# units; uV ** 2 / Hz
# the LFP band data is obviously much stronger so auto-adjust the default threshold

def detect_bad_channels_ibl(raw, fs, similarity_threshold=(-0.5, 1), psd_hf_threshold=None, scale_for_testing=False):
    """
    Bad channels detection for Neuropixel probes
    Labels channels
     0: all clear
     1: dead low coherence / amplitude
     2: noisy
     3: outside of the brain
    :param raw: [nc, ns]
    :param fs: sampling frequency
    :param similarity_threshold:
    :param psd_hf_threshold:
    :return: labels (numpy vector [nc]), xfeats: dictionary of features [nc]
    """
    nc, _ = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset
    xcor = channels_similarity(raw)

    scale = 1e6 if scale_for_testing else 1
    fscale, psd = scipy.signal.welch(raw * scale, fs=fs)

    if psd_hf_threshold is None:
        psd_hf_threshold = 1.4 if fs < 5000 else 0.02

    sos_hp = scipy.signal.butter(**{'N': 3, 'Wn': 300 / fs * 2, 'btype': 'highpass'}, output='sos')
    hf = scipy.signal.sosfiltfilt(sos_hp, raw)
    xcorf = channels_similarity(hf)

    xfeats = ({
        'ind': np.arange(nc),
        'rms_raw': rms(raw),  # very similar to the rms avfter butterworth filter
        'xcor_hf': detrend(xcor, 11),
        'xcor_lf': xcorf - detrend(xcorf, 11) - 1,
        'psd_hf': np.mean(psd[:, fscale > (fs / 2 * 0.8)], axis=-1),  # 80% nyquists
    })

    # make recommendation
    ichannels = np.zeros(nc)
    idead = np.where(similarity_threshold[0] > xfeats['xcor_hf'])[0]
    inoisy = np.where(np.logical_or(xfeats['psd_hf'] > psd_hf_threshold, xfeats['xcor_hf'] > similarity_threshold[1]))[0]

    # the channels outside of the brains are the contiguous channels below the threshold on the trend coherency
    ioutside = np.where(xfeats['xcor_lf'] < -0.75)[0]
    if ioutside.size > 0 and ioutside[-1] == (nc - 1):
        a = np.cumsum(np.r_[0, np.diff(ioutside) - 1])
        ioutside = ioutside[a == np.max(a)]
        ichannels[ioutside] = 3

    ichannels[idead] = 1
    ichannels[inoisy] = 2

    return ichannels, xfeats

# ----------------------------------------------------------------------------------------------
# IBL Helpers
# ----------------------------------------------------------------------------------------------

def rms(x, axis=-1):
    """
    Root mean square of array along axis
    :param x: array on which to compute RMS
    :param axis: (optional, -1)
    :return: numpy array
    """
    return np.sqrt(np.mean(x ** 2, axis=axis))

def detrend(x, nmed):
    """
    Subtract the trend from a vector
    The trend is a median filtered version of the said vector with tapering
    :param x: input vector
    :param nmed: number of points of the median filter
    :return: np.array
    """
    ntap = int(np.ceil(nmed / 2))
    xf = np.r_[np.zeros(ntap) + x[0], x, np.zeros(ntap) + x[-1]]

    xf = scipy.signal.medfilt(xf, nmed)[ntap:-ntap]
    return x - xf

def channels_similarity(raw, nmed=0):
    """
    Computes the similarity based on zero-lag crosscorrelation of each channel with the median
    trace referencing
    :param raw: [nc, ns]
    :param nmed:
    :return:
    """
    def fxcor(x, y):
        return scipy.fft.irfft(scipy.fft.rfft(x) * np.conj(scipy.fft.rfft(y)), n=raw.shape[-1])

    def nxcor(x, ref):
        ref = ref - np.mean(ref)
        apeak = fxcor(ref, ref)[0]
        x = x - np.mean(x, axis=-1)[:, np.newaxis]  # remove DC component
        return fxcor(x, ref)[:, 0] / apeak

    ref = np.median(raw, axis=0)
    xcor = nxcor(raw, ref)

    if nmed > 0:
        xcor = detrend(xcor, nmed) + 1
    return xcor
