from __future__ import annotations
import warnings

import numpy as np
from typing import Literal

from .filter import highpass_filter
from ..core import get_random_data_chunks, order_channels_by_depth, BaseRecording


def detect_bad_channels(
    recording: BaseRecording,
    method: str = "coherence+psd",
    std_mad_threshold: float = 5,
    psd_hf_threshold: float = 0.02,
    dead_channel_threshold: float = -0.5,
    noisy_channel_threshold: float = 1.0,
    outside_channel_threshold: float = -0.75,
    outside_channels_location: Literal["top", "bottom", "both"] = "top",
    n_neighbors: int = 11,
    nyquist_threshold: float = 0.8,
    direction: Literal["x", "y", "z"] = "y",
    chunk_duration_s: float = 0.3,
    num_random_chunks: int = 100,
    welch_window_ms: float = 10.0,
    highpass_filter_cutoff: float = 300,
    neighborhood_r2_threshold: float = 0.9,
    neighborhood_r2_radius_um: float = 30.0,
    seed: int | None = None,
):
    """
    Perform bad channel detection.
    The recording is assumed to be filtered. If not, a highpass filter is applied on the fly.

    Different methods are implemented:

    * std : threhshold on channel standard deviations
        If the standard deviation of a channel is greater than `std_mad_threshold` times the median of all
        channels standard deviations, the channel is flagged as noisy
    * mad : same as std, but using median absolute deviations instead
    * coeherence+psd : method developed by the International Brain Laboratory that detects bad channels of three types:
        * Dead channels are those with low similarity to the surrounding channels (n=`n_neighbors` median)
        * Noise channels are those with power at >80% Nyquist above the psd_hf_threshold (default 0.02 uV^2 / Hz)
          and a high coherence with "far away" channels"
        * Out of brain channels are contigious regions of channels dissimilar to the median of all channels
          at the top end of the probe (i.e. large channel number)
    * neighborhood_r2
        A method tuned for LFP use-cases, where channels should be highly correlated with their spatial
        neighbors. This method estimates the correlation of each channel with the median of its spatial
        neighbors, and considers channels bad when this correlation is too small.

    Parameters
    ----------
    recording : BaseRecording
        The recording for which bad channels are detected
    method : "coeherence+psd" | "std" | "mad" | "neighborhood_r2", default: "coeherence+psd"
        The method to be used for bad channel detection
    std_mad_threshold : float, default: 5
        The standard deviation/mad multiplier threshold
    psd_hf_threshold : float, default: 0.02
        For coherence+psd - an absolute threshold (uV^2/Hz) used as a cutoff for noise channels.
        Channels with average power at >80% Nyquist larger than this threshold
        will be labeled as noise
    dead_channel_threshold : float, default: -0.5
        For coherence+psd - threshold for channel coherence below which channels are labeled as dead
    noisy_channel_threshold : float, default: 1
        Threshold for channel coherence above which channels are labeled as noisy (together with psd condition)
    outside_channel_threshold : float, default: -0.75
        For coherence+psd - threshold for channel coherence above which channels at the edge of the recording are marked as outside
        of the brain
    outside_channels_location : "top" | "bottom" | "both", default: "top"
        For coherence+psd - location of the outside channels. If "top", only the channels at the top of the probe can be
        marked as outside channels. If "bottom", only the channels at the bottom of the probe can be
        marked as outside channels. If "both", both the channels at the top and bottom of the probe can be
        marked as outside channels
    n_neighbors : int, default: 11
        For coeherence+psd - number of channel neighbors to compute median filter (needs to be odd)
    nyquist_threshold : float, default: 0.8
        For coherence+psd - frequency with respect to Nyquist (Fn=1) above which the mean of the PSD is calculated and compared
        with psd_hf_threshold
    direction : "x" | "y" | "z", default: "y"
        For coherence+psd - the depth dimension
    highpass_filter_cutoff : float, default: 300
        If the recording is not filtered, the cutoff frequency of the highpass filter
    chunk_duration_s : float, default: 0.5
        Duration of each chunk
    num_random_chunks : int, default: 100
        Number of random chunks
        Having many chunks is important for reproducibility.
    welch_window_ms : float, default: 10
        Window size for the scipy.signal.welch that will be converted to nperseg
    neighborhood_r2_threshold : float, default: 0.95
        R^2 threshold for the neighborhood_r2 method.
    neighborhood_r2_radius_um : float, default: 30
        Spatial radius below which two channels are considered neighbors in the neighborhood_r2 method.
    seed : int or None, default: None
        The random seed to extract chunks

    Returns
    -------
    bad_channel_ids : np.array
        The identified bad channel ids
    channel_labels : np.array of str
        Channels labels depending on the method:
          * (coeherence+psd) good/dead/noise/out
          * (std, mad) good/noise

    Examples
    --------

    >>> import spikeinterface.preprocessing as spre
    >>> bad_channel_ids, channel_labels = spre.detect_bad_channels(recording, method="coherence+psd")
    >>> # remove bad channels
    >>> recording_clean = recording.remove_channels(bad_channel_ids)

    Notes
    -----
    For details refer to:
    International Brain Laboratory et al. (2022). Spike sorting pipeline for the International Brain Laboratory.
    https://www.internationalbrainlab.com/repro-ephys
    """
    import scipy.stats

    method_list = ("std", "mad", "coherence+psd", "neighborhood_r2")
    assert method in method_list, f"{method} is not a valid method. Available methods are {method_list}"

    # Get random subset of data to estimate from
    random_chunk_kwargs = dict(
        num_chunks_per_segment=num_random_chunks,
        chunk_size=int(chunk_duration_s * recording.sampling_frequency),
        seed=seed,
    )

    # If recording is not filtered, apply a highpass filter
    if not recording.is_filtered():
        recording_hp = highpass_filter(recording, freq_min=highpass_filter_cutoff)
    else:
        recording_hp = recording

    # Adjust random chunk kwargs based on method
    if method in ("std", "mad"):
        random_chunk_kwargs["return_scaled"] = False
        random_chunk_kwargs["concatenated"] = True
    elif method == "coherence+psd":
        random_chunk_kwargs["return_scaled"] = True
        random_chunk_kwargs["concatenated"] = False
    elif method == "neighborhood_r2":
        random_chunk_kwargs["return_scaled"] = False
        random_chunk_kwargs["concatenated"] = False

    random_data = get_random_data_chunks(recording_hp, **random_chunk_kwargs)

    channel_labels = np.zeros(recording.get_num_channels(), dtype="U5")
    channel_labels[:] = "good"

    if method in ("std", "mad"):
        if method == "std":
            deviations = np.std(random_data, axis=0)
        else:
            deviations = scipy.stats.median_abs_deviation(random_data, axis=0)
        thresh = std_mad_threshold * np.median(deviations)
        mask = deviations > thresh
        bad_channel_ids = recording.channel_ids[mask]
        channel_labels[mask] = "noise"

    elif method == "coherence+psd":
        # some checks
        assert recording.has_scaleable_traces(), (
            "The 'coherence+psd' method uses thresholds assuming the traces are in uV, "
            "but the recording does not have scaled traces. If the recording is already scaled, "
            "you need to set gains and offsets: "
            ">>> recording.set_channel_gains(1); recording.set_channel_offsets(0)"
        )
        assert 0 < nyquist_threshold < 1, "nyquist_threshold must be between 0 and 1"

        # If location are not sorted, estimate forward and reverse sorting
        channel_locations = recording.get_channel_locations()
        dim = ["x", "y", "z"].index(direction)
        assert dim < channel_locations.shape[1], f"Direction {direction} is wrong"
        order_f, order_r = order_channels_by_depth(recording=recording, dimensions=("x", "y"))
        if np.all(np.diff(order_f) == 1):
            # already ordered
            order_f = None
            order_r = None

        # Create empty channel labels and fill with bad-channel detection estimate for each chunk
        chunk_channel_labels = np.zeros((recording.get_num_channels(), len(random_data)), dtype=np.int8)

        for i, random_chunk in enumerate(random_data):
            random_chunk_sorted = random_chunk[:, order_f] if order_f is not None else random_chunk
            chunk_labels = detect_bad_channels_ibl(
                raw=random_chunk_sorted,
                fs=recording.sampling_frequency,
                psd_hf_threshold=psd_hf_threshold,
                dead_channel_thr=dead_channel_threshold,
                noisy_channel_thr=noisy_channel_threshold,
                outside_channel_thr=outside_channel_threshold,
                n_neighbors=n_neighbors,
                nyquist_threshold=nyquist_threshold,
                welch_window_ms=welch_window_ms,
                outside_channels_location=outside_channels_location,
            )
            chunk_channel_labels[:, i] = chunk_labels[order_r] if order_r is not None else chunk_labels

        # Take the mode of the chunk estimates as final result. Convert to binary good / bad channel output.
        mode_channel_labels, _ = scipy.stats.mode(chunk_channel_labels, axis=1, keepdims=False)

        (bad_inds,) = np.where(mode_channel_labels != 0)
        bad_channel_ids = recording.channel_ids[bad_inds]

        channel_labels[mode_channel_labels == 1] = "dead"
        channel_labels[mode_channel_labels == 2] = "noise"
        channel_labels[mode_channel_labels == 3] = "out"

        if bad_channel_ids.size > recording.get_num_channels() / 3:
            warnings.warn(
                "Over 1/3 of channels are detected as bad. In the presence of a high"
                "number of dead / noisy channels, bad channel detection may fail "
                "(good channels may be erroneously labeled as dead)."
            )

    elif method == "neighborhood_r2":
        # make neighboring channels structure. this should probably be a function in core.
        geom = recording.get_channel_locations()
        num_channels = recording.get_num_channels()
        chan_distances = np.linalg.norm(geom[:, None, :] - geom[None, :, :], axis=2)
        np.fill_diagonal(chan_distances, neighborhood_r2_radius_um + 1)
        neighbors_mask = chan_distances < neighborhood_r2_radius_um
        if neighbors_mask.sum(axis=1).min() < 1:
            warnings.warn(
                f"neighborhood_r2_radius_um={neighborhood_r2_radius_um} led "
                "to channels with no neighbors for this geometry, which has "
                f"minimal channel distance {chan_distances.min()}um. These "
                "channels will not be marked as bad, but you might want to "
                "check them."
            )
        max_neighbors = neighbors_mask.sum(axis=1).max()
        channel_index = np.full((num_channels, max_neighbors), num_channels)
        for c in range(num_channels):
            my_neighbors = np.flatnonzero(neighbors_mask[c])
            channel_index[c, : my_neighbors.size] = my_neighbors

        # get the correlation of each channel with its neighbors' median inside each chunk
        # note that we did not concatenate the chunks here
        correlations = []
        for chunk in random_data:
            chunk = chunk.astype(np.float32, copy=False)
            chunk = chunk - np.median(chunk, axis=0, keepdims=True)
            padded_chunk = np.pad(chunk, [(0, 0), (0, 1)], constant_values=np.nan)
            # channels with no neighbors will get a pure-nan median trace here
            neighbmeans = np.nanmedian(
                padded_chunk[:, channel_index],
                axis=2,
            )
            denom = np.sqrt(np.nanmean(np.square(chunk), axis=0) * np.nanmean(np.square(neighbmeans), axis=0))
            denom[denom == 0] = 1
            # channels with no neighbors will get a nan here
            chunk_correlations = np.nanmean(chunk * neighbmeans, axis=0) / denom
            correlations.append(chunk_correlations)

        # now take the median over chunks and threshold to finish
        median_correlations = np.nanmedian(correlations, 0)
        r2s = median_correlations**2
        # channels with no neighbors will have r2==nan, and nan<x==False always
        bad_channel_mask = r2s < neighborhood_r2_threshold
        bad_channel_ids = recording.channel_ids[bad_channel_mask]
        channel_labels[bad_channel_mask] = "noise"

    return bad_channel_ids, channel_labels


# ----------------------------------------------------------------------------------------------
# IBL Detect Bad Channels
# ----------------------------------------------------------------------------------------------


def detect_bad_channels_ibl(
    raw,
    fs,
    psd_hf_threshold,
    dead_channel_thr=-0.5,
    noisy_channel_thr=1.0,
    outside_channel_thr=-0.75,
    n_neighbors=11,
    nyquist_threshold=0.8,
    welch_window_ms=0.3,
    outside_channels_location="top",
):
    """
    Bad channels detection for Neuropixel probes developed by IBL

    Parameters
    ----------
    raw : traces
        (num_samples, n_channels) raw traces
    fs : float
        sampling frequency
    psd_hf_threshold : float
        Threshold for high frequency PSD. If mean PSD above `nyquist_threshold` * fn is greater than this
        value, channels are flagged as noisy (together with channel coherence condition).
    dead_channel_thr : float, default: -0.5
        Threshold for channel coherence below which channels are labeled as dead
    noisy_channel_thr : float, default: 1
        Threshold for channel coherence above which channels are labeled as noisy (together with psd condition)
    outside_channel_thr : float, default: -0.75
        Threshold for channel coherence above which channels
    n_neighbors : int, default: 11
        Number of neighbors to compute median fitler
    nyquist_threshold : float, default: 0.8
        Threshold on Nyquist frequency to calculate HF noise band
    welch_window_ms : float, default: 0.3
        Window size for the scipy.signal.welch that will be converted to nperseg
    outside_channels_location : "top" | "bottom" | "both", default: "top"
        Location of the outside channels. If "top", only the channels at the top of the probe can be
        marked as outside channels. If "bottom", only the channels at the bottom of the probe can be
        marked as outside channels. If "both", both the channels at the top and bottom of the probe can be
        marked as outside channels

    Returns
    -------
    1d array
        Channels labels: 0: good,  1: dead low coherence / amplitude, 2: noisy, 3: outside of the brain
    """
    _, nc = raw.shape
    raw = raw - np.mean(raw, axis=0)[np.newaxis, :]
    nperseg = int(welch_window_ms * fs / 1000)
    import scipy.signal

    fscale, psd = scipy.signal.welch(raw, fs=fs, axis=0, window="hann", nperseg=nperseg)

    # compute similarities
    ref = np.median(raw, axis=1)
    xcorr = np.sum(raw * ref[:, np.newaxis], axis=0) / np.sum(ref**2)

    # compute coherence
    xcorr_neighbors = detrend(xcorr, n_neighbors)
    xcorr_distant = xcorr - detrend(xcorr, n_neighbors) - 1

    # make recommendation
    psd_hf = np.mean(psd[fscale > (fs / 2 * nyquist_threshold), :], axis=0)

    ichannels = np.zeros(nc, dtype=int)
    idead = np.where(xcorr_neighbors < dead_channel_thr)[0]
    inoisy = np.where(np.logical_or(psd_hf > psd_hf_threshold, xcorr_neighbors > noisy_channel_thr))[0]

    ichannels[idead] = 1
    ichannels[inoisy] = 2

    # the channels outside of the brains are the contiguous channels below the threshold on the trend coherency
    # the chanels outside need to be at the extreme of the probe
    (ioutside,) = np.where(xcorr_distant < outside_channel_thr)
    a = np.cumsum(np.r_[0, np.diff(ioutside) - 1])
    if ioutside.size > 0:
        if outside_channels_location == "top":
            # channels are sorted bottom to top, so the last channel needs to be (nc - 1)
            if ioutside[-1] == (nc - 1):
                ioutside = ioutside[(a == np.max(a)) & (a > 0)]
                ichannels[ioutside] = 3
        elif outside_channels_location == "bottom":
            # outside channels are at the bottom of the probe, so the first channel needs to be 0
            if ioutside[0] == 0:
                ioutside = ioutside[(a == np.min(a)) & (a < np.max(a))]
                ichannels[ioutside] = 3
        else:  # both extremes are considered
            if ioutside[-1] == (nc - 1) or ioutside[0] == 0:
                ioutside = ioutside[(a == np.max(a)) | (a == np.min(a))]
                ichannels[ioutside] = 3

    return ichannels


# ----------------------------------------------------------------------------------------------
# IBL Helpers
# ----------------------------------------------------------------------------------------------


def detrend(x, nmed):
    """
    Subtract the trend from a vector
    The trend is a median filtered version of the said vector with tapering
    :param x : input vector
    :param nmed : number of points of the median filter
    :return : np.array
    """
    ntap = int(np.ceil(nmed / 2))
    xf = np.r_[np.zeros(ntap) + x[0], x, np.zeros(ntap) + x[-1]]

    import scipy.signal

    xf = scipy.signal.medfilt(xf, nmed)[ntap:-ntap]
    return x - xf
