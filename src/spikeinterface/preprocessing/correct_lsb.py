from __future__ import annotations

import warnings
import numpy as np

from .normalize_scale import scale
from spikeinterface.core import get_random_data_chunks


def correct_lsb(recording, num_chunks_per_segment=20, chunk_size=10000, seed=None, verbose=False):
    """
    Estimates the LSB of the recording and divide traces by LSB
    to ensure LSB = 1. Medians are also subtracted to avoid rounding errors.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be LSB-corrected.
    num_chunks_per_segment : int, default: 20
        Number of chunks per segment for random chunk
    chunk_size : int, default: 10000
        Size of a chunk in number for random chunk
    seed : int or None, default: None
        Random seed for random chunk
    verbose : bool, default: False
        If True, estimate LSB value is printed

    Returns
    -------
    correct_lsb_recording : ScaleRecording
        The recording extractor with corrected LSB
    """
    random_data = get_random_data_chunks(
        recording,
        num_chunks_per_segment=num_chunks_per_segment,
        chunk_size=chunk_size,
        concatenated=True,
        seed=seed,
        return_scaled=False,
    )
    # compute medians and lsb
    medians = np.median(random_data, axis=0)
    lsb = _estimate_lsb_from_data(random_data)

    if verbose:
        print(f"Estimated LSB value: {lsb}")

    # take care of cases where LSB cannot be estimated
    if lsb == -1:
        warnings.warn("LSB could not be estimated. No operation is applied")
        recording_lsb = recording
    elif lsb == 1:
        warnings.warn("Estimated LSB=1. No operation is applied")
        recording_lsb = recording
    else:
        dtype = recording.get_dtype()
        # first remove medians
        recording_lsb = scale(recording, gain=1.0, offset=-medians, dtype=dtype)
        # apply LSB division and instantiate parent
        recording_lsb = scale(recording_lsb, gain=1.0 / lsb, dtype=dtype)
        # if recording has scaled traces, correct gains
        if recording.has_scaleable_traces():
            recording_lsb.set_channel_gains(recording_lsb.get_channel_gains() * lsb)
    return recording_lsb


def _estimate_lsb_from_data(data):
    """
    Estimate LSB by taking the minimum LSB found across channels.

    Parameters
    ----------
    data : 2d np.array
        The data to use (n_samples, n_channels)

    Returns
    -------
    lsb_value : int
        The estimated LSB value. -1 indicates that lsb could not be estimated
    """
    lsb_value = None
    num_channels = data.shape[1]
    for ch in np.arange(num_channels):
        unique_values = np.unique(data[:, ch])
        # It might happen that there is a single unique value (e.g. a channel is broken, or all zeros)
        if len(unique_values) > 1:
            chan_lsb_val = np.min(np.diff(unique_values))
        else:
            # in this case we can't estimate the LSB for the channel
            continue

        if lsb_value is None:
            lsb_value = chan_lsb_val
        lsb_value = min(lsb_value, chan_lsb_val)
    if lsb_value is None:
        lsb_value = -1
    return lsb_value
