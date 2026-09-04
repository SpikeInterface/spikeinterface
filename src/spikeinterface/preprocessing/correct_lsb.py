import warnings
import numpy as np

from .normalize_scale import scale
from spikeinterface.core import get_random_data_chunks


def correct_lsb(recording, num_chunks_per_segment=20, chunk_size=10000, seed=None, verbose=False):
    """
    Estimates the LSB of the recording and divide traces by LSB
    to ensure LSB = 1. Medians are also subtracted to avoid rounding errors.

    Since the LSB is set at the acquisition level, it is shared across channels. The global
    LSB is therefore estimated as the mode (most frequent value) of the per-channel LSBs,
    which is robust both to undersampled channels (that overestimate the LSB) and to
    spuriously small per-channel values arising from rounding errors.

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
        return_in_uV=False,
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
    Estimate the global LSB as the mode of the per-channel LSBs.

    For each channel the LSB is the smallest difference between consecutive unique values.
    Since the LSB is shared across channels, the global estimate is the most frequent
    per-channel value, which is robust to undersampled channels (that overestimate the LSB)
    and to spuriously small values from rounding errors.

    Parameters
    ----------
    data : 2d np.array
        The data to use (n_samples, n_channels)

    Returns
    -------
    lsb_value : int
        The estimated LSB value. -1 indicates that lsb could not be estimated
    """
    num_channels = data.shape[1]
    per_channel_lsb = []
    for ch in np.arange(num_channels):
        # cast to int64 to avoid integer overflow in np.diff when consecutive
        # unique values are farther apart than the dtype range (e.g. int16)
        unique_values = np.unique(data[:, ch]).astype(np.int64)
        # It might happen that there is a single unique value (e.g. a channel is broken, or all zeros)
        if len(unique_values) > 1:
            per_channel_lsb.append(np.min(np.diff(unique_values)))
        else:
            # in this case we can't estimate the LSB for the channel
            continue

    if len(per_channel_lsb) == 0:
        return -1

    # mode: most frequent per-channel LSB (ties broken towards the smaller value)
    values, counts = np.unique(per_channel_lsb, return_counts=True)
    lsb_value = int(values[np.argmax(counts)])
    return lsb_value
