from __future__ import annotations

from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor


def scale_to_uV(recording: BasePreprocessor) -> BasePreprocessor:
    """
    Scale raw traces to microvolts (µV).

    This preprocessor uses the channel-specific gain and offset information
    stored in the recording extractor to convert the raw traces to µV units.

    Parameters
    ----------
    recording : BaseRecording
        The recording extractor to be scaled. The recording extractor must
        have gains and offsets otherwise an error will be raised.

    Raises
    ------
    AssertionError
        If the recording extractor does not have scaleable traces.
    """
    # To avoid a circular import
    from spikeinterface.preprocessing import ScaleRecording

    if not recording.has_scaleable_traces():
        error_msg = "Recording must have gains and offsets set to be scaled to µV"
        raise RuntimeError(error_msg)

    gain = recording.get_channel_gains()
    offset = recording.get_channel_offsets()

    scaled_to_uV_recording = ScaleRecording(recording, gain=gain, offset=offset, dtype="float32")

    # We do this so when get_traces(return_scaled=True) is called, the return is the same.
    scaled_to_uV_recording.set_channel_gains(gains=1.0)
    scaled_to_uV_recording.set_channel_offsets(offsets=0.0)

    return scaled_to_uV_recording
