from __future__ import annotations

import numpy as np

from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor
from spikeinterface.preprocessing.normalize_scale import ScaleRecording
from spikeinterface.core.core_tools import define_function_handling_dict_from_class


class ScaleToPhysicalUnits(ScaleRecording):
    """
    Scale raw traces to their physical units using gain_to_physical_unit and offset_to_physical_unit.

    This preprocessor uses the channel-specific gain and offset information
    stored in the recording extractor to convert the raw traces to their physical units.
    Most commonly this will be microvolts (µV) for voltage recordings, but some extractors
    might use different physical units (e.g., Newtons for force measurements).

    Parameters
    ----------
    recording : BaseRecording
        The recording extractor to be scaled. The recording extractor must
        have gain_to_physical_unit and offset_to_physical_unit properties set.

    Returns
    -------
    ScaleToPhysicalUnits
        The recording with traces scaled to physical units.

    Raises
    ------
    ValueError
        If the recording extractor does not have gain_to_physical_unit and offset_to_physical_unit properties.
    """

    name = "recording_in_physical_units"

    def __init__(self, recording):
        if "gain_to_physical_unit" not in recording.get_property_keys():
            error_msg = (
                "Recording must have 'gain_to_physical_unit' property to convert to physical units. \n"
                "Set the gain using `recording.set_property(key='gain_to_physical_unit', values=values)`."
            )
            raise ValueError(error_msg)
        if "offset_to_physical_unit" not in recording.get_property_keys():
            error_msg = (
                "Recording must have 'offset_to_physical_unit' property to convert to physical units. \n"
                "Set the offset using `recording.set_property(key='offset_to_physical_unit', values=values)`."
            )
            raise ValueError(error_msg)

        gain = recording.get_property("gain_to_physical_unit")
        offset = recording.get_property("offset_to_physical_unit")

        # Initialize parent ScaleRecording with the gain and offset values
        ScaleRecording.__init__(self, recording, gain=gain, offset=offset, dtype="float32")

        # Reset gain/offset since data is now in physical units
        self.set_property(key="gain_to_physical_unit", values=np.ones(recording.get_num_channels(), dtype="float32"))
        self.set_property(key="offset_to_physical_unit", values=np.zeros(recording.get_num_channels(), dtype="float32"))

        # Also reset channel gains and offsets
        self.set_channel_gains(gains=1.0)
        self.set_channel_offsets(offsets=0.0)


scale_to_physical_units = define_function_handling_dict_from_class(ScaleToPhysicalUnits, name="scale_to_physical_units")


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
    from spikeinterface.preprocessing.preprocessing_classes import ScaleRecording

    if not recording.has_scaleable_traces():
        error_msg = "Recording must have gains and offsets set to be scaled to µV"
        raise RuntimeError(error_msg)

    gain = recording.get_channel_gains()
    offset = recording.get_channel_offsets()

    scaled_to_uV_recording = ScaleRecording(recording, gain=gain, offset=offset, dtype="float32")

    # We do this so when get_traces(return_in_uV=True) is called, the return is the same.
    scaled_to_uV_recording.set_channel_gains(gains=1.0)
    scaled_to_uV_recording.set_channel_offsets(offsets=0.0)

    return scaled_to_uV_recording
