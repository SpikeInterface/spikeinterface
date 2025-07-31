from pathlib import Path
from typing import List, Union, Optional

from spikeinterface.core import BinaryRecordingExtractor
from spikeinterface.core.core_tools import define_function_from_class


class WhiteMatterRecordingExtractor(BinaryRecordingExtractor):
    """
    RecordingExtractor for the WhiteMatter binary format.

    The recording format is a raw binary file containing int16 data,
    with an 8-byte header offset.

    Parameters
    ----------
    file_path : Path
        Path to the binary file.
    sampling_frequency : float
        The sampling frequency.
    num_channels : int
        Number of channels in the recording.
    channel_ids : list or None, default: None
        A list of channel ids. If None, channel_ids = list(range(num_channels)).
    is_filtered : bool or None, default: None
        If True, the recording is assumed to be filtered. If None, `is_filtered` is not set.
    gain_to_uV : float | None, default: 0.190734863
        Micro-volt conversion factor (ADC count → µV).

    Notes
    -----
    • **Head-stage files** (64 neural chan.):
        *settingsHeadstages.xml* →
        `<CHANNEL … voltsperbit="1.907348633e-7" />`
        → 0.190 734 863 µV per count.

    • **Analog-panel files** (32 aux chan., ±10 V range in our example):
        *settingsAnalogPanel.xml* →
        `<CHANNEL … voltsperbit="3.0517578125e-4" />`
        → 305.175 781 µV per count.

    Use the value from the corresponding XML file when loading data.
    If *gain_to_uV* is left as *None* the extractor assumes the
    head-stage constant (0.190734863 µV/bit).  Different hardware
    ranges will have different *voltsperbit* numbers, so always double-check
    for your specific setup.
    """

    # Specific parameters for WhiteMatter format
    DTYPE = "int16"
    HEADSTAGE_GAIN_TO_UV = 0.190734863
    OFFSET_TO_UV = 0.0
    FILE_OFFSET = 8
    TIME_AXIS = 0
    # This extractor is based on a single example file without a formal specification from WhiteMatter.
    # The parameters above are currently assumed to be constant for all WhiteMatter files.
    # If you encounter issues with this extractor, these assumptions may need to be revisited.

    def __init__(
        self,
        file_path: Union[str, Path],
        sampling_frequency: float,
        num_channels: int,
        channel_ids: Optional[List] = None,
        is_filtered: Optional[bool] = None,
        gain_to_uV: Optional[float] = None,
    ):

        gain_to_uV = gain_to_uV if gain_to_uV is not None else self.HEADSTAGE_GAIN_TO_UV

        super().__init__(
            file_paths=[file_path],
            sampling_frequency=sampling_frequency,
            num_channels=num_channels,
            dtype=self.DTYPE,
            time_axis=self.TIME_AXIS,
            file_offset=self.FILE_OFFSET,
            gain_to_uV=gain_to_uV,
            offset_to_uV=self.OFFSET_TO_UV,
            is_filtered=is_filtered,
            channel_ids=channel_ids,
        )

        self._kwargs = {
            "file_path": file_path,
            "sampling_frequency": sampling_frequency,
            "num_channels": num_channels,
            "channel_ids": channel_ids,
            "is_filtered": is_filtered,
            "gain_to_uV": gain_to_uV,
        }


# Define function equivalent for convenience
read_whitematter = define_function_from_class(source_class=WhiteMatterRecordingExtractor, name="read_whitematter")
