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
    """

    mode = "file"

    # Specific parameters for WhiteMatter format
    DTYPE = "int16"
    GAIN_TO_UV = 6.25e3 / 32768
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
    ):
        super().__init__(
            file_paths=[file_path],
            sampling_frequency=sampling_frequency,
            num_channels=num_channels,
            dtype=self.DTYPE,
            time_axis=self.TIME_AXIS,
            file_offset=self.FILE_OFFSET,
            gain_to_uV=self.GAIN_TO_UV,
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
        }


# Define function equivalent for convenience
read_whitematter = define_function_from_class(source_class=WhiteMatterRecordingExtractor, name="read_whitematter")
