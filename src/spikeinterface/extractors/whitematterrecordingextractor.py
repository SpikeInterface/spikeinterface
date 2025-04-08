from pathlib import Path
from typing import List, Union, Optional

import numpy as np

from spikeinterface.core import BaseRecording, BinaryRecordingExtractor
from spikeinterface.core.core_tools import define_function_from_class


class WhiteMatterRecordingExtractor(BinaryRecordingExtractor):
    """
    RecordingExtractor for the WhiteMatter binary format.

    The recording format is a raw binary file containing int16 data,
    with an 8-byte header offset.

    Parameters
    ----------
    file_paths : list or Path
        List of paths to the binary files.
    sampling_frequency : float
        The sampling frequency.
    num_channels : int
        Number of channels in the recording.
    channel_ids : list or None, default: None
        A list of channel ids. If None, channels are automatically numbered 0..num_channels-1.
    time_axis : int, default: 0
        The axis indicating time (0 for samples, 1 for channels). Default is 0.
    is_filtered : bool or None, default: None
        Whether the recording is filtered. If None, it is inferred from the file extension.
    """

    extractor_name = "WhiteMatterRecording"
    extensions = ["bin"]
    mode = "file"
    is_writable = False

    def __init__(
        self,
        file_paths: Union[List[Union[str, Path]], Union[str, Path]],
        sampling_frequency: float,
        num_channels: int,
        channel_ids: Optional[List] = None,
        time_axis: int = 0,
        is_filtered: Optional[bool] = None,
    ):
        # Specific parameters for WhiteMatter format
        dtype = "int16"
        gain_to_uV = 6.25e3 / 32768
        offset_to_uV = 0.0
        file_offset = 8

        BinaryRecordingExtractor.__init__(
            self,
            file_paths=file_paths,
            sampling_frequency=sampling_frequency,
            num_channels=num_channels,
            dtype=dtype,
            time_axis=time_axis,
            file_offset=file_offset,
            gain_to_uV=gain_to_uV,
            offset_to_uV=offset_to_uV,
            is_filtered=is_filtered,
            channel_ids=channel_ids,
        )

        # Store parameters for provenance
        self._kwargs = {
            "file_paths": [str(Path(p).absolute()) for p in self.file_paths],
            "sampling_frequency": sampling_frequency,
            "num_channels": num_channels,
            "channel_ids": channel_ids,
            "time_axis": time_axis,
            "is_filtered": is_filtered,
            # Include format-specific params in kwargs for clarity
            "dtype": dtype,
            "gain_to_uV": gain_to_uV,
            "offset_to_uV": offset_to_uV,
            "file_offset": file_offset,
        }


# Define function equivalent for convenience
read_whitematter = define_function_from_class(source_class=WhiteMatterRecordingExtractor, name="read_whitematter")
