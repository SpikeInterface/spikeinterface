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
    file_paths : list or Path
        List of paths to the binary files.
    sampling_frequency : float
        The sampling frequency.
    num_channels : int
        Number of channels in the recording.
    channel_ids : list or None, default: None
        A list of channel ids. If None, channel_ids = list(range(num_channels)).
    time_axis : int, default: 0
        The axis of the time dimension.
    is_filtered : bool or None, default: None
        If True, the recording is assumed to be filtered. If None, is_filtered is not set.
    """

    mode = "file"

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

        super().__init__(
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

    @staticmethod
    def write_recording(recording, file_paths, **job_kwargs):
        """
        Save the traces of a recording extractor in binary .dat format.

        Parameters
        ----------
        recording : RecordingExtractor
            The recording extractor object to be saved in .dat format
        file_paths : str
            The path to the file.
        """
        job_kwargs["byte_offset"] = 8
        BinaryRecordingExtractor.write_recording(recording, file_paths=file_paths, dtype="int16", **job_kwargs)


# Define function equivalent for convenience
read_whitematter = define_function_from_class(source_class=WhiteMatterRecordingExtractor, name="read_whitematter")
