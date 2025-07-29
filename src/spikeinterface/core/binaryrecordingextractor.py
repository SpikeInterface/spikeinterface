from __future__ import annotations
import mmap
import warnings
from pathlib import Path

import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment
from .core_tools import define_function_from_class
from .recording_tools import write_binary_recording
from .job_tools import _shared_job_kwargs_doc


class BinaryRecordingExtractor(BaseRecording):
    """
    RecordingExtractor for a binary format

    Parameters
    ----------
    file_paths : str or Path or list
        Path to the binary file
    sampling_frequency : float
        The sampling frequency
    num_channels : int
        Number of channels
    dtype : str or dtype
        The dtype of the binary file
    time_axis : int, default: 0
        The axis of the time dimension
    t_starts : None or list of float, default: None
        Times in seconds of the first sample for each segment
    channel_ids : list, default: None
        A list of channel ids
    file_offset : int, default: 0
        Number of bytes in the file to offset by during memmap instantiation.
    gain_to_uV : float or array-like, default: None
        The gain to apply to the traces
    offset_to_uV : float or array-like, default: None
        The offset to apply to the traces
    is_filtered : bool or None, default: None
        If True, the recording is assumed to be filtered. If None, is_filtered is not set.

    Notes
    -----
    When both num_channels and num_chan are provided, `num_channels` is used and `num_chan` is ignored.

    Returns
    -------
    recording : BinaryRecordingExtractor
        The recording Extractor
    """

    def __init__(
        self,
        file_paths,
        sampling_frequency,
        dtype,
        num_channels: int,
        t_starts=None,
        channel_ids=None,
        time_axis=0,
        file_offset=0,
        gain_to_uV=None,
        offset_to_uV=None,
        is_filtered=None,
    ):

        if channel_ids is None:
            channel_ids = list(range(num_channels))
        else:
            assert len(channel_ids) == num_channels, "Provided recording channels have the wrong length"

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        if isinstance(file_paths, list):
            # several segment
            file_path_list = [Path(p) for p in file_paths]
        else:
            # one segment
            file_path_list = [Path(file_paths)]

        if t_starts is not None:
            assert len(t_starts) == len(file_path_list), "t_starts must be a list of the same size as file_paths"
            t_starts = [float(t_start) for t_start in t_starts]

        dtype = np.dtype(dtype)

        for i, file_path in enumerate(file_path_list):
            if t_starts is None:
                t_start = None
            else:
                t_start = t_starts[i]
            rec_segment = BinaryRecordingSegment(
                file_path, sampling_frequency, t_start, num_channels, dtype, time_axis, file_offset
            )
            self.add_recording_segment(rec_segment)

        if is_filtered is not None:
            self.annotate(is_filtered=is_filtered)

        if gain_to_uV is not None:
            self.set_channel_gains(gain_to_uV)

        if offset_to_uV is not None:
            self.set_channel_offsets(offset_to_uV)

        self._kwargs = {
            "file_paths": [str(Path(e).absolute()) for e in file_path_list],
            "sampling_frequency": sampling_frequency,
            "t_starts": t_starts,
            "num_channels": num_channels,
            "dtype": dtype.str,
            "channel_ids": channel_ids,
            "time_axis": time_axis,
            "file_offset": file_offset,
            "gain_to_uV": gain_to_uV,
            "offset_to_uV": offset_to_uV,
            "is_filtered": is_filtered,
        }

    @staticmethod
    def write_recording(recording, file_paths, dtype=None, **job_kwargs):
        """
        Save the traces of a recording extractor in binary .dat format.

        Parameters
        ----------
        recording : RecordingExtractor
            The recording extractor object to be saved in .dat format
        file_paths : str
            The path to the file.
        dtype : dtype, default: None
            Type of the saved data
        {}
        """
        write_binary_recording(recording, file_paths=file_paths, dtype=dtype, **job_kwargs)

    def is_binary_compatible(self) -> bool:
        return True

    def get_binary_description(self):
        d = dict(
            file_paths=self._kwargs["file_paths"],
            dtype=np.dtype(self._kwargs["dtype"]),
            num_channels=self._kwargs["num_channels"],
            time_axis=self._kwargs["time_axis"],
            file_offset=self._kwargs["file_offset"],
        )
        return d

    def __del__(self):
        """
        Ensures that all segment resources are properly cleaned up when this recording extractor is deleted.
        Closes any open file handles in the recording segments.
        """
        # Close all recording segments
        if hasattr(self, "_recording_segments"):
            for segment in self._recording_segments:
                # This will trigger the __del__ method of the BinaryRecordingSegment
                # which will close the file handle
                del segment


BinaryRecordingExtractor.write_recording.__doc__ = BinaryRecordingExtractor.write_recording.__doc__.format(
    _shared_job_kwargs_doc
)


class BinaryRecordingSegment(BaseRecordingSegment):
    def __init__(self, file_path, sampling_frequency, t_start, num_channels, dtype, time_axis, file_offset):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency, t_start=t_start)
        self.num_channels = num_channels
        self.dtype = np.dtype(dtype)
        self.file_offset = file_offset
        self.time_axis = time_axis
        self.file_path = file_path
        self.file = open(self.file_path, "rb")
        self.bytes_per_sample = self.num_channels * self.dtype.itemsize
        self.data_size_in_bytes = Path(file_path).stat().st_size - file_offset
        self.num_samples = self.data_size_in_bytes // self.bytes_per_sample

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex : Number of samples in the signal block
        """
        return self.num_samples

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | None = None,
    ) -> np.ndarray:

        # Calculate byte offsets for start and end frames
        start_byte = self.file_offset + start_frame * self.bytes_per_sample
        end_byte = self.file_offset + end_frame * self.bytes_per_sample

        # Calculate the length of the data chunk to load into memory
        length = end_byte - start_byte

        # The mmap offset must be a multiple of mmap.ALLOCATIONGRANULARITY
        memmap_offset, start_offset = divmod(start_byte, mmap.ALLOCATIONGRANULARITY)
        memmap_offset *= mmap.ALLOCATIONGRANULARITY

        # Adjust the length so it includes the extra data from rounding down
        # the memmap offset to a multiple of ALLOCATIONGRANULARITY
        length += start_offset

        # Create the mmap object
        memmap_obj = mmap.mmap(self.file.fileno(), length=length, access=mmap.ACCESS_READ, offset=memmap_offset)

        # Create a numpy array using the mmap object as the buffer
        # Note that the shape must be recalculated based on the new data chunk
        if self.time_axis == 0:
            shape = ((end_frame - start_frame), self.num_channels)
        else:
            shape = (self.num_channels, (end_frame - start_frame))

        # Now the entire array should correspond to the data between start_frame and end_frame, so we can use it directly
        traces = np.ndarray(
            shape=shape,
            dtype=self.dtype,
            buffer=memmap_obj,
            offset=start_offset,
        )

        if self.time_axis == 1:
            traces = traces.T

        if channel_indices is not None:
            traces = traces[:, channel_indices]

        return traces

    def __del__(self):
        # Ensure that the file handle is closed when the segment is garbage-collected
        try:
            if hasattr(self, "file") and self.file and not self.file.closed:
                self.file.close()
        except Exception as e:
            warnings.warn(f"Error closing file handle in BinaryRecordingSegment: {e}")
            pass


# For backward compatibility (old good time)
BinDatRecordingExtractor = BinaryRecordingExtractor

read_binary = define_function_from_class(source_class=BinaryRecordingExtractor, name="read_binary")
