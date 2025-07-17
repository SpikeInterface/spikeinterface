from __future__ import annotations
from pathlib import Path

import numpy as np

from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.core import concatenate_recordings
from .neobaseextractor import NeoBaseRecordingExtractor


class IntanRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a intan board. Supports rhd and rhs format.

    Based on :py:class:`neo.rawio.IntanRawIO`

    Parameters
    ----------
    file_path : str
        The file path to load the recordings from.
    stream_id : str, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    ignore_integrity_checks : bool, default: False.
        If True, data that violates integrity assumptions will be loaded. At the moment the only integrity
        check we perform is that timestamps are continuous. Setting this to True will ignore this check and set
        the attribute `discontinuous_timestamps` to True in the underlying neo object.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.

        In Intan the ids provided by NeoRawIO are the hardware channel ids while the names are custom names given by
        the user

    Examples
    --------
    >>> from spikeinterface.extractors import read_intan
    # intan amplifier data is stored in stream_id = '0'
    >>> recording = read_intan(file_path=r'my_data.rhd', stream_id='0')
    # intan has multi-file formats as well, but in this case our path should point to the header file 'info.rhd'
    >>> recording = read_intan(file_path=r'info.rhd', stream_id='0')
    """

    NeoRawIOClass = "IntanRawIO"

    def __init__(
        self,
        file_path,
        stream_id=None,
        stream_name=None,
        all_annotations=False,
        use_names_as_ids=False,
        ignore_integrity_checks: bool = False,
    ):
        neo_kwargs = self.map_to_neo_kwargs(file_path, ignore_integrity_checks=ignore_integrity_checks)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )

        amplifier_streams = ["RHS2000 amplifier channel", "RHD2000 amplifier channel"]
        if self.stream_name in amplifier_streams:
            self._add_channel_groups()

        self._kwargs.update(
            dict(file_path=str(Path(file_path).resolve()), ignore_integrity_checks=ignore_integrity_checks),
        )

    @classmethod
    def map_to_neo_kwargs(cls, file_path, ignore_integrity_checks: bool = False):

        neo_kwargs = {"filename": str(file_path), "ignore_integrity_checks": ignore_integrity_checks}
        return neo_kwargs

    def _add_channel_groups(self):

        num_channels = self.get_num_channels()
        groups = np.zeros(shape=num_channels, dtype="uint16")
        group_names = np.zeros(shape=num_channels, dtype="str")

        signal_header = self.neo_reader.header["signal_channels"]
        amplifier_signal_header = signal_header[signal_header["stream_id"] == self.stream_id]
        original_ids = amplifier_signal_header["id"]

        # The hard-coded IDS of intan ids is "Port-Number" (e.g. A-001, C-017, B-020, etc) for amplifier channels
        channel_ports = [id[:1] for id in original_ids if id[1] == "-"]

        # This should be A, B, C, D, ...
        amplifier_ports = np.unique(channel_ports).tolist()

        for port in amplifier_ports:
            channel_index = np.where(np.array(channel_ports) == port)
            group_names[channel_index] = port
            groups[channel_index] = amplifier_ports.index(port)

        self.set_channel_groups(groups)
        self.set_property(key="group_names", values=group_names)


read_intan = define_function_from_class(source_class=IntanRecordingExtractor, name="read_intan")


def read_intan_segmented(folder_path, **kwargs):
    """
    Read Intan traditional format split files from a folder and concatenate them in temporal order.

    Intan traditional format creates multiple files with time-based naming when recording for extended
    periods. This function automatically sorts the files by filename and concatenates them
    to create a continuous recording.

    Parameters
    ----------
    folder_path : str or Path
        Path to the folder containing split Intan files (.rhd or .rhs extensions)
    **kwargs
        Additional keyword arguments passed to read_intan() for each file

    Returns
    -------
    recording : ConcatenatedRecordingExtractor
        Concatenated recording from all split files in temporal order

    Examples
    --------
    >>> from spikeinterface.extractors import read_intan_segmented
    >>> recording = read_intan_segmented("/path/to/intan/folder")
    """
    folder_path = Path(folder_path)

    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Folder path {folder_path} does not exist or is not a directory")

    # Find all Intan files
    file_path_list = [p for p in folder_path.iterdir() if p.suffix.lower() in [".rhd", ".rhs"]]

    if not file_path_list:
        raise ValueError(f"No Intan files (.rhd or .rhs) found in {folder_path}")

    # Sort files by filename (natural sort)
    file_path_list.sort(key=lambda x: x.name)

    # Read each file and create recording list
    recording_list = []
    for file_path in file_path_list:
        recording = read_intan(file_path, **kwargs)
        recording_list.append(recording)

    # Concatenate all recordings
    concatenated_recording = concatenate_recordings(recording_list=recording_list)

    return concatenated_recording
