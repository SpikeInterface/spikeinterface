from __future__ import annotations

import numpy as np
from pathlib import Path

import probeinterface

from spikeinterface import BaseEvent, BaseEventSegment
from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class MaxwellRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from Maxwell device.
    It handles MaxOne (old and new format) and MaxTwo.

    Based on :py:class:`neo.rawio.MaxwellRawIO`

    Parameters
    ----------
    file_path : str
        The file path to the maxwell h5 file.
    stream_id : str, default: None
        If there are several streams, specify the stream id you want to load.
        For MaxTwo when there are several wells at the same time you
        need to specify stream_id='well000' or 'well0001', etc.
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
    names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.
    rec_name : str, default: None
        When the file contains several recordings you need to specify the one
        you want to extract. (rec_name='rec0000').
    install_maxwell_plugin : bool, default: False
        If True, install the maxwell plugin for neo.
    block_index : int, default: None
        If there are several blocks (experiments), specify the block index you want to load
    """

    NeoRawIOClass = "MaxwellRawIO"

    def __init__(
        self,
        file_path,
        stream_id=None,
        stream_name=None,
        block_index=None,
        all_annotations=False,
        rec_name=None,
        install_maxwell_plugin=False,
        use_names_as_ids: bool = False,
    ):
        if install_maxwell_plugin:
            self.install_maxwell_plugin()

        neo_kwargs = self.map_to_neo_kwargs(file_path, rec_name)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            block_index=block_index,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )

        self.extra_requirements.append("h5py")

        # well_name is stream_id
        well_name = self.stream_id
        # rec_name auto set by neo
        rec_name = self.neo_reader.rec_name
        probe = probeinterface.read_maxwell(file_path, well_name=well_name, rec_name=rec_name)
        self.set_probe(probe, in_place=True)
        self.set_property("electrode", self.get_property("contact_vector")["electrode"])
        self._kwargs.update(dict(file_path=str(Path(file_path).absolute()), rec_name=rec_name))

    @classmethod
    def map_to_neo_kwargs(cls, file_path, rec_name=None):
        neo_kwargs = {"filename": str(file_path), "rec_name": rec_name}
        return neo_kwargs

    def install_maxwell_plugin(self, force_download=False):
        from neo.rawio.maxwellrawio import auto_install_maxwell_hdf5_compression_plugin

        auto_install_maxwell_hdf5_compression_plugin(force_download=False)


_maxwell_event_dtype = np.dtype([("frame", "int64"), ("state", "int8"), ("time", "float64")])


class MaxwellEventExtractor(BaseEvent):
    """
    Class for reading TTL events from Maxwell files.
    """

    def __init__(self, file_path):
        import h5py

        self.file_path = file_path
        h5_file = h5py.File(self.file_path, mode="r")
        version = int(h5_file["version"][0].decode())
        fs = 20000

        bits = h5_file["bits"]
        bit_states = bits["bits"]
        channel_ids = np.unique(bit_states[bit_states != 0])

        BaseEvent.__init__(self, channel_ids, structured_dtype=_maxwell_event_dtype)
        event_segment = MaxwellEventSegment(h5_file, version, fs)
        self.add_event_segment(event_segment)


class MaxwellEventSegment(BaseEventSegment):
    def __init__(self, h5_file, version, fs):
        BaseEventSegment.__init__(self)
        self.h5_file = h5_file
        self.version = version
        self.bits = self.h5_file["bits"]
        self.fs = fs

    def get_events(self, channel_id, start_time, end_time):
        if self.version != 20160704:
            raise NotImplementedError(f"Version {self.version} not supported")

        framevals = self.h5_file["sig"][-2:, 0]
        first_frame = framevals[1] << 16 | framevals[0]
        ttl_frames = self.bits["frameno"] - first_frame
        ttl_states = self.bits["bits"]
        if channel_id is not None:
            bits_channel_idx = np.where((ttl_states == channel_id) | (ttl_states == 0))[0]
            ttl_frames = ttl_frames[bits_channel_idx]
            ttl_states = ttl_states[bits_channel_idx]
        ttl_states[ttl_states == 0] = -1
        event = np.zeros(len(ttl_frames), dtype=_maxwell_event_dtype)
        event["frame"] = ttl_frames
        event["time"] = ttl_frames / self.fs
        event["state"] = ttl_states

        if start_time is not None:
            event = event[event["time"] >= start_time]
        if end_time is not None:
            event = event[event["time"] < end_time]
        return event


read_maxwell = define_function_from_class(source_class=MaxwellRecordingExtractor, name="read_maxwell")
read_maxwell_event = define_function_from_class(source_class=MaxwellEventExtractor, name="read_maxwell_event")
