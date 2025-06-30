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


_maxwell_event_dtype = np.dtype(
    [("id", "int8"), ("frame", "uint32"), ("time", "float64"), ("state", "uint32"), ("message", "object")]
)


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

        if version < 20190530:
            raise NotImplementedError(f"Version {self.version} not supported")

        # get ttl events
        bits = h5_file["bits"]

        channel_ids = np.zeros((0), dtype=np.int8)
        if len(bits) > 0:
            bit_state = bits["bits"]
            channel_ids = np.int8(np.unique(bit_state[bit_state != 0]))
            if -1 in channel_ids or 1 in channel_ids:
                raise ValueError("TTL bits cannot be -1 or 1.")

        # access data_store from h5_file
        data_store_keys = [x for x in h5_file["data_store"].keys()]
        data_store_keys_id = [
            ("events" in h5_file["data_store"][x].keys()) and ("groups" in h5_file["data_store"][x].keys())
            for x in data_store_keys
        ]
        data_store = data_store_keys[data_store_keys_id.index(True)]

        # get stim events
        event_raw = h5_file["data_store"][data_store]["events"]
        channel_ids_stim = np.int8(np.unique([x[1] for x in event_raw]))
        if -1 in channel_ids_stim or 0 in channel_ids_stim:
            raise ValueError("Stimulation bits cannot be -1 or 0.")
        if len(channel_ids) > 0:
            if set(channel_ids) & set(channel_ids_stim):
                raise ValueError("TTL and stimulation bits overlap.")
        channel_ids = np.concatenate((channel_ids, channel_ids_stim), dtype=np.int8)

        # set spike events channel == -1
        spike_raw = h5_file["data_store"][data_store]["spikes"]
        if len(spike_raw) > 0:
            channel_ids = np.concatenate((channel_ids, [-1]), dtype=np.int8)

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
        bits = self.bits

        # get ttl events
        channel_ids = np.zeros((0), dtype=np.int8)
        bit_channel = np.zeros((0), dtype=np.int8)
        bit_frameno = np.zeros((0), dtype=np.uint32)
        bit_state = np.zeros((0), dtype=np.uint32)
        bit_message = np.zeros((0), dtype=object)
        if len(bits) > 0:
            good_idx = np.where(bits["bits"] != 0)[0]
            channel_ids = np.concatenate((channel_ids, np.int8(np.unique(bits["bits"][good_idx]))))
            if 1 in channel_ids:
                raise ValueError("TTL bits cannot be 1.")
            bit_channel = np.concatenate((bit_channel, np.uint8(bits["bits"][good_idx])))
            bit_frameno = np.concatenate((bit_frameno, np.uint32(bits["frameno"][good_idx])))
            bit_state = np.concatenate((bit_state, np.uint32(bits["bits"][good_idx])))
            bit_message = np.concatenate((bit_message, [b"{}\n"] * len(bit_state)), dtype=object)

        # access data_store from h5_file
        h5_file = self.h5_file
        data_store_keys = [x for x in h5_file["data_store"].keys()]
        data_store_keys_id = [
            ("events" in h5_file["data_store"][x].keys()) and ("groups" in h5_file["data_store"][x].keys())
            for x in data_store_keys
        ]
        data_store = data_store_keys[data_store_keys_id.index(True)]

        # get stim events
        event_raw = h5_file["data_store"][data_store]["events"]
        channel_ids_stim = np.int8(np.unique([x[1] for x in event_raw]))
        stim_arr = np.array(event_raw)
        bit_channel_stim = stim_arr["eventtype"]
        bit_frameno_stim = stim_arr["frameno"]
        bit_state_stim = stim_arr["eventid"]
        bit_message_stim = stim_arr["eventmessage"]

        # get spike events
        spike_raw = h5_file["data_store"][data_store]["spikes"]
        if len(spike_raw) > 0:
            channel_ids_spike = np.int8([-1])
        spike_arr = np.array(spike_raw)
        bit_channel_spike = -np.ones(len(spike_arr), dtype=np.int8)
        bit_frameno_spike = spike_arr["frameno"]
        bit_state_spike = spike_arr["channel"]
        bit_message_spike = spike_arr["amplitude"]

        # final array in order: spikes, stims, ttl
        bit_channel = np.concatenate((bit_channel_spike, bit_channel_stim, bit_channel))
        bit_frameno = np.concatenate((bit_frameno_spike, bit_frameno_stim, bit_frameno))
        bit_state = np.concatenate((bit_state_spike, bit_state_stim, bit_state))
        bit_message = np.concatenate((bit_message_spike, bit_message_stim, bit_message))

        first_frame = h5_file["data_store"][data_store]["groups/routed/frame_nos"][0]
        bit_frameno = bit_frameno - first_frame

        if channel_id is not None:
            good_idx = np.where(bit_channel == channel_id)[0]
            bit_channel = bit_channel[good_idx]
            bit_frameno = bit_frameno[good_idx]
            bit_state = bit_state[good_idx]
            bit_message = bit_message[good_idx]
        event = np.zeros(len(bit_channel), dtype=_maxwell_event_dtype)
        event["id"] = bit_channel
        event["frame"] = bit_frameno
        event["time"] = np.float64(bit_frameno) / self.fs
        event["state"] = bit_state
        event["message"] = bit_message

        if start_time is not None:
            event = event[event["time"] >= start_time]
        if end_time is not None:
            event = event[event["time"] < end_time]
        return event


read_maxwell = define_function_from_class(source_class=MaxwellRecordingExtractor, name="read_maxwell")
read_maxwell_event = define_function_from_class(source_class=MaxwellEventExtractor, name="read_maxwell_event")
