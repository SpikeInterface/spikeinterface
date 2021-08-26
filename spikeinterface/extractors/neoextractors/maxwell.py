from spikeinterface import BaseEvent, BaseEventSegment

from .neobaseextractor import NeoBaseRecordingExtractor
import probeinterface as pi
import numpy as np


class MaxwellRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from Maxwell device.
    It handle MaxOne (old and new format) and MaxTwo.
    
    Based on neo.rawio.IntanRawIO
    
    Parameters
    ----------
    file_path: str
        Path to maxwell h5 file
    stream_id: str or None
        For MaxTwo when there are several wells at the same time you
        need to specify stream_id='well000' or 'well0001' or ...
    rec_name: str or None
        When the file contains several blocks (aka recordings) you need to specify the one
        you want to extract. (rec_name='rec0000')
    """
    mode = 'file'
    NeoRawIOClass = 'MaxwellRawIO'

    def __init__(self, file_path, stream_id=None, rec_name=None):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, rec_name=rec_name, **neo_kwargs)

        # well_name is stream_id
        well_name = self.stream_id
        # rec_name auto set by neo
        rec_name = self.neo_reader.rec_name
        probe = pi.read_maxwell(file_path, well_name=well_name, rec_name=rec_name)
        self.set_probe(probe, in_place=True)
        self.set_property("electrode", self.get_property("contact_vector")["electrode"])
        self._kwargs = dict(file_path=str(file_path), stream_id=stream_id, rec_name=rec_name)


def read_maxwell(*args, **kwargs):
    recording = MaxwellRecordingExtractor(*args, **kwargs)
    return recording


read_maxwell.__doc__ = MaxwellRecordingExtractor.__doc__


_maxwell_event_dtype = np.dtype([("frame", "int64"), ("state", "int8"), ("time", "float64")])


class MaxwellEventExtractor(BaseEvent):
    """
    Class for reading TTL events from Maxwell files

    """
    def __init__(self, file_path):
        import h5py
        self.file_path = file_path
        h5_file = h5py.File(self.file_path, mode='r')
        version = int(h5_file["version"][0].decode())
        fs = 20000

        bits = h5_file['bits']
        bit_states = bits['bits']
        channel_ids = np.unique(bit_states[bit_states != 0])

        BaseEvent.__init__(self, channel_ids, structured_dtype=_maxwell_event_dtype)
        event_segment = MaxwellEventSegment(h5_file, version, fs)
        self.add_event_segment(event_segment)


class MaxwellEventSegment(BaseEventSegment):
    def __init__(self, h5_file, version, fs):
        BaseEventSegment.__init__(self)
        self.h5_file = h5_file
        self.version = version
        self.bits = self.h5_file['bits']
        self.fs = fs

    def get_event_times(self, channel_id, start_time, end_time):
        if self.version == 20160704:
            framevals = self.h5_file["sig"][-2:, 0]
            first_frame = framevals[1] << 16 | framevals[0]
            ttl_frames = self.bits['frameno'] - first_frame
            ttl_states = self.bits['bits']
            if channel_id is not None:
                bits_channel_idx = np.where((ttl_states == channel_id) | (ttl_states == 0))[0]
                ttl_frames = ttl_frames[bits_channel_idx]
                ttl_states = ttl_states[bits_channel_idx]
            if start_time is not None:
                bit_idxs = np.where(ttl_frames / self.fs >= start_time)[0]
                ttl_frames = ttl_frames[bit_idxs]
                ttl_states = ttl_states[bit_idxs]
            if end_time is not None:
                bit_idxs = np.where(ttl_frames / self.fs < end_time)[0]
                ttl_frames = ttl_frames[bit_idxs]
                ttl_states = ttl_states[bit_idxs]
            ttl_states[ttl_states == 0] = -1
            event = np.zeros(len(ttl_frames), dtype=_maxwell_event_dtype)
            event["frame"] = ttl_frames
            event["time"] = ttl_frames / self.fs
            event["state"] = ttl_states
        else:
            raise NotImplementedError

        return event


def read_maxwell_event(*args, **kwargs):
    event = MaxwellEventExtractor(*args, **kwargs)
    return event


read_maxwell_event.__doc__ = MaxwellEventExtractor.__doc__
