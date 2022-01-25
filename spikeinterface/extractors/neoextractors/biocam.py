import probeinterface as pi
from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class BiocamRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a Biocam file from 3Brain.

    Based on neo.rawio.BiocamRawIO

    Parameters
    ----------
    file_path: str

    stream_id: str or None

    mea_pitch: float or None
    
    electrode_width: float or None

    """
    mode = 'file'
    NeoRawIOClass = 'BiocamRawIO'

    def __init__(self, file_path, stream_id=None, mea_pitch=None, electrode_width=None):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        # load probe from probeinterface
        probe_kwargs = {}
        if mea_pitch is not None:
            probe_kwargs["mea_pitch"] = mea_pitch
        if electrode_width is not None:
            probe_kwargs["electrode_width"] = electrode_width
        probe = pi.read_3brain(file_path, **probe_kwargs)
        self.set_probe(probe, in_place=True)
        self.set_property("row", self.get_property("contact_vector")["row"])
        self.set_property("col", self.get_property("contact_vector")["col"])
        self._kwargs = dict(file_path=str(file_path), stream_id=stream_id, mea_pitch=mea_pitch)


def read_biocam(*args, **kwargs):
    recording = BiocamRecordingExtractor(*args, **kwargs)
    return recording


read_biocam.__doc__ = BiocamRecordingExtractor.__doc__
