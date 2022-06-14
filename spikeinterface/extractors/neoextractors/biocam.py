import probeinterface as pi

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class BiocamRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a Biocam file from 3Brain.

    Based on neo.rawio.BiocamRawIO

    Parameters
    ----------
    file_path: str

    mea_pitch: float or None
    
    electrode_width: float or None

    stream_id: str or None
        If several stream, specify the one you want.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.


    """
    mode = 'file'
    NeoRawIOClass = 'BiocamRawIO'

    def __init__(self, file_path, mea_pitch=None, electrode_width=None, stream_id=None,  all_annotations=False):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)

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
        
        self._kwargs.update( {'file_path': str(file_path), 'mea_pitch':mea_pitch, 'electrode_width':electrode_width})


read_biocam = define_function_from_class(source_class=BiocamRecordingExtractor, name="read_biocam")
