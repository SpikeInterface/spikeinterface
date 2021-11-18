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

    """
    mode = 'file'
    NeoRawIOClass = 'BiocamRawIO'

    def __init__(self, file_path, stream_id=None, mea_pitch=None):
        neo_kwargs = {'filename': str(file_path)}
        if mea_pitch is not None:
            neo_kwargs.update({'mea_pitch': mea_pitch})
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        # load locations from neo array annotations
        stream_ann = self.neo_reader.raw_annotations["blocks"][0]["segments"][0]["signals"][self.stream_index]
        locations = stream_ann["__array_annotations__"]["locations"]
        self.set_dummy_probe_from_locations(locations)

        self._kwargs = dict(file_path=str(file_path), stream_id=stream_id, mea_pitch=mea_pitch)


def read_biocam(*args, **kwargs):
    recording = BiocamRecordingExtractor(*args, **kwargs)
    return recording


read_biocam.__doc__ = BiocamRecordingExtractor.__doc__
