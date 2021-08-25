from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

from probeinterface import read_spikeglx


class SpikeGLXRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a SpikeGLX system  (NI-DAQ for neuropixel probe)
    See https://billkarsh.github.io/SpikeGLX/    
    
    Based on neo.rawio.SpikeGLXRawIO
    
    Contrary to older verion this reader is folder based.
    So if the folder contain several streams ('imec0.ap' 'nidq' 'imec0.lf')
    then it has to be specified xwith stream_id=
    
    Parameters
    ----------
    folder_path: str
        
    stream_id: str or None
        stream for instance : 'imec0.ap' 'nidq' or 'imec0.lf'
    """
    mode = 'folder'
    NeoRawIOClass = 'SpikeGLXRawIO'

    def __init__(self, folder_path, stream_id=None):
        neo_kwargs = {'dirname': str(folder_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        # open the corresponding stream probe
        #  probe = read_spikeglx(folder_path, in_place=True)
        #  self.set_probegroup(probe)

        self._kwargs = dict(folder_path=str(folder_path), stream_id=stream_id)


def read_spikeglx(*args, **kwargs):
    recording = SpikeGLXRecordingExtractor(*args, **kwargs)
    return recording


read_spikeglx.__doc__ = SpikeGLXRecordingExtractor.__doc__
