from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import probeinterface as pi 

import neo

import distutils.version

HAS_NEO_11 = distutils.version.LooseVersion(neo.__version__) >= '0.11'


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
        if HAS_NEO_11:
            neo_kwargs['load_sync_channel'] = False
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        #~ # open the corresponding stream probe
        if HAS_NEO_11 and '.ap' in self.stream_id:
            signals_info_dict = {e['stream_name'] : e for e  in self.neo_reader.signals_info_list}
            meta_filename = signals_info_dict[self.stream_id]['meta_file']
            probe = pi.read_spikeglx(meta_filename)
            self.set_probe(probe, in_place=True)

        self._kwargs = dict(folder_path=str(folder_path), stream_id=stream_id)


def read_spikeglx(*args, **kwargs):
    recording = SpikeGLXRecordingExtractor(*args, **kwargs)
    return recording


read_spikeglx.__doc__ = SpikeGLXRecordingExtractor.__doc__
