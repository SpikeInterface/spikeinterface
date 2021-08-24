from .neobaseextractor import NeoBaseRecordingExtractor
import probeinterface as pi


class MaxwellRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from Maxwell device.
    It handle MaxOne (old and new format) and MaxTwo.
    
    
    
    Based on neo.rawio.IntanRawIO
    
    Parameters
    ----------
    file_path: str
        
    stream_id: str or None
        For Maxtwo when there are several well at the same time you 
        need to speficify. stream_id='well000' or 'well0001' or ...
    rec_name: str or None
        When th file contain several block (aka rec) you need to specify the one
        you want.  rec_name='rec0000'
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
        self._kwargs = dict(file_path=str(file_path), stream_id=stream_id, rec_name=rec_name)


def read_maxwell(*args, **kwargs):
    recording = MaxwellRecordingExtractor(*args, **kwargs)
    return recording


read_maxwell.__doc__ = MaxwellRecordingExtractor.__doc__
