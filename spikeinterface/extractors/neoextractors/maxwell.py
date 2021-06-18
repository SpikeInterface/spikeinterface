import neo

from .neobaseextractor import NeoBaseRecordingExtractor


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
        self._kwargs = dict(file_path=file_path, stream_id=stream_id, rec_name = rec_name)


def read_maxwell(*args, **kargs):
    recording = MaxwellRecordingExtractor(*args, **kargs)
    return recording
read_maxwell.__doc__ = MaxwellRecordingExtractor.__doc__
