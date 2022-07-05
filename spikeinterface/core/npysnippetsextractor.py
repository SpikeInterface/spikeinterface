from .basesnippets import BaseSnippets
from .core_tools import define_function_from_class
from .numpyextractors import NumpySnippetsSegment
from pathlib import Path
import numpy as np 


class NpySnippetsExtractor(BaseSnippets):
    """
    Dead simple and super light format based on the NPY numpy format.

    It is in fact an archive of several .npy format.
    All spike are store in two columns maner index+labels
    """

    extractor_name = 'NpySnippetsExtractor'
    installed = True  # depend only on numpy
    installation_mesg = "Always installed"
    is_writable = True
    mode = 'file'

    def __init__(self, file_path, sampling_frequency, channel_ids=None, nbefore=None,
                 gain_to_uV=None, offset_to_uV=None):

        self.npy_filename = file_path

        npy = np.load(file_path, mmap_mode='r')
        segments = npy['segment']
        
        BaseSnippets.__init__(self, sampling_frequency,  nbefore=nbefore,
                              snippet_len=npy['snippet'].shape[1],
                              dtype=npy['snippet'].dtype, channel_ids=channel_ids)
        
        num_segments = segments[-1]
        for i in range(num_segments+1):
            snippets = npy['snippet'][segments==i,:,:]
            spikesframes = npy['frame'][segments==i]
            snp_segment = NumpySnippetsSegment(snippets, spikesframes)
            self.add_snippets_segment(snp_segment)

        if gain_to_uV is not None:
            self.set_channel_gains(gain_to_uV)

        if offset_to_uV is not None:
            self.set_channel_offsets(offset_to_uV)
        
        self._kwargs = {'file_path': file_path, 'sampling_frequency': sampling_frequency,
                        'channel_ids': channel_ids, 'gain_to_uV': gain_to_uV, 
                        'offset_to_uV': offset_to_uV }

    @staticmethod
    def write_snippets(snippets, save_path):

        snippets_t = np.dtype([('segment', np.uint16),('frame', np.int64), 
                    ('snippet', snippets.dtype, (snippets.snippet_len, snippets.get_num_channels()))])

        segments = []
        waveforms = []
        frames = []
        for i in range(snippets.get_num_segments()):
            frames.append(snippets.get_frames(segment_index=i))
            segments.append(np.zeros(len(frames[-1]),dtype= np.uint16)+i)
            waveforms.append(snippets.get_snippets(segment_index=i))

        arr = np.empty(sum([len(f) for f in frames]), dtype=snippets_t, order ='F')
        
        arr['segment'] = np.concatenate(segments)
        arr['frame'] = np.concatenate(frames)
        arr['snippet']= np.concatenate(waveforms)
    
        np.save(save_path, arr)


read_npy_snippets = define_function_from_class(source_class=NpySnippetsExtractor, name="read_npy_snippets")
