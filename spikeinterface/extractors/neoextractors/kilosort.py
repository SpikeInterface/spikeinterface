import numpy as np
from pathlib import Path

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import neo

class KiloSortSortingExtractor(NeoBaseSortingExtractor):
    mode = 'folder'
    NeoRawIOClass = 'PhyRawIO'
    handle_raw_spike_directly = True
    
    def __init__(self, folder_path, use_natural_unit_ids=True):
        neo_kwargs = {'dirname' : folder_path}
        
        # sample rate is in "params.py"
        d = {}
        with open(Path(folder_path) / 'params.py') as f:
            exec(f.read(), None, d)
        sampling_frequency = float(d['sample_rate'])
        
        
        NeoBaseSortingExtractor.__init__(self, 
                    sampling_frequency=sampling_frequency, # auto guess is correct here
                    use_natural_unit_ids=use_natural_unit_ids,
                    **neo_kwargs)