
import numpy as np

from .basewaveformtransformer import WaveformTransformer


class MaxValueNormalization(WaveformTransformer):
    
            
    def transform(self, waveforms):
        max_value = np.abs(waveforms).max(axis=1)
        return waveforms / max_value[:, None, :]
    