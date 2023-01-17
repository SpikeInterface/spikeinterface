
import numpy as np

from .basewaveformtransformer import WaveformTransofmer


class MaxValueNormalization(WaveformTransofmer):
    
            
    def transform(self, waveforms):
        max_value = np.abs(waveforms).max(axis=1)
        return waveforms / max_value[:, None, :]
    