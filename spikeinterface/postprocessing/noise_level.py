import numpy as np

from spikeinterface.core.waveform_extractor import BaseWaveformExtractorExtension, WaveformExtractor
from spikeinterface.core import get_noise_levels


class NoiseLevelsCalculator(BaseWaveformExtractorExtension):
    extension_name = 'noise_levels'
    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

    def _set_params(self, num_chunks_per_segment=20, 
                           chunk_size=10000, seed=None):
        params = dict(num_chunks_per_segment=num_chunks_per_segment, chunk_size=chunk_size, seed=seed)
        return params

    def _select_extension_data(self, unit_ids):
        # this do not depend on units
        return  self._extension_data
        
    def _run(self):
        return_scaled = self.waveform_extractor.return_scaled
        self._extension_data['noise_levels'] = get_noise_levels(self.waveform_extractor.recording,
                                                                return_scaled=return_scaled, **self._params)
    
    def get_data(self):
        return self._extension_data['noise_levels']

    @staticmethod
    def get_extension_function():
        return compute_noise_levels            


WaveformExtractor.register_extension(NoiseLevelsCalculator)


def compute_noise_levels(waveform_extractor, load_if_exists=False, **params):
    """
    This wrap the 
    `noise_levels = get_noise_levels(recording)`
    into 
    `noise_levels = compute_noise_levels(waveform_extractor)`

    This is done with the BaseWaveformExtractorExtension machinery so the results is persistent on disk (folder or zarr).

    The reults do not depend on the unit list, only the recording, but it is a convinient way to retrieve the noise directly ine the WaveformExtractor.
    
    The result can be scaled or not, this depends if the waveform_extractor itself is scaled or not.

    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object.
    num_chunks_per_segment: int (deulf 20)
        Number of chunks to estimate the noise
    chunk_size: int (default 10000)
        Size of chunks in sample
    seed: int (default None)
        Eventualy a seed for reproducibility.

    Returns
    -------
    noise_levels: np.array
        noise level vector.
    """
    if load_if_exists and waveform_extractor.is_extension(NoiseLevelsCalculator.extension_name):
        ext = waveform_extractor.load_extension(NoiseLevelsCalculator.extension_name)
    else:
        ext = NoiseLevelsCalculator(waveform_extractor)
        ext.set_params(**params)
        ext.run()

    return ext.get_data()



