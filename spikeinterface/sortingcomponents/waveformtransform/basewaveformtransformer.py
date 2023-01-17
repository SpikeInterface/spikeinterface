from abc import abstractmethod, ABC


class WaveformTransofmer(ABC):
    
    
    def __init__(self):
        self.instantiation_kwargs = dict()
    
    @abstractmethod
    def transform(self, waveforms):
        pass
    
    def fit(self):
        """
        Implement this method if the transformer needs to be fitted (.e.g parameters need to be estimated)
        """
        
        raise NotImplementedError("This class does not implement a fit method.")
    
    def to_dict(self):
        return self.instantiation_kwargs
    
    @classmethod
    def from_dict(cls, instantiation_kwargs):
        return cls(**instantiation_kwargs)
    
    def _to_channelless_representation(self, waveforms):
        """
        Transform waveforms to channelless representation. Collapses the channel dimension leaving only 
        temporal information. 
        """
        num_waveforms, num_time_samples, num_channels = waveforms.shape
        num_channelless_waveforms = num_waveforms * num_channels
        channelless_waveforms = waveforms.swapaxes(1, 2).reshape((num_channelless_waveforms, num_time_samples))        
        
        return channelless_waveforms
    
    def _from_channelless_representation(self, channelless_waveforms, num_channels):
        """
        Transform waveforms from channelless representation. The inverse of to_channelless_representation
        """
        num_channelless_waveforms, num_time_samples = channelless_waveforms.shape
        num_waveforms = num_channelless_waveforms // num_channels

        waveforms = channelless_waveforms.reshape(num_waveforms, num_channels, num_time_samples).swapaxes(2, 1)
        return waveforms