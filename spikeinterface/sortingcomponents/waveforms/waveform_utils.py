def to_channelless_representation(waveforms):
    """
    Transform waveforms to channelless representation. Collapses the channel dimension leaving only 
    temporal information. 
    """
    num_waveforms, num_time_samples, num_channels = waveforms.shape
    num_channelless_waveforms = num_waveforms * num_channels
    channelless_waveforms = waveforms.swapaxes(1, 2).reshape((num_channelless_waveforms, num_time_samples))        

    return channelless_waveforms

def from_channelless_representation(channelless_waveforms, num_channels):
    """
    Transform waveforms from channelless representation. The inverse of to_channelless_representation
    """
    num_channelless_waveforms, num_time_samples = channelless_waveforms.shape
    num_waveforms = num_channelless_waveforms // num_channels

    waveforms = channelless_waveforms.reshape(num_waveforms, num_channels, num_time_samples).swapaxes(2, 1)
    return waveforms