from __future__ import annotations


def to_temporal_representation(waveforms):
    """
    Transform waveforms to temporal representation. Collapses the channel dimension (spatial) leaving only
    temporal information.
    """
    num_waveforms, num_time_samples, num_channels = waveforms.shape
    num_temporal_waveforms = num_waveforms * num_channels
    temporal_waveforms = waveforms.swapaxes(1, 2).reshape((num_temporal_waveforms, num_time_samples))

    return temporal_waveforms


def from_temporal_representation(temporal_waveforms, num_channels):
    """
    Transform waveforms from temporal representation. The inverse of to_temporal_representation
    """
    num_temporal_waveforms, num_time_samples = temporal_waveforms.shape
    num_waveforms = num_temporal_waveforms // num_channels

    waveforms = temporal_waveforms.reshape(num_waveforms, num_channels, num_time_samples).swapaxes(2, 1)
    return waveforms
