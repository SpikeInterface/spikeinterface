import numpy as np

def get_neuropixels_sample_shifts(num_channels = 384, num_adcs = 12):

    """
    Calculates the relative sampling phase of each channel that results 
    from Neuropixels ADC multiplexing.

    This information is needed to perform the preprocessing.phase_shift operation.

    See https://github.com/int-brain-lab/ibllib/blob/master/ibllib/ephys/neuropixel.py
    for the original implementation.

    Params
    ======

    num_channels : The total number of channels in a recording. 
        All currently available Neuropixels variants have 384 channels.

    num_adcs : The total number of ADCs on the probe
        Neuropixels 1.0 probes have 12 ADCs
        Neuropixels 2.0 probes have 16 ADCs


    Returns
    =======
    sample_shifts : The relative phase (from 0-1) of each channel
   
    """

    adc_indices = np.floor(np.arange(num_channels) / (num_adcs * 2)) * 2 + np.mod(np.arange(num_channels), 2)

    sample_shifts = np.zeros_like(adc_indices)

    for a in adc_indices:
        sample_shifts[adc_indices == a] = np.arange(num_adcs) / num_adcs

    return sample_shifts


def get_neuropixels_channel_groups(num_channels = 384, num_adcs = 12):
    
    """
    Returns groups of simultaneously sampled channels on a Neuropixels probe.

    The Neuropixels ADC sampling pattern is as follows:

    Channels:   ADCs:
    |||         |||
    ...         ...
    26 27       2 3
    24 25       2 3 
    22 23       0 1
    ...         ...
    2 3         0 1
    0 1         0 1 <-- even and odd channels are digitized by separate ADCs
    |||         |||
     V           V

    This information is needed to perform the preprocessing.common_reference operation
    on channels that are sampled synchronously.

    Params
    ======

    num_channels : The total number of channels in a recording. 
        All currently available Neuropixels variants have 384 channels.

    num_adcs : The total number of ADCs on the probe
        Neuropixels 1.0 probes have 12 ADCs
        Neuropixels 2.0 probes have 16 ADCs
        

    Returns
    =======
    groups : A list of lists of simultaneously sampled channel indices
   
    """

    groups = []

    for i in range(num_adcs):
        
        groups.append(
            list(
                np.sort(np.concatenate([np.arange(i*2, num_channels, num_adcs*2),
                                        np.arange(i*2+1, num_channels, num_adcs*2)]))
            )
        )
        
    return groups
