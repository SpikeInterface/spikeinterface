from __future__ import annotations

import numpy as np
from typing import Optional


def get_neuropixels_sample_shifts(
    num_channels: int = 384, num_channels_per_adc: int = 12, num_cycles: Optional[int] = None
) -> np.ndarray:
    """
    Calculate the relative sampling phase (inter-sample shifts) for each channel
    in Neuropixels probes due to ADC multiplexing.

    Neuropixels probes sample channels sequentially through multiple ADCs,
    introducing slight temporal delays between channels within each sampling cycle.
    These inter-sample shifts are fractions of the sampling period and are crucial
    to consider during preprocessing steps, such as phase correction, to ensure
    accurate alignment of the recorded signals.

    This function computes these relative phase shifts, returning an array where
    each value represents the fractional delay (ranging from 0 to 1) for the
    corresponding channel.

    Parameters
    ----------
    num_channels : int, default: 384
        Total number of channels in the recording.
        Neuropixels probes typically have 384 channels.
    num_channels_per_adc : int, default: 12
        Number of channels assigned to each ADC on the probe.
        Neuropixels 1.0 probes have 32 ADCs, each handling 12 channels.
        Neuropixels 2.0 probes have 24 ADCs, each handling 16 channels.
    num_cycles : int or None, default: None
        Number of cycles in the ADC sampling sequence.
        Neuropixels 1.0 probes have 13 cycles for AP (action potential) signals
        and 12 for LFP (local field potential) signals.
        Neuropixels 2.0 probes have 16 cycles.
        If None, defaults to the value of `num_channels_per_adc`.

    Returns
    -------
    sample_shifts : np.ndarray
        Array of relative phase shifts for each channel, with values ranging from 0 to 1,
        representing the fractional delay within the sampling period due to sequential ADC sampling.
    """
    if num_cycles is None:
        num_cycles = num_channels_per_adc

    adc_indices = np.floor(np.arange(num_channels) / (num_channels_per_adc * 2)) * 2 + np.mod(
        np.arange(num_channels), 2
    )

    sample_shifts = np.zeros_like(adc_indices)

    for adc_index in adc_indices:
        sample_shifts[adc_indices == adc_index] = np.arange(num_channels_per_adc) / num_cycles
    return sample_shifts


def get_neuropixels_channel_groups(num_channels=384, num_channels_per_adc=12):
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

    Parameters
    ----------
    num_channels : int, default: 384
        The total number of channels in a recording.
        All currently available Neuropixels variants have 384 channels.
    num_channels_per_adc : int, default: 12
        The number of channels per ADC on the probe.
        Neuropixels 1.0 probes have 32 ADCs, each handling 12 channels.
        Neuropixels 2.0 probes have 24 ADCs, each handling 16 channels.

    Returns
    -------
    groups : list
        A list of lists of simultaneously sampled channel indices
    """

    groups = []

    for i in range(num_channels_per_adc):
        groups.append(
            list(
                np.sort(
                    np.concatenate(
                        [
                            np.arange(i * 2, num_channels, num_channels_per_adc * 2),
                            np.arange(i * 2 + 1, num_channels, num_channels_per_adc * 2),
                        ]
                    )
                )
            )
        )

    return groups


def synchronize_neuropixel_streams(recording_ref, recording_other):
    """
    Use the last "sync" channel from spikeglx or openephys neuropixels to synchronize
    recordings.

    Method used :
      1. detect pulse times on both streams.
      2. make a linear regression from "other" to "ref".
          The slope is nclose to 1 and corresponds to the sample rate correction
          The intercept is close to 0 and corresponds to the delta time start

    """
    # This will be done very very soon, I promise.
    raise NotImplementedError

    synhcro_chan_id = recording_ref.channel_ids[-1]
    trig_ref = recording_ref.get_traces(channel_ids=[synhcro_chan_id], return_scaled=False)
    trig_ref = trig_ref[:, 0]
    times_ref = recording_ref.get_times()

    synhcro_chan_id = recording_other.channel_ids[-1]
    trig_other = recording_other.get_traces(channel_ids=[synhcro_chan_id], return_scaled=False)
    trig_other = trig_other[:, 0]
    times_other = recording_other.get_times()

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(times_ref, trig_ref)
    # ax.plot(times_other, trig_other)
    # plt.show()
