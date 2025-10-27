from __future__ import annotations

import warnings
import numpy as np
from typing import Optional

from probeinterface import Probe


def get_neuropixels_sample_shifts_from_probe(probe: Probe, stream_name: str = "ap") -> np.ndarray:
    """
    Get the inter-sample shifts for Neuropixels probes based on the probe information.

    Parameters
    ----------
    probe : Probe
        The probe object containing channel and ADC information.
    stream_name : str, default: "ap"
        The name of the stream for which to calculate the sample shifts.
        This is used for Neuropixels 1.0 technology to correctly set the number of cycles.

    Returns
    -------
    sample_shifts : np.ndarray
        Array of relative phase shifts for each channel.
    """
    # get inter-sample shifts based on the probe information and ADC sampling pattern
    num_channels_per_adc = probe.annotations.get("num_channels_per_adc", None)
    adc_sample_order = probe.contact_annotations.get("adc_sample_order", None)
    ap_sample_frequency_hz = probe.annotations.get("ap_sample_frequency_hz", None)
    lf_sample_frequency_hz = probe.annotations.get("lf_sample_frequency_hz", None)

    if (
        num_channels_per_adc is None
        or adc_sample_order is None
        or ap_sample_frequency_hz is None
        or lf_sample_frequency_hz is None
    ):
        warning_message = (
            "Unable to find inter-sample shifts in the Neuropixels probe metadata. "
            "The sample shifts will not be loaded. "
        )
        warnings.warn(warning_message, UserWarning, stacklevel=2)
        return None

    # The number of cycles is determined by the number of channels per ADC times 1 + the ratio
    # between the lf and ap sample rate.
    # For NP 1.0, this gives 13 cycles: 12 * (1 + 2500 / 30000) = 13
    # For NP 2.0, the lf sample rate is 0 and so the number of cycles is equal to the number of
    # channels per ADC.
    # see: https://github.com/billkarsh/ProbeTable/issues/3#issuecomment-3438263027
    num_cycles_in_adc = int(num_channels_per_adc * (1 + lf_sample_frequency_hz / ap_sample_frequency_hz))

    # The inter-sample shifts are given by the adc sample order divided by the number of cycles in ADC
    # This makes sure we also handle cases where only a subset of channels are recorded
    # see: https://github.com/SpikeInterface/spikeinterface/issues/4144
    sample_shifts = adc_sample_order / num_cycles_in_adc

    return sample_shifts


def get_neuropixels_sample_shifts(
    num_channels: int = 384, num_channels_per_adc: int = 12, num_cycles: Optional[int] = None
) -> np.ndarray:
    """
    DEPRECATED
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
    warnings.warn(
        "`get_neuropixels_sample_shifts` is deprecated and will be removed in 0.104.0. "
        "Use `get_neuropixels_sample_shifts_from_probe` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    DEPRECATED
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
    warnings.warn(
        "`get_neuropixels_channel_groups` is deprecated and will be removed in 0.104.0. "
        "Use the `adc_group` contact annotation from the `Probe` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    trig_ref = recording_ref.get_traces(channel_ids=[synhcro_chan_id], return_in_uV=False)
    trig_ref = trig_ref[:, 0]
    times_ref = recording_ref.get_times()

    synhcro_chan_id = recording_other.channel_ids[-1]
    trig_other = recording_other.get_traces(channel_ids=[synhcro_chan_id], return_in_uV=False)
    trig_other = trig_other[:, 0]
    times_other = recording_other.get_times()

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(times_ref, trig_ref)
    # ax.plot(times_other, trig_other)
    # plt.show()
