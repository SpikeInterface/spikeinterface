from typing import Optional
import numpy as np

from probeinterface import Probe, ProbeGroup
from spikeinterface.core import BaseRecording


def get_neuropixels_sample_shifts(
    num_channels: int = 384, num_channels_per_adc: int = 12, num_cycles: Optional[int] = None
):
    """
    Calculates the relative sampling phase of each channel that results
    from Neuropixels ADC multiplexing.

    This information is needed to perform the preprocessing.phase_shift operation.

    See https://github.com/int-brain-lab/ibllib/blob/master/ibllib/ephys/neuropixel.py


    for the original implementation.

    Parameters
    ----------
    num_channels : int, default: 384
        The total number of channels in a recording.
        All currently available Neuropixels variants have 384 channels.
    num_channels_per_adc : int, default: 12
        The number of channels per ADC on the probe.
        Neuropixels 1.0 probes have 12 ADCs.
        Neuropixels 2.0 probes have 16 ADCs.
    num_cycles: int or None, default: None
        The number of cycles in the ADC on the probe.
        Neuropixels 1.0 probes have 13 cycles for AP and 12 for LFP.
        Neuropixels 2.0 probes have 16 cycles.
        If None, the num_channels_per_adc is used.

    Returns
    -------
    sample_shifts : ndarray
        The relative phase (from 0-1) of each channel
    """
    if num_cycles is None:
        num_cycles = num_channels_per_adc

    adc_indices = np.floor(np.arange(num_channels) / (num_channels_per_adc * 2)) * 2 + np.mod(
        np.arange(num_channels), 2
    )

    sample_shifts = np.zeros_like(adc_indices)

    for a in adc_indices:
        sample_shifts[adc_indices == a] = np.arange(num_channels_per_adc) / num_cycles

    return sample_shifts


def get_neuropixels_channel_groups(num_channels: int = 384, num_adcs: int = 12):
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
        Neuropixels 1.0 probes have 12 ADCs.
        Neuropixels 2.0 probes have 16 ADCs.

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


def synchronize_neuropixel_streams(recording_ref: BaseRecording, recording_other: BaseRecording):
    """
    Use the last "sync" channel from spikeglx or openephys neuropixels to synchronize
    recordings.

    Method used :
      1. detect pulse times on both streams.
      2. make a linear regression from 'other' to 'ref'.
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


def get_neuropixels_probe_group(probe: Probe, load_sync_channel: bool, num_channels_recording: int):
    """
    Constructs a probegroup for a Neuropixels probe, handling the sync channel if necessary.

    Parameters
    ----------
    probe : Probe
        The NP probe
    load_sync_channel : bool
        True if the sync channel should be handles, False otherwise
    num_channels_recording : int
        The number of channels in the recording

    Returns
    -------
    probegroup : ProbeGroup
        The NP probe group
    """
    probegroup = ProbeGroup()
    group_mode = "by_shank" if probe.shank_ids is not None else "by_probe"

    if load_sync_channel:
        # create a dummy probe for the sync channel
        sync_channel_dummy = Probe(ndim=2, si_units="um")
        if group_mode == "by_shank":
            shankd_ids = [-1]
        sync_channel_dummy.set_contacts(positions=[[0, -1000]], shank_ids=shankd_ids)
        sync_channel_dummy.set_device_channel_indices([len(probe.contact_positions)])
        probegroup.add_probe(probe)
        probegroup.add_probe(sync_channel_dummy)
    else:
        probegroup.add_probe(probe)

    if probegroup.get_channel_count() > num_channels_recording:
        exception_msg = f"The probe group loaded exceeds the number of channels in the recording: {probegroup.get_channel_count()} vs {num_channels_recording}."
        if load_sync_channel:
            exception_msg += "\nThe dataset may not contain a SYNC channel. Try setting load_sync_channel=False."
        raise Exception(exception_msg)

    return probegroup
