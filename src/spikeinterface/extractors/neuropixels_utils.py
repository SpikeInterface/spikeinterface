import warnings
import numpy as np

from probeinterface import Probe


def get_neuropixels_sample_shifts_from_probe(probe: Probe) -> np.ndarray:
    """
    Get the inter-sample shifts for Neuropixels probes based on the probe information.

    Parameters
    ----------
    probe : Probe
        The probe object containing channel and ADC information.

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
