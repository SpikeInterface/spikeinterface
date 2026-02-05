from __future__ import annotations

import numpy as np

from probeinterface import Probe
from spikeinterface.core import NumpySorting
from spikeinterface.core.generate import (
    generate_sorting,
    generate_channel_locations,
    generate_unit_locations,
    generate_templates,
    generate_ground_truth_recording,
)


def toy_example(
    duration=10,
    num_channels=4,
    num_units=10,
    sampling_frequency=30000.0,
    num_segments=2,
    average_peak_amplitude=-100,
    upsample_factor=None,
    contact_spacing_um=40.0,
    num_columns=1,
    spike_times=None,
    spike_labels=None,
    # score_detection=1,
    firing_rate=3.0,
    seed=None,
):
    """
    Returns a generated dataset with "toy" units and spikes on top on white noise.
    This is useful to test api, algos, postprocessing and visualization without any downloading.

    This a rewrite (with the lazy approach) of the old spikeinterface.extractor.toy_example() which itself was also
    a rewrite from the very old spikeextractor.toy_example() (from Jeremy Magland).
    In this new version, the recording is totally lazy and so it does not use disk space or memory.
    It internally uses NoiseGeneratorRecording + generate_templates + InjectTemplatesRecording.

    For better control, you should use the  `generate_ground_truth_recording()`, but provides better control over
    the parameters.

    Parameters
    ----------
    duration : float or list[float], default: 10
        Duration in seconds. If a list is provided, it will be the duration of each segment.
    num_channels : int, default: 4
        Number of channels
    num_units : int, default: 10
        Number of units
    sampling_frequency : float, default: 30000
        Sampling frequency
    num_segments : int, default: 2
        Number of segments.
    spike_times : np.array or list[nparray] or None, default: None
        Spike time in the recording
    spike_labels : np.array or list[nparray] or None, default: None
        Cluster label for each spike time (needs to specified both together).
    firing_rate : float, default: 3.0
        The firing rate for the units (in Hz)
    seed : int or None, default: None
        Seed for random initialization.
    upsample_factor : None or int, default: None
        An upsampling factor, used only when templates are not provided.
    num_columns : int, default:  1
        Number of columns in probe.
    average_peak_amplitude : float, default: -100
        Average peak amplitude of generated templates.
    contact_spacing_um : float, default: 40.0
        Spacing between probe contacts in micrometers.

    Returns
    -------
    recording : RecordingExtractor
        The output recording extractor.
    sorting : SortingExtractor
        The output sorting extractor.

    """
    if upsample_factor is not None:
        raise NotImplementedError(
            "InjectTemplatesRecording do not support yet upsample_factor but this will be done soon"
        )

    assert num_channels > 0
    assert num_units > 0

    if isinstance(duration, int):
        duration = float(duration)

    if isinstance(duration, float):
        durations = [duration] * num_segments
    else:
        durations = duration
        assert isinstance(duration, list)
        assert len(durations) == num_segments
        assert all(isinstance(d, float) for d in durations)

    unit_ids = np.arange(num_units, dtype="int64")

    # generate probe
    channel_locations = generate_channel_locations(num_channels, num_columns, contact_spacing_um)
    probe = Probe(ndim=2)
    probe.set_contacts(positions=channel_locations, shapes="circle", shape_params={"radius": 5})
    probe.create_auto_shape(probe_type="rect", margin=20.0)
    probe.set_device_channel_indices(np.arange(num_channels, dtype="int64"))

    # generate templates
    # this is hard coded now but it use to be like this
    ms_before = 1.5
    ms_after = 3.0
    unit_locations = generate_unit_locations(
        num_units, channel_locations, margin_um=15.0, minimum_z=5.0, maximum_z=50.0, seed=seed
    )
    templates = generate_templates(
        channel_locations,
        unit_locations,
        sampling_frequency,
        ms_before,
        ms_after,
        upsample_factor=upsample_factor,
        seed=seed,
        dtype="float32",
    )

    if average_peak_amplitude is not None:
        # ajustement au mean amplitude
        amps = np.min(templates, axis=(1, 2))
        templates *= average_peak_amplitude / np.mean(amps)

    # construct sorting
    if spike_times is not None:
        assert isinstance(spike_times, list)
        assert isinstance(spike_labels, list)
        assert len(spike_times) == len(spike_labels)
        assert len(spike_times) == num_segments
        sorting = NumpySorting.from_samples_and_labels(spike_times, spike_labels, sampling_frequency, unit_ids=unit_ids)
    else:
        sorting = generate_sorting(
            num_units=num_units,
            sampling_frequency=sampling_frequency,
            durations=durations,
            firing_rates=firing_rate,
            empty_units=None,
            refractory_period_ms=4.0,
            seed=seed,
        )

    recording, sorting = generate_ground_truth_recording(
        durations=durations,
        sampling_frequency=sampling_frequency,
        sorting=sorting,
        probe=probe,
        templates=templates,
        ms_before=ms_before,
        ms_after=ms_after,
        dtype="float32",
        seed=seed,
        noise_kwargs=dict(noise_levels=10.0, strategy="on_the_fly"),
    )

    return recording, sorting
