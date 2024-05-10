"""
Important : this benchmark machinery is very heavy.
This is not tested on github because not relevant at all.
This only a local testing.
"""

import pytest
from pathlib import Path
import os

import numpy as np

from spikeinterface.core import (
    generate_ground_truth_recording,
    generate_templates,
    estimate_templates,
    Templates,
    generate_sorting,
    NoiseGeneratorRecording,
)
from spikeinterface.core.generate import generate_unit_locations
from spikeinterface.generation import DriftingTemplates, make_linear_displacement, InjectDriftingTemplatesRecording


from probeinterface import generate_multi_columns_probe


ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sortingcomponents_benchmark"
else:
    cache_folder = Path("cache_folder") / "sortingcomponents_benchmark"


def make_dataset():
    recording, gt_sorting = generate_ground_truth_recording(
        durations=[60.0],
        sampling_frequency=30000.0,
        num_channels=16,
        num_units=10,
        generate_probe_kwargs=dict(
            num_columns=2,
            xpitch=20,
            ypitch=20,
            contact_shapes="circle",
            contact_shape_params={"radius": 6},
        ),
        generate_sorting_kwargs=dict(firing_rates=6.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="on_the_fly"),
        seed=2205,
    )
    return recording, gt_sorting


def compute_gt_templates(recording, gt_sorting, ms_before=2.0, ms_after=3.0, return_scaled=False, **job_kwargs):
    spikes = gt_sorting.to_spike_vector()  # [spike_indices]
    fs = recording.sampling_frequency
    nbefore = int(ms_before * fs / 1000)
    nafter = int(ms_after * fs / 1000)
    templates_array = estimate_templates(
        recording,
        spikes,
        gt_sorting.unit_ids,
        nbefore,
        nafter,
        return_scaled=return_scaled,
        **job_kwargs,
    )

    gt_templates = Templates(
        templates_array=templates_array,
        sampling_frequency=fs,
        nbefore=nbefore,
        sparsity_mask=None,
        channel_ids=recording.channel_ids,
        unit_ids=gt_sorting.unit_ids,
        probe=recording.get_probe(),
    )
    return gt_templates


def make_drifting_dataset():

    num_units = 15
    duration = 125.5
    sampling_frequency = 30000.0
    ms_before = 1.0
    ms_after = 3.0
    displacement_sampling_frequency = 5.0

    probe = generate_multi_columns_probe(
        num_columns=3,
        num_contact_per_column=12,
        xpitch=15,
        ypitch=15,
        contact_shapes="square",
        contact_shape_params={"width": 10},
    )
    probe.set_device_channel_indices(np.arange(probe.contact_ids.size))

    channel_locations = probe.contact_positions

    unit_locations = generate_unit_locations(
        num_units,
        channel_locations,
        margin_um=20.0,
        minimum_z=5.0,
        maximum_z=40.0,
        minimum_distance=20.0,
        max_iteration=100,
        distance_strict=False,
        seed=None,
    )

    nbefore = int(sampling_frequency * ms_before / 1000.0)

    generate_kwargs = dict(
        sampling_frequency=sampling_frequency,
        ms_before=ms_before,
        ms_after=ms_after,
        seed=2205,
        unit_params=dict(
            alpha=(100.0, 500.0),
            depolarization_ms=(0.09, 0.16),
            repolarization_ms=np.ones(num_units) * 0.8,
        ),
    )
    templates_array = generate_templates(channel_locations, unit_locations, **generate_kwargs)

    templates = Templates(
        templates_array=templates_array,
        sampling_frequency=sampling_frequency,
        nbefore=nbefore,
        probe=probe,
    )

    drifting_templates = DriftingTemplates.from_static(templates)
    channel_locations = probe.contact_positions

    start = np.array([0, -15.0])
    stop = np.array([0, 12])
    displacements = make_linear_displacement(start, stop, num_step=29)

    sorting = generate_sorting(
        num_units=num_units,
        sampling_frequency=sampling_frequency,
        durations=[
            duration,
        ],
        firing_rates=25.0,
    )
    sorting

    times = np.arange(0, duration, 1 / displacement_sampling_frequency)
    times

    # 2 rythm
    mid = (start + stop) / 2
    freq0 = 0.1
    displacement_vector0 = np.sin(2 * np.pi * freq0 * times)[:, np.newaxis] * (start - stop) + mid
    # freq1 = 0.01
    # displacement_vector1 = 0.2 * np.sin(2 * np.pi * freq1 *times)[:, np.newaxis] * (start - stop) + mid

    # print()

    displacement_vectors = displacement_vector0[:, :, np.newaxis]

    # TODO gradient
    num_motion = displacement_vectors.shape[2]
    displacement_unit_factor = np.zeros((num_units, num_motion))
    displacement_unit_factor[:, 0] = 1

    drifting_templates.precompute_displacements(displacements)

    direction = 1
    unit_displacements = np.zeros((displacement_vectors.shape[0], num_units))
    for i in range(displacement_vectors.shape[2]):
        m = displacement_vectors[:, direction, i][:, np.newaxis] * displacement_unit_factor[:, i][np.newaxis, :]
        unit_displacements[:, :] += m

    noise = NoiseGeneratorRecording(
        num_channels=probe.contact_ids.size,
        sampling_frequency=sampling_frequency,
        durations=[duration],
        noise_levels=1.0,
        dtype="float32",
    )

    drifting_rec = InjectDriftingTemplatesRecording(
        sorting=sorting,
        parent_recording=noise,
        drifting_templates=drifting_templates,
        displacement_vectors=[displacement_vectors],
        displacement_sampling_frequency=displacement_sampling_frequency,
        displacement_unit_factor=displacement_unit_factor,
        num_samples=[int(duration * sampling_frequency)],
        amplitude_factor=None,
    )

    static_rec = InjectDriftingTemplatesRecording(
        sorting=sorting,
        parent_recording=noise,
        drifting_templates=drifting_templates,
        displacement_vectors=[displacement_vectors],
        displacement_sampling_frequency=displacement_sampling_frequency,
        displacement_unit_factor=np.zeros_like(displacement_unit_factor),
        num_samples=[int(duration * sampling_frequency)],
        amplitude_factor=None,
    )

    my_dict = _variable_from_namespace(
        [
            drifting_rec,
            static_rec,
            sorting,
            displacement_vectors,
            displacement_sampling_frequency,
            unit_locations,
            displacement_unit_factor,
            unit_displacements,
        ],
        locals(),
    )
    return my_dict


def _variable_from_namespace(objs, namespace):
    d = dict()
    for obj in objs:
        for name in namespace:
            if namespace[name] is obj:
                d[name] = obj
    return d
