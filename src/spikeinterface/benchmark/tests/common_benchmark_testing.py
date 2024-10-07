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
    estimate_templates,
    Templates,
    create_sorting_analyzer,
)
from spikeinterface.generation import generate_drifting_recording

ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))


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

    gt_analyzer = create_sorting_analyzer(gt_sorting, recording, sparse=True, format="memory")
    gt_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    # analyzer.compute("waveforms")
    gt_analyzer.compute("templates")
    gt_analyzer.compute("noise_levels")

    return recording, gt_sorting, gt_analyzer


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
        is_scaled=return_scaled,
    )
    return gt_templates


def make_drifting_dataset():

    static_recording, drifting_recording, sorting, extra_infos = generate_drifting_recording(
        num_units=15,
        duration=125.5,
        sampling_frequency=30000.0,
        probe_name=None,
        generate_probe_kwargs=dict(
            num_columns=3,
            num_contact_per_column=12,
            xpitch=15,
            ypitch=15,
            contact_shapes="square",
            contact_shape_params={"width": 10},
        ),
        generate_unit_locations_kwargs=dict(
            margin_um=20.0,
            minimum_z=5.0,
            maximum_z=40.0,
            minimum_distance=18.0,
            max_iteration=100,
            distance_strict=False,
        ),
        generate_displacement_vector_kwargs=dict(
            displacement_sampling_frequency=5.0,
            drift_start_um=[0, 15],
            drift_stop_um=[0, -15],
            drift_step_um=1,
            motion_list=[
                dict(
                    drift_mode="zigzag",
                    non_rigid_gradient=None,
                    t_start_drift=20.0,
                    t_end_drift=None,
                    period_s=50,
                ),
            ],
        ),
        generate_templates_kwargs=dict(
            ms_before=1.5,
            ms_after=3.0,
            mode="ellipsoid",
            unit_params=dict(
                alpha=(150.0, 500.0),
                spatial_decay=(10, 45),
            ),
        ),
        generate_sorting_kwargs=dict(firing_rates=25.0, refractory_period_ms=4.0),
        generate_noise_kwargs=dict(noise_levels=(12.0, 15.0), spatial_decay=25.0),
        extra_outputs=True,
        seed=None,
    )

    return dict(
        drifting_rec=drifting_recording,
        static_rec=static_recording,
        sorting=sorting,
        displacement_vectors=extra_infos["displacement_vectors"],
        displacement_sampling_frequency=extra_infos["displacement_sampling_frequency"],
        unit_locations=extra_infos["unit_locations"],
        displacement_unit_factor=extra_infos["displacement_unit_factor"],
        unit_displacements=extra_infos["unit_displacements"],
    )
