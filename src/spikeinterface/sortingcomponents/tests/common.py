from __future__ import annotations

from spikeinterface.core import generate_ground_truth_recording


def make_dataset():
    # this replace the MEArec 10s file for testing
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=30000.0,
        num_channels=32,
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
    return recording, sorting
