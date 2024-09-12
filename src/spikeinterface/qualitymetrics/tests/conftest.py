import pytest

from spikeinterface.core import (
    generate_ground_truth_recording,
    create_sorting_analyzer,
)

job_kwargs = dict(n_jobs=2, progress_bar=True, chunk_duration="1s")


@pytest.fixture(scope="module")
def small_sorting_analyzer():
    recording, sorting = generate_ground_truth_recording(
        durations=[2.0],
        num_units=10,
        seed=1205,
    )

    sorting = sorting.select_units([2, 7, 0], ["#3", "#9", "#4"])

    sorting_analyzer = create_sorting_analyzer(recording=recording, sorting=sorting, format="memory")

    extensions_to_compute = {
        "random_spikes": {"seed": 1205},
        "noise_levels": {"seed": 1205},
        "waveforms": {},
        "templates": {"operators": ["average", "median"]},
        "spike_amplitudes": {},
        "spike_locations": {},
        "principal_components": {},
    }

    sorting_analyzer.compute(extensions_to_compute)

    return sorting_analyzer


@pytest.fixture(scope="module")
def sorting_analyzer_simple():
    # we need high firing rate for amplitude_cutoff
    recording, sorting = generate_ground_truth_recording(
        durations=[
            120.0,
        ],
        sampling_frequency=30_000.0,
        num_channels=6,
        num_units=10,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        generate_unit_locations_kwargs=dict(
            margin_um=5.0,
            minimum_z=5.0,
            maximum_z=20.0,
        ),
        generate_templates_kwargs=dict(
            unit_params=dict(
                alpha=(200.0, 500.0),
            )
        ),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=1205,
    )

    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=True)

    sorting_analyzer.compute("random_spikes", max_spikes_per_unit=300, seed=1205)
    sorting_analyzer.compute("noise_levels")
    sorting_analyzer.compute("waveforms", **job_kwargs)
    sorting_analyzer.compute("templates")
    sorting_analyzer.compute("spike_amplitudes", **job_kwargs)

    return sorting_analyzer
