import pytest

from spikeinterface.core import (
    generate_ground_truth_recording,
    create_sorting_analyzer,
)


def _small_sorting_analyzer():
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
def small_sorting_analyzer():
    return _small_sorting_analyzer()
