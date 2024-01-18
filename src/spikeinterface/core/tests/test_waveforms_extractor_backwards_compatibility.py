import pytest
from pathlib import Path

import shutil


from spikeinterface.core import generate_ground_truth_recording

from spikeinterface.core.waveforms_extractor_backwards_compatibility import extract_waveforms as mock_extract_waveforms

# remove this when WaveformsExtractor will be removed
from spikeinterface.core import extract_waveforms as old_extract_waveforms




if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def get_dataset():
    recording, sorting = generate_ground_truth_recording(
        durations=[3600.0], sampling_frequency=16000.0, num_channels=128, num_units=100,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        generate_unit_locations_kwargs=dict(
            margin_um=5.0,
            minimum_z=5.0,
            maximum_z=20.0,
        ),
        generate_templates_kwargs=dict(
            unit_params_range=dict(
                alpha=(9_000.0, 12_000.0),
            )
        ),
        noise_kwargs=dict(noise_level=5.0, strategy="tile_pregenerated"),
        seed=2406,
    )
    return recording, sorting


def test_extract_waveforms():
    recording, sorting = get_dataset()
    print(recording)

    folder = cache_folder / "mock_waveforms_extractor"
    if folder.exists():
        shutil.rmtree(folder)

    we = mock_extract_waveforms(recording, sorting, folder=folder, sparse=True)
    print(we)

    folder = cache_folder / "old_waveforms_extractor"
    if folder.exists():
        shutil.rmtree(folder)

    we = old_extract_waveforms(recording, sorting, folder=folder, sparse=True)
    print(we)


if __name__ == "__main__":
    test_extract_waveforms()
