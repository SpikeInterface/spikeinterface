import pytest
import shutil
from pathlib import Path

import pytest

import numpy as np

from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms, NumpySorting
from spikeinterface.core import generate_sorting

from spikeinterface.postprocessing import align_sorting

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"


def test_compute_unit_center_of_mass():
    sorting = generate_sorting(durations=[10.0])
    print(sorting)

    unit_ids = sorting.unit_ids

    unit_peak_shifts = {unit_id: 0 for unit_id in unit_ids}
    unit_peak_shifts[unit_ids[-1]] = 5
    unit_peak_shifts[unit_ids[-2]] = -5

    # sorting to dict
    d = {unit_id: sorting.get_unit_spike_train(unit_id) + unit_peak_shifts[unit_id] for unit_id in sorting.unit_ids}
    sorting_unaligned = NumpySorting.from_unit_dict(d, sampling_frequency=sorting.get_sampling_frequency())
    print(sorting_unaligned)

    sorting_aligned = align_sorting(sorting_unaligned, unit_peak_shifts)
    print(sorting_aligned)

    for start_frame, end_frame in [(None, None), (10000, 50000)]:
        for unit_id in unit_ids[-2:]:
            st = sorting.get_unit_spike_train(unit_id)
            st_clean = sorting_aligned.get_unit_spike_train(unit_id)
            assert np.array_equal(st, st_clean)


if __name__ == "__main__":
    test_compute_unit_center_of_mass()
