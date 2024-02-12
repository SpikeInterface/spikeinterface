import pytest
import shutil
from pathlib import Path

import pytest

import numpy as np

from spikeinterface import start_sorting_result
from spikeinterface.core.generate import inject_some_duplicate_units


from spikeinterface.curation.tests.common import make_sorting_result

from spikeinterface.curation import remove_redundant_units


def test_remove_redundant_units(sorting_result_for_curation):

    sorting = sorting_result_for_curation.sorting
    recording = sorting_result_for_curation.recording

    sorting_with_dup = inject_some_duplicate_units(sorting, ratio=0.8, num=4, seed=2205)
    # print(sorting.unit_ids)
    # print(sorting_with_dup.unit_ids)

    job_kwargs = dict(n_jobs=-1)
    sorting_result = start_sorting_result(sorting_with_dup, recording, format="memory")
    sorting_result.select_random_spikes()
    sorting_result.compute("waveforms", **job_kwargs)
    sorting_result.compute("templates")

    for remove_strategy in ("max_spikes", "minimum_shift", "highest_amplitude"):
        sorting_clean = remove_redundant_units(sorting_result, remove_strategy=remove_strategy)
        # print(sorting_clean)
        # print(sorting_clean.unit_ids)
        assert np.array_equal(sorting_clean.unit_ids, sorting.unit_ids)


if __name__ == "__main__":
    sorting_result = make_sorting_result(sparse=True)
    test_remove_redundant_units(sorting_result)
