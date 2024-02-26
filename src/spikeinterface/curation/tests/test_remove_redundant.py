import pytest
import shutil
from pathlib import Path

import pytest

import numpy as np

from spikeinterface import create_sorting_analyzer
from spikeinterface.core.generate import inject_some_duplicate_units


from spikeinterface.curation.tests.common import make_sorting_analyzer, sorting_analyzer_for_curation

from spikeinterface.curation import remove_redundant_units


def test_remove_redundant_units(sorting_analyzer_for_curation):

    sorting = sorting_analyzer_for_curation.sorting
    recording = sorting_analyzer_for_curation.recording

    sorting_with_dup = inject_some_duplicate_units(sorting, ratio=0.8, num=4, seed=2205)
    # print(sorting.unit_ids)
    # print(sorting_with_dup.unit_ids)

    job_kwargs = dict(n_jobs=-1)
    sorting_analyzer = create_sorting_analyzer(sorting_with_dup, recording, format="memory")
    sorting_analyzer.compute("random_spikes")
    sorting_analyzer.compute("waveforms", **job_kwargs)
    sorting_analyzer.compute("templates")

    for remove_strategy in ("max_spikes", "minimum_shift", "highest_amplitude"):
        sorting_clean = remove_redundant_units(sorting_analyzer, remove_strategy=remove_strategy)
        # print(sorting_clean)
        # print(sorting_clean.unit_ids)
        assert np.array_equal(sorting_clean.unit_ids, sorting.unit_ids)


if __name__ == "__main__":
    sorting_analyzer = make_sorting_analyzer(sparse=True)
    test_remove_redundant_units(sorting_analyzer)
