import unittest
import shutil
from pathlib import Path

import pytest

from spikeinterface import extract_waveforms, WaveformExtractor
from spikeinterface.extractors import toy_example

from spikeinterface.toolkit.postprocessing import calculate_template_metrics, get_template_channel_sparsity

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "toolkit"
else:
    cache_folder = Path("cache_folder") / "toolkit"


def setup_module():
    for folder_name in ('toy_rec', 'toy_sorting', 'toy_waveforms'):
        if (cache_folder / folder_name).is_dir():
            shutil.rmtree(cache_folder / folder_name)

    recording, sorting = toy_example(num_segments=2, num_units=10)
    recording = recording.save(folder=cache_folder / 'toy_rec')
    sorting = sorting.save(folder=cache_folder / 'toy_sorting')

    we = extract_waveforms(recording, sorting, cache_folder / 'toy_waveforms',
                           ms_before=3., ms_after=4., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000)


def test_calculate_template_metrics():
    we = WaveformExtractor.load_from_folder(cache_folder / 'toy_waveforms')
    features = calculate_template_metrics(we, upsampling_factor=1)
    print(features)

    features_up = calculate_template_metrics(we, upsampling_factor=2)
    print(features_up)

    sparsity = get_template_channel_sparsity(we, method="radius", radius_um=20)
    features_sparse = calculate_template_metrics(we, upsampling_factor=2,
                                                 sparsity=sparsity)
    print(features_sparse)


if __name__ == '__main__':
    # setup_module()

    test_calculate_template_metrics()
