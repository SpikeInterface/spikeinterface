import unittest
import shutil
from pathlib import Path

import pytest

from spikeinterface import extract_waveforms, WaveformExtractor
from spikeinterface.extractors import toy_example

from spikeinterface.postprocessing import (compute_template_metrics, get_template_channel_sparsity, 
                                           TemplateMetricsCalculator)

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"


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
    folder = cache_folder / 'toy_waveforms'
    we = WaveformExtractor.load_from_folder(folder)
    tm = compute_template_metrics(we, upsampling_factor=1)
    print(tm)
    
    # reload as an extension from we
    assert TemplateMetricsCalculator in we.get_available_extensions()
    assert we.is_extension('template_metrics')
    tmc = we.load_extension('template_metrics')
    assert isinstance(tmc, TemplateMetricsCalculator)
    assert 'metrics' in tmc._extension_data
    tmc = TemplateMetricsCalculator.load_from_folder(folder)
    assert 'metrics' in tmc._extension_data

    tm_up = compute_template_metrics(we, upsampling_factor=2)
    print(tm_up)
    
    sparsity = get_template_channel_sparsity(we, method="radius", radius_um=20)
    tm_sparse = compute_template_metrics(we, upsampling_factor=2,
                                                 sparsity=sparsity)
    print(tm_sparse)
    
    # in-memory
    we_mem = extract_waveforms(we.recording, we.sorting, mode="memory")
    metrics = compute_template_metrics(we_mem)

    # reload as an extension from we
    assert TemplateMetricsCalculator in we_mem.get_available_extensions()
    assert we_mem.is_extension('template_metrics')
    tsc = we_mem.load_extension('template_metrics')
    assert isinstance(tsc, TemplateMetricsCalculator)
    assert 'metrics' in tsc._extension_data


if __name__ == '__main__':
    setup_module()

    test_calculate_template_metrics()
