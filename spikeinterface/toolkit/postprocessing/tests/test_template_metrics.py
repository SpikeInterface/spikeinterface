import unittest
import shutil
from pathlib import Path

import pytest

from spikeinterface import extract_waveforms, WaveformExtractor
from spikeinterface.extractors import toy_example

from spikeinterface.toolkit.postprocessing import calculate_template_metrics


def setup_module():
    for folder in ('toy_rec', 'toy_sorting', 'toy_waveforms'):
        if Path(folder).is_dir():
            shutil.rmtree(folder)

    recording, sorting = toy_example(num_segments=2, num_units=10)
    recording = recording.save(folder='toy_rec')
    sorting = sorting.save(folder='toy_sorting')

    we = extract_waveforms(recording, sorting, 'toy_waveforms',
                           ms_before=3., ms_after=4., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000)


def test_calculate_template_metrics():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    features = calculate_template_metrics(we)
    print(features)


if __name__ == '__main__':
    # setup_module()

    test_calculate_template_metrics()
