import unittest
import shutil
from pathlib import Path

import pytest

from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms
from spikeinterface.extractors import toy_example
from spikeinterface.toolkit.postprocessing import localize_template


def setup_module():
    for folder in ('toy_rec', 'toy_sort', 'toy_waveforms', 'toy_waveforms_1'):
        if Path(folder).is_dir():
            shutil.rmtree(folder)

    recording, sorting = toy_example(num_segments=2, num_units=10, num_channels=4)
    recording.set_channel_groups([0, 0, 1, 1])
    recording = recording.save(folder='toy_rec')
    sorting.set_property("group", [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    sorting = sorting.save(folder='toy_sort')

    we = WaveformExtractor.create(recording, sorting, 'toy_waveforms')
    we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)
    we.run_extract_waveforms(n_jobs=1, chunk_size=30000)




def test_compute_unit_center_of_mass():
    we = WaveformExtractor.load_from_folder('toy_waveforms')

    unit_location = localize_template(we, method='center_of_mass',  num_channels=4)
    print(unit_location)


def test_compute_monopolar_triangulation():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    unit_location = localize_template(we, method='monopolar_triangulation', radius_um=150)
    print(unit_location)



if __name__ == '__main__':
    setup_module()

    test_compute_unit_center_of_mass()
    test_compute_monopolar_triangulation()
