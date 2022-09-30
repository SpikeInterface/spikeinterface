import pytest
import shutil
from pathlib import Path

import pytest

from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms
from spikeinterface.extractors import toy_example
from spikeinterface.postprocessing import compute_unit_locations, UnitLocationsCalculator


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"


def setup_module():
    for folder_name in ('toy_rec', 'toy_sort', 'toy_waveforms', 'toy_waveforms_1'):
        if (cache_folder / folder_name).is_dir():
            shutil.rmtree(cache_folder / folder_name)

    recording, sorting = toy_example(
        num_segments=2, num_units=10, num_channels=4)
    recording.set_channel_groups([0, 0, 1, 1])
    recording = recording.save(folder=cache_folder / 'toy_rec')
    sorting.set_property("group", [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    sorting = sorting.save(folder=cache_folder / 'toy_sort')

    we = WaveformExtractor.create(
        recording, sorting, cache_folder / 'toy_waveforms')
    we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)
    we.run_extract_waveforms(n_jobs=1, chunk_size=30000)


def test_compute_unit_center_of_mass():
    folder = cache_folder / 'toy_waveforms'
    we = WaveformExtractor.load_from_folder(folder)

    unit_location = compute_unit_locations(
        we, method='center_of_mass',  num_channels=4)
    unit_location_dict = compute_unit_locations(
        we, method='center_of_mass',  num_channels=4, outputs='dict')
    
    # reload as an extension from we
    assert UnitLocationsCalculator in we.get_available_extensions()
    assert we.is_extension('unit_locations')
    ulc = we.load_extension('unit_locations')
    assert isinstance(ulc, UnitLocationsCalculator)
    assert 'unit_locations' in ulc._extension_data
    ulc = UnitLocationsCalculator.load_from_folder(folder)
    assert 'unit_locations' in ulc._extension_data

    # in-memory
    we_mem = extract_waveforms(we.recording, we.sorting, mode="memory")
    locs = compute_unit_locations(we_mem)

    # reload as an extension from we
    assert UnitLocationsCalculator in we_mem.get_available_extensions()
    assert we_mem.is_extension('unit_locations')
    ulc = we_mem.load_extension('unit_locations')
    assert isinstance(ulc, UnitLocationsCalculator)
    assert 'unit_locations' in ulc._extension_data


def test_compute_monopolar_triangulation():
    we = WaveformExtractor.load_from_folder(cache_folder / 'toy_waveforms')
    unit_location = compute_unit_locations(
        we, method='monopolar_triangulation', radius_um=150)
    unit_location_dict = compute_unit_locations(
        we, method='monopolar_triangulation', radius_um=150, outputs='dict',
        optimizer='least_square')

    unit_location_dict = compute_unit_locations(
        we, method='monopolar_triangulation', radius_um=150, outputs='dict',
        optimizer='minimize_with_log_penality')


if __name__ == '__main__':
    setup_module()

    test_compute_unit_center_of_mass()
    test_compute_monopolar_triangulation()
