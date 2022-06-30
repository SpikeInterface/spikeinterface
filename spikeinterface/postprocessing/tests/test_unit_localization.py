import pytest
import shutil
from pathlib import Path

import pytest

from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms
from spikeinterface.extractors import toy_example
from spikeinterface.postprocessing import localize_units


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
    we = WaveformExtractor.load_from_folder(cache_folder / 'toy_waveforms')

    unit_location = localize_units(
        we, method='center_of_mass',  num_channels=4)
    unit_location_dict = localize_units(
        we, method='center_of_mass',  num_channels=4, output='dict')

    # ~ import matplotlib.pyplot as plt
    # ~ from probeinterface.plotting import plot_probe
    # ~ fig, ax = plt.subplots()
    #~ plot_probe(we.recording.get_probe(), ax=ax)
    #~ ax.scatter(unit_location[:, 0], unit_location[:, 1])
    #~ plt.show()


def test_compute_monopolar_triangulation():
    we = WaveformExtractor.load_from_folder(cache_folder / 'toy_waveforms')
    unit_location = localize_units(
        we, method='monopolar_triangulation', radius_um=150)
    unit_location_dict = localize_units(
        we, method='monopolar_triangulation', radius_um=150, output='dict',
        optimizer='least_square')

    unit_location_dict = localize_units(
        we, method='monopolar_triangulation', radius_um=150, output='dict',
        optimizer='minimize_with_log_penality')


    #~ import matplotlib.pyplot as plt
    #~ from probeinterface.plotting import plot_probe
    #~ fig, ax = plt.subplots()
    #~ plot_probe(we.recording.get_probe(), ax=ax)
    #~ ax.scatter(unit_location[:, 0], unit_location[:, 1])
    #~ fig = plt.figure()
    #~ ax = fig.add_subplot(1, 1, 1, projection='3d')
    #~ plot_probe(we.recording.get_probe().to_3d(axes='xy'), ax=ax)
    #~ ax.scatter(unit_location[:, 0], unit_location[:, 1], unit_location[:, 2])
    #~ plt.show()


if __name__ == '__main__':
    setup_module()

    test_compute_unit_center_of_mass()
    test_compute_monopolar_triangulation()
