import pytest
import numpy as np

from spikeinterface.extractors import toy_example


def test_toy_example():
    rec, sorting = toy_example(num_segments=2, num_units=10)
    assert rec.get_num_segments() == 2
    assert sorting.get_num_segments() == 2
    assert sorting.get_num_units() == 10
    # print(rec)
    # print(sorting)

    rec, sorting = toy_example(num_segments=1)
    assert rec.get_num_segments() == 1
    assert sorting.get_num_segments() == 1
    print(rec)
    print(sorting)

    # print(rec.get_channel_locations())

    probe = rec.get_probe()
    print(probe)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(rec.get_traces())
    # from spikeinterface.core import extract_waveforms
    # from spikeinterface.widgets import plot_unit_templates
    # we = extract_waveforms(rec, sorting, 'toy_waveforms',
    #                         n_jobs=1, total_memory="10M", max_spikes_per_unit=100,
    #                         return_scaled=False)
    # print(we)
    # plot_unit_templates(we)
    #  plt.show()


if __name__ == '__main__':
    test_toy_example()
