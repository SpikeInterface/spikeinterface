import unittest

import pytest

import numpy as np

from spikeinterface.core.tests.testing_tools import generate_recording


from spikeinterface.toolkit.utils import (get_random_data_for_scaling,
    get_closest_channels, get_noise_levels)


def test_get_random_data_for_scaling():
    rec = generate_recording(num_channels=1, sampling_frequency = 1000., durations = [10., 20.])
    chunks = get_random_data_for_scaling(rec, num_chunks_per_segment=50, chunk_size=500, seed=0)
    assert chunks.shape == (50000, 1)


def test_get_closest_channels():
    rec = generate_recording(num_channels=32, sampling_frequency = 1000., durations = [0.1])
    closest_channels_inds, distances = get_closest_channels(rec)
    closest_channels_inds, distances = get_closest_channels(rec, num_channels=4)
    

def test_get_noise_levels():
    rec = generate_recording(num_channels=2, sampling_frequency = 1000., durations = [60.])
    
    noise_levels = get_noise_levels(rec)
    print(noise_levels)
    
    
if __name__ == '__main__':
    #~ test_get_random_data_for_scaling()
    #~ test_get_closest_channels()
    test_get_noise_levels()
