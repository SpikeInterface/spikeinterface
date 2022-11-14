import pytest

import numpy as np

from spikeinterface. preprocessing import get_spatial_interpolation_kernel


def test_get_spatial_interpolation_kernel():
    source_location = np.zeros((10, 2))
    source_location[:, 1] = np.linspace(0, 150, 10)
    
    target_location = np.zeros((4, 2))
    target_location[:, 1] = np.linspace(30, 130, 4)

    for method in ('kriging', 'idw'):
        kernel = get_spatial_interpolation_kernel(source_location, target_location, method=method)
        assert kernel.shape == (source_location.shape[0], target_location.shape[0])




if __name__ == '__main__':
    test_get_spatial_interpolation_kernel()