import pytest

import numpy as np

from spikeinterface. preprocessing import get_spatial_interpolation_kernel


def test_get_spatial_interpolation_kernel():
    source_location = np.zeros((10, 2))
    source_location[:, 1] = np.linspace(10, 150, 10)
    
    target_location = np.zeros((4, 2))
    target_location[:, 1] = np.linspace(0, 130, 4)

    for method in ('kriging', 'idw', 'nearest'):
        kernel = get_spatial_interpolation_kernel(source_location, target_location, method=method, force_extrapolate=False)
        # check size
        assert kernel.shape == (source_location.shape[0], target_location.shape[0])
        # check no extrapolate
        assert np.all(kernel[:, 0] == 0)
        # check sum sum close 1
        assert np.allclose(np.sum(kernel[:, 1:], axis=0), 1.)



if __name__ == '__main__':
    test_get_spatial_interpolation_kernel()