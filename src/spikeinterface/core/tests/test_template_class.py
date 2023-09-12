from pathlib import Path

import numpy as np
import pytest

from spikeinterface.core.template import Templates


def test_dense_template_instance():
    num_units = 2
    num_samples = 4
    num_channels = 3
    templates_shape = (num_units, num_samples, num_channels)
    templates_array = np.arange(num_units * num_samples * num_channels).reshape(templates_shape)

    templates = Templates(templates_array=templates_array)

    assert np.array_equal(templates.templates_array, templates_array)
    assert templates.sparsity is None
    assert templates.num_units == num_units
    assert templates.num_samples == num_samples
    assert templates.num_channels == num_channels


def test_numpy_like_behavior():
    num_units = 2
    num_samples = 4
    num_channels = 3
    templates_shape = (num_units, num_samples, num_channels)
    templates_array = np.arange(num_units * num_samples * num_channels).reshape(templates_shape)

    templates = Templates(templates_array=templates_array)

    # Test that slicing works as in numpy
    assert np.array_equal(templates[:], templates_array[:])
    assert np.array_equal(templates[0], templates_array[0])
    assert np.array_equal(templates[0, :], templates_array[0, :])
    assert np.array_equal(templates[0, :, :], templates_array[0, :, :])
    assert np.array_equal(templates[3:5, :, 2], templates_array[3:5, :, 2])

    # Test unary ufuncs
    assert np.array_equal(np.sqrt(templates), np.sqrt(templates_array))
    assert np.array_equal(np.abs(templates), np.abs(templates_array))
    assert np.array_equal(np.mean(templates, axis=0), np.mean(templates_array, axis=0))

    # Test binary ufuncs
    other_array = np.random.rand(*templates_shape)
    other_template = Templates(templates_array=other_array)

    assert np.array_equal(np.add(templates, other_template), np.add(templates_array, other_array))
    assert np.array_equal(np.multiply(templates, other_template), np.multiply(templates_array, other_array))

    # Test chaining of operations
    chained_result = np.mean(np.multiply(templates, other_template), axis=0)
    chained_expected = np.mean(np.multiply(templates_array, other_array), axis=0)
    assert np.array_equal(chained_result, chained_expected)

    # Test ufuncs that return non-ndarray results
    assert np.all(np.greater(templates, -1))
    assert not np.any(np.less(templates, 0))
