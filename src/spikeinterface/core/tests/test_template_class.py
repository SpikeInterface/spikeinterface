from pathlib import Path
import pickle
import json

import numpy as np
import pytest

from spikeinterface.core.template import Templates
from spikeinterface.core.sparsity import ChannelSparsity


@pytest.fixture
def dense_templates():
    num_units = 2
    num_samples = 4
    num_channels = 3
    templates_shape = (num_units, num_samples, num_channels)
    templates_array = np.arange(num_units * num_samples * num_channels).reshape(templates_shape)

    return Templates(templates_array=templates_array)


def test_dense_template_instance(dense_templates):
    templates = dense_templates
    templates_array = templates.templates_array
    num_units, num_samples, num_channels = templates_array.shape

    assert np.array_equal(templates.templates_array, templates_array)
    assert templates.sparsity is None
    assert templates.num_units == num_units
    assert templates.num_samples == num_samples
    assert templates.num_channels == num_channels


def test_numpy_like_behavior(dense_templates):
    templates = dense_templates
    templates_array = templates.templates_array

    # Test that slicing works as in numpy
    assert np.array_equal(templates[:], templates_array[:])
    assert np.array_equal(templates[0], templates_array[0])
    assert np.array_equal(templates[0, :], templates_array[0, :])
    assert np.array_equal(templates[0, :, :], templates_array[0, :, :])
    assert np.array_equal(templates[3:5, :, 2], templates_array[3:5, :, 2])
    # Test fancy indexing
    indices = np.array([0, 1])
    assert np.array_equal(templates[indices], templates_array[indices])
    row_indices = np.array([0, 1])
    col_indices = np.array([2, 3])
    assert np.array_equal(templates[row_indices, col_indices], templates_array[row_indices, col_indices])
    mask = templates_array > 0.5
    assert np.array_equal(templates[mask], templates_array[mask])

    # Test unary ufuncs
    assert np.array_equal(np.sqrt(templates), np.sqrt(templates_array))
    assert np.array_equal(np.abs(templates), np.abs(templates_array))
    assert np.array_equal(np.mean(templates, axis=0), np.mean(templates_array, axis=0))

    # Test binary ufuncs
    other_array = np.random.rand(*templates_array.shape)
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


def test_pickle(dense_templates):
    templates = dense_templates

    # Serialize and deserialize the object
    serialized = pickle.dumps(templates)
    deserialized_templates = pickle.loads(serialized)

    assert np.array_equal(templates.templates_array, deserialized_templates.templates_array)
    assert templates.sparsity == deserialized_templates.sparsity
    assert templates.num_units == deserialized_templates.num_units
    assert templates.num_samples == deserialized_templates.num_samples
    assert templates.num_channels == deserialized_templates.num_channels


def test_jsonification(dense_templates):
    templates = dense_templates
    # Serialize to JSON string
    serialized = templates.to_json()

    # Deserialize back to object
    deserialized = Templates.from_json(serialized)

    # Check if deserialized object matches original
    assert np.array_equal(templates.templates_array, deserialized.templates_array)
    assert templates.sparsity == deserialized.sparsity
    assert templates.num_units == deserialized.num_units
    assert templates.num_samples == deserialized.num_samples
    assert templates.num_channels == deserialized.num_channels
