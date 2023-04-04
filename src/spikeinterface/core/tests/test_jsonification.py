import json

import pytest
import numpy as np

from spikeinterface.core.core_tools import SIJsonEncoder
from spikeinterface.core.generate import generate_recording, generate_sorting


@pytest.fixture(scope="module")
def numpy_generated_recording():
    recording = generate_recording()
    return recording


@pytest.fixture(scope="module")
def numpy_generated_sorting():
    sorting = generate_sorting()
    return sorting


@pytest.fixture(scope="module")
def numpy_array_integer():
    return np.arange(0, 3, dtype="int32")


@pytest.fixture(scope="module")
def numpy_array_float():
    return np.ones(3, dtype="float32")


@pytest.fixture(scope="module")
def numpy_array_bool(numpy_array_float):
    return numpy_array_float > 0.5


def test_numpy_array_encoding(numpy_array_integer, numpy_array_float, numpy_array_bool):
    json.dumps(numpy_array_integer, cls=SIJsonEncoder)
    json.dumps(numpy_array_float, cls=SIJsonEncoder)
    json.dumps(numpy_array_bool, cls=SIJsonEncoder)


def test_encoding_dict_with_array_values(numpy_array_integer, numpy_array_float, numpy_array_bool):
    dict_with_array_as_values = dict(int=numpy_array_integer, float=numpy_array_float, boolean=numpy_array_bool)
    json.dumps(dict_with_array_as_values, cls=SIJsonEncoder)


def test_encoding_dict_with_array_values_nested(numpy_array_integer, numpy_array_float, numpy_array_bool):
    nested_dict_with_array_as_values = dict(
        boolean=numpy_array_bool,
        nested=dict(
            int=numpy_array_integer,
            float=numpy_array_float,
            double_nested=dict(extra=numpy_array_integer * 2),
        ),
    )
    json.dumps(nested_dict_with_array_as_values, cls=SIJsonEncoder)


def test_encoding_numpy_scalars_values(numpy_array_integer, numpy_array_float, numpy_array_bool):
    numpy_integer_scalar = numpy_array_integer[0]
    numpy_float_scalar = numpy_array_float[0]
    numpy_boolean_scalar = numpy_array_bool[0]

    dict_with_numpy_scalar_values = dict(
        integer=numpy_integer_scalar,
        float=numpy_float_scalar,
        boolean=numpy_boolean_scalar,
    )
    json.dumps(dict_with_numpy_scalar_values, cls=SIJsonEncoder)


def test_encoding_numpy_scalars_keys(numpy_array_integer, numpy_array_float, numpy_array_bool):
    numpy_integer_scalar = numpy_array_integer[0]
    numpy_float_scalar = numpy_array_float[0]
    numpy_boolean_scalar = numpy_array_bool[1]

    dict_with_numpy_scalar_keys = {
        numpy_integer_scalar: "first_string",
        numpy_float_scalar: "second_string",
        numpy_boolean_scalar: "third_string",
    }
    json.dumps(dict_with_numpy_scalar_keys, cls=SIJsonEncoder)


def test_encoding_numpy_scalars_keys_nested(numpy_array_integer, numpy_array_float, numpy_array_bool):
    numpy_integer_scalar = numpy_array_integer[0]
    numpy_float_scalar = numpy_array_float[0]
    numpy_boolean_scalar = numpy_array_bool[1]

    dict_with_nested_numpy_scalars = {numpy_integer_scalar: {numpy_float_scalar: {numpy_boolean_scalar: "deep_value"}}}
    json.dumps(dict_with_nested_numpy_scalars, cls=SIJsonEncoder)


def test_encoding_numpy_scalars_keys_nestes_mixed(numpy_array_integer, numpy_array_float, numpy_array_bool):
    numpy_integer_scalar = numpy_array_integer[0]
    numpy_float_scalar = numpy_array_float[0]
    numpy_boolean_scalar = numpy_array_bool[1]
    another_numpy_integer_scalar = numpy_array_integer[1]
    another_numpy_float_scalar = numpy_array_float[1]

    dict_with_nested_numpy_scalars = {
        numpy_integer_scalar: {
            another_numpy_float_scalar: False,
            numpy_float_scalar: {numpy_boolean_scalar: "deep_value"},
        },
        another_numpy_integer_scalar: [
            {another_numpy_integer_scalar: "deper_value"},
            "list_value",
        ],
    }
    json.dumps(dict_with_nested_numpy_scalars, cls=SIJsonEncoder)


def test_numpy_dtype_encoding():
    json.dumps(np.dtype("int32"), cls=SIJsonEncoder)
    json.dumps(np.dtype("float32"), cls=SIJsonEncoder)
    json.dumps(np.dtype("bool"), cls=SIJsonEncoder)


def test_numpy_dtype_alises_encoding():
    # People tend to use as a dtype instead of the proper classes
    json.dumps(np.int32, cls=SIJsonEncoder)
    json.dumps(np.float32, cls=SIJsonEncoder)
    json.dumps(np.bool_, cls=SIJsonEncoder)  # Note that np.bool was deperecated in numpy 1.20.0


def test_recording_encoding(numpy_generated_recording):
    recording = numpy_generated_recording
    json.dumps(recording, cls=SIJsonEncoder)


def test_sorting_encoding(numpy_generated_sorting):
    sorting = numpy_generated_sorting
    json.dumps(sorting, cls=SIJsonEncoder)
