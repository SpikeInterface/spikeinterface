import json

import pytest
import numpy as np

from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.core_tools import SIJsonEncoder
from spikeinterface.core.generate import generate_recording, generate_sorting

from pathlib import Path


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


@pytest.fixture(scope="module")
def dictionary_with_numpy_scalar_keys(numpy_array_integer, numpy_array_float, numpy_array_bool):
    numpy_integer_scalar = numpy_array_integer[0]
    numpy_float_scalar = numpy_array_float[0]
    numpy_boolean_scalar = numpy_array_bool[1]

    dictionary = {
        numpy_integer_scalar: "value_of_numpy_integer_scalar",
        numpy_float_scalar: "value_of_numpy_float_scalar",
        numpy_boolean_scalar: "value_of_boolean_scalar",
    }

    return dictionary


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
            int=numpy_array_integer, float=numpy_array_float, double_nested=dict(extra=numpy_array_integer * 2)
        ),
    )
    json.dumps(nested_dict_with_array_as_values, cls=SIJsonEncoder)


def test_encoding_numpy_scalars_values(numpy_array_integer, numpy_array_float, numpy_array_bool):
    numpy_integer_scalar = numpy_array_integer[0]
    numpy_float_scalar = numpy_array_float[0]
    numpy_boolean_scalar = numpy_array_bool[0]

    dict_with_numpy_scalar_values = dict(
        integer=numpy_integer_scalar, float=numpy_float_scalar, boolean=numpy_boolean_scalar
    )
    json.dumps(dict_with_numpy_scalar_values, cls=SIJsonEncoder)


def test_encoding_numpy_scalars_keys(dictionary_with_numpy_scalar_keys):
    json.dumps(dictionary_with_numpy_scalar_keys, cls=SIJsonEncoder)


def test_encoding_numpy_scalars_keys_nested(numpy_array_integer, numpy_array_float, numpy_array_bool):
    numpy_integer_scalar = numpy_array_integer[0]
    numpy_float_scalar = numpy_array_float[0]
    numpy_boolean_scalar = numpy_array_bool[1]

    dict_with_nested_numpy_scalars = {numpy_integer_scalar: {numpy_float_scalar: {numpy_boolean_scalar: "deep_value"}}}
    json.dumps(dict_with_nested_numpy_scalars, cls=SIJsonEncoder)


def test_encoding_numpy_scalars_keys_nested_mixed(numpy_array_integer, numpy_array_float, numpy_array_bool):
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
        another_numpy_integer_scalar: [{another_numpy_integer_scalar: "deper_value"}, "list_value"],
    }
    json.dumps(dict_with_nested_numpy_scalars, cls=SIJsonEncoder)


def test_numpy_dtype_encoding():
    json.dumps(np.dtype("int32"), cls=SIJsonEncoder)
    json.dumps(np.dtype("float32"), cls=SIJsonEncoder)
    json.dumps(np.dtype("bool"), cls=SIJsonEncoder)


def test_numpy_dtype_alises_encoding():
    # People tend to use this a dtype instead of the proper classes
    json.dumps(np.int32, cls=SIJsonEncoder)
    json.dumps(np.float32, cls=SIJsonEncoder)


def test_path_encoding(tmp_path):

    temporary_path = tmp_path / "a_path_for_this_test"

    json.dumps(temporary_path, cls=SIJsonEncoder)


def test_path_as_annotation(tmp_path):
    temporary_path = tmp_path / "a_path_for_this_test"

    recording = generate_recording()
    recording.annotate(path=temporary_path)

    json.dumps(recording, cls=SIJsonEncoder)


def test_recording_encoding():
    recording = generate_recording()

    json.dumps(recording, cls=SIJsonEncoder)


def test_sorting_encoding(numpy_generated_sorting):
    sorting = numpy_generated_sorting
    json.dumps(sorting, cls=SIJsonEncoder)


class DummyExtractor(BaseExtractor):
    def __init__(self, attribute, other_extractor=None, extractor_list=None, extractor_dict=None):
        self.an_attribute = attribute
        self.other_extractor = other_extractor
        self.extractor_list = extractor_list
        self.extractor_dict = extractor_dict

        BaseExtractor.__init__(self, main_ids=["1", "2"])
        # this already the case by default
        self._serializability["memory"] = True
        self._serializability["json"] = True
        self._serializability["pickle"] = True

        self._kwargs = {
            "attribute": attribute,
            "other_extractor": other_extractor,
            "extractor_list": extractor_list,
            "extractor_dict": extractor_dict,
        }


@pytest.fixture(scope="module")
def nested_extractor(dictionary_with_numpy_scalar_keys):
    inner_extractor = DummyExtractor(attribute=dictionary_with_numpy_scalar_keys)
    extractor = DummyExtractor(attribute="a random attribute", other_extractor=inner_extractor)

    return extractor


@pytest.fixture(scope="module")
def nested_extractor_list(dictionary_with_numpy_scalar_keys):
    inner_extractor1 = DummyExtractor(attribute=dictionary_with_numpy_scalar_keys)
    inner_extractor2 = DummyExtractor(attribute=dictionary_with_numpy_scalar_keys)

    extractor = DummyExtractor(attribute="a random attribute", extractor_list=[inner_extractor1, inner_extractor2])

    return extractor


@pytest.fixture(scope="module")
def nested_extractor_dict(dictionary_with_numpy_scalar_keys):
    inner_extractor1 = DummyExtractor(attribute=dictionary_with_numpy_scalar_keys)
    inner_extractor2 = DummyExtractor(attribute=dictionary_with_numpy_scalar_keys)

    extractor = DummyExtractor(
        attribute="a random attribute",
        extractor_dict=dict(inner_extractor1=inner_extractor1, inner_extractor2=inner_extractor2),
    )

    return extractor


def test_encoding_numpy_scalars_within_nested_extractors(nested_extractor):
    json.dumps(nested_extractor, cls=SIJsonEncoder)


def test_encoding_numpy_scalars_within_nested_extractors_list(nested_extractor_list):
    json.dumps(nested_extractor_list, cls=SIJsonEncoder)


def test_encoding_numpy_scalars_within_nested_extractors_dict(nested_extractor_dict):
    json.dumps(nested_extractor_dict, cls=SIJsonEncoder)


if __name__ == "__main__":
    nested_extractor = nested_extractor()
    test_encoding_numpy_scalars_within_nested_extractors(nested_extractor)
