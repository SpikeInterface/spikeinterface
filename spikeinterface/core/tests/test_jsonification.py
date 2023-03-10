import json

import pytest
import numpy as np

from spikeinterface.core.core_tools import SIJsonEncoder
from spikeinterface.core.generate import generate_recording, generate_sorting
from spikeinterface.preprocessing.common_reference import CommonReferenceRecording
    

@pytest.fixture(scope="module")
def numpy_generated_recording():
    recording = generate_recording()
    return recording

@pytest.fixture(scope="module")
def sorting_generated_recording():
    sorting = generate_sorting()
    return sorting


def test_numpy_encoding():
    numpy_array_integer = np.arange(0, 10, dtype="int32")
    json_dict = json.dumps(numpy_array_integer, cls=SIJsonEncoder)

    numpy_array_float = np.random.rand(10)
    json_dict =  json.dumps(numpy_array_float,cls=SIJsonEncoder)

    numpy_array_bool = numpy_array_float > 0.5
    json_dict = json.dumps(numpy_array_bool, cls=SIJsonEncoder)

    dictionary_with_scalary_arrays = dict(
        integer=numpy_array_integer[0], float=numpy_array_float[0], boolean=numpy_array_bool[0]
    )
    json_dict = json.dumps(dictionary_with_scalary_arrays, cls=SIJsonEncoder)

def test_recording_encoding(numpy_generated_recording):
    recording = numpy_generated_recording
    recording_json = json.dumps(recording, cls=SIJsonEncoder)
    
def test_sorting_encoding(sorting_generated_recording):
    sorting = sorting_generated_recording
    sorting_json = json.dumps(sorting, cls=SIJsonEncoder)
    
def test_pre_processing_encoding(numpy_generated_recording):
    recording = numpy_generated_recording
    common_reference_recording = CommonReferenceRecording(recording=recording)
    post_processing_json = json.dumps(common_reference_recording, cls=SIJsonEncoder)
