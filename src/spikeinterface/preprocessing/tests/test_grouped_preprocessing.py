from spikeinterface.core import generate_recording
from spikeinterface.preprocessing import common_reference, silence_periods

import numpy as np


def get_some_traces(recording):
    return recording.get_traces(start_frame=10, end_frame=12)


def check_recordings_are_equal(recording_1, recording_2):
    assert np.all(get_some_traces(recording_1) == get_some_traces(recording_2))
    assert recording_1._kwargs == recording_2._kwargs


def test_grouped_preprocessing():
    """Here we make a dict of two recordings and apply preprocessing steps directly
    to the dict. This should give the same result as applying preprocessing steps to
    each recording separately.

    The arg/kwarg logic in `source_class_or_dict_of_sources_classes` is non-trivial,
    so we test some arg/kwarg possibilities here.
    """

    recording_1 = generate_recording(seed=1205, durations=[5])
    recording_2 = generate_recording(seed=1205, durations=[6])

    dict_of_recordings = {"one": recording_1, "two": recording_2}

    # First use dict_of_recordings as an arg
    operator = "average"
    dict_of_preprocessed_recordings = common_reference(dict_of_recordings, operator=operator)

    pp_recording_1 = common_reference(recording_1, operator=operator)
    pp_recording_2 = common_reference(recording_2, operator=operator)

    check_recordings_are_equal(dict_of_preprocessed_recordings["one"], pp_recording_1)
    check_recordings_are_equal(dict_of_preprocessed_recordings["two"], pp_recording_2)

    # Re-try using recording as a kwarg
    dict_of_preprocessed_recordings = common_reference(recording=dict_of_recordings, operator=operator)
    check_recordings_are_equal(dict_of_preprocessed_recordings["one"], pp_recording_1)
    check_recordings_are_equal(dict_of_preprocessed_recordings["two"], pp_recording_2)

    # Now try a `silence periods` which has two args
    list_periods = [[1, 2]]
    mode = "noise"

    sp_recording_1 = silence_periods(recording_1, list_periods=list_periods, mode=mode)
    sp_recording_2 = silence_periods(recording_2, list_periods=list_periods, mode=mode)

    dict_of_preprocessed_recordings = silence_periods(dict_of_recordings, list_periods, mode=mode)
    check_recordings_are_equal(dict_of_preprocessed_recordings["one"], sp_recording_1)
    check_recordings_are_equal(dict_of_preprocessed_recordings["two"], sp_recording_2)

    dict_of_preprocessed_recordings = silence_periods(dict_of_recordings, list_periods=list_periods, mode=mode)
    check_recordings_are_equal(dict_of_preprocessed_recordings["one"], sp_recording_1)
    check_recordings_are_equal(dict_of_preprocessed_recordings["two"], sp_recording_2)

    dict_of_preprocessed_recordings = silence_periods(
        recording=dict_of_recordings, list_periods=list_periods, mode=mode
    )
    check_recordings_are_equal(dict_of_preprocessed_recordings["one"], sp_recording_1)
    check_recordings_are_equal(dict_of_preprocessed_recordings["two"], sp_recording_2)


if __name__ == "__main__":
    test_grouped_preprocessing()
