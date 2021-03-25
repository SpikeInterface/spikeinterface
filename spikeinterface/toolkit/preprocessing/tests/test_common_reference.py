import unittest
import pytest

from spikeinterface.core.tests.testing_tools import generate_recording

from spikeinterface.toolkit.preprocessing import CommonReferenceRecording, common_reference

import numpy as np

def test_common_reference():
    rec = generate_recording(durations=[5.], num_channels=4)

    
    # no groups
    rec_cmr = common_reference(rec, reference='global', operator='median')
    rec_car = common_reference(rec, reference='global', operator='average')
    rec_sin = common_reference(rec, reference='single', ref_channels=0)
    rec_local_car = common_reference(rec, reference='local', local_radius=(20, 65), operator='median')
    
    rec_cmr.save(verbose=False)
    rec_car.save(verbose=False)
    rec_sin.save(verbose=False)
    rec_local_car.save(verbose=False)

    traces = rec.get_traces()
    assert np.allclose(traces, rec_cmr.get_traces() + np.median(traces, axis=1, keepdims=True), atol=0.01)
    assert np.allclose(traces, rec_car.get_traces() + np.mean(traces, axis=1, keepdims=True), atol=0.01)
    assert not np.all(rec_sin.get_traces()[0])
    assert np.allclose(rec_sin.get_traces()[:, 1], traces[:, 1] - traces[:, 0])
    
    assert np.allclose(traces[:, 0], rec_local_car.get_traces()[:, 0] + np.mean(traces[:, [2, 3]], axis=1),
                       atol=0.01)
    assert np.allclose(traces[:, 1], rec_local_car.get_traces()[:, 1] + np.mean(traces[:, [3]], axis=1),
                       atol=0.01)

    
if __name__ == '__main__':
    test_common_reference()
