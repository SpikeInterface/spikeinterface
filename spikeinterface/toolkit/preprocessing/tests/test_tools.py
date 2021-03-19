

import unittest
import pytest

from spikeinterface.toolkit.preprocessing.tests.testing_tools import generate_recording

from spikeinterface.toolkit.preprocessing import get_chunk_with_margin


def test_get_chunk_with_margin():
    rec = generate_recording(num_channels=1, sampling_frequency = 1000., durations = [10.])
    rec_seg = rec._recording_segments[0]
    length = rec_seg.get_num_samples()
    
    # rec_segment, start_frame, end_frame, channel_indices, sample_margin
    
    traces, l, r = get_chunk_with_margin(rec_seg, None, None, None, 10)
    assert l == 0 and r == 0
    
    traces, l, r = get_chunk_with_margin(rec_seg, 5, None, None, 10)
    assert l == 5 and r == 0
    
    traces, l, r = get_chunk_with_margin(rec_seg, length-1000, length-5, None, 10)
    assert l == 10 and r == 5
    assert traces.shape[0] == 1010

    traces, l, r = get_chunk_with_margin(rec_seg, 2000, 3000, None, 10)
    assert l == 10 and r == 10
    assert traces.shape[0] == 1020




if __name__ == '__main__':
    test_get_chunk_with_margin()
