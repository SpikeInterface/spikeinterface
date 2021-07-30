import unittest
import pytest

from spikeinterface.core.tests.testing_tools import generate_recording

from spikeinterface.toolkit.preprocessing import remove_artifacts

import numpy as np


def test_remove_artifacts():
    # one segment only
    rec = generate_recording(durations=[10.])

    triggers = [15000, 30000]
    list_triggers = [triggers]

    ms = 10
    ms_frames = int(ms * rec.get_sampling_frequency() / 1000)

    traces_all_0_clean = rec.get_traces(start_frame=triggers[0] - ms_frames, end_frame=triggers[0] + ms_frames)
    traces_all_1_clean = rec.get_traces(start_frame=triggers[1] - ms_frames, end_frame=triggers[1] + ms_frames)

    rec_rmart = remove_artifacts(rec, triggers, ms_before=10, ms_after=10)
    traces_all_0 = rec_rmart.get_traces(start_frame=triggers[0] - ms_frames, end_frame=triggers[0] + ms_frames)
    traces_short_0 = rec_rmart.get_traces(start_frame=triggers[0] - 10, end_frame=triggers[0] + 10)
    traces_all_1 = rec_rmart.get_traces(start_frame=triggers[1] - ms_frames, end_frame=triggers[1] + ms_frames)
    traces_short_1 = rec_rmart.get_traces(start_frame=triggers[1] - 10, end_frame=triggers[1] + 10)

    assert not np.any(traces_all_0)
    assert not np.any(traces_all_1)
    assert not np.any(traces_short_0)
    assert not np.any(traces_short_1)

    rec_rmart_lin = remove_artifacts(rec, triggers, ms_before=10, ms_after=10, mode="linear")
    traces_all_0 = rec_rmart_lin.get_traces(start_frame=triggers[0] - ms_frames, end_frame=triggers[0] + ms_frames)
    traces_all_1 = rec_rmart_lin.get_traces(start_frame=triggers[1] - ms_frames, end_frame=triggers[1] + ms_frames)
    assert not np.allclose(traces_all_0, traces_all_0_clean)
    assert not np.allclose(traces_all_1, traces_all_1_clean)

    rec_rmart_cub = remove_artifacts(rec, triggers, ms_before=10, ms_after=10, mode="cubic")
    traces_all_0 = rec_rmart_cub.get_traces(start_frame=triggers[0] - ms_frames, end_frame=triggers[0] + ms_frames)
    traces_all_1 = rec_rmart_cub.get_traces(start_frame=triggers[1] - ms_frames, end_frame=triggers[1] + ms_frames)

    assert not np.allclose(traces_all_0, traces_all_0_clean)
    assert not np.allclose(traces_all_1, traces_all_1_clean)


if __name__ == '__main__':
    test_remove_artifacts()
