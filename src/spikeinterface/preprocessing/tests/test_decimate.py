import pytest
from pathlib import Path

import itertools
from spikeinterface import NumpyRecording
from spikeinterface.core import generate_recording
from spikeinterface.preprocessing.decimate import DecimateRecording
import numpy as np


def test_decimate():
    rec = generate_recording()

    N = 101
    rec = NumpyRecording([np.arange(N).reshape(N,1)], 1)
    parent_traces = rec.get_traces()

    for decimation_frame_start, decimation_frame_step in itertools.product(
        [0, 1, 9, 10, 11, 200],
        [1, 9, 10, 11, 200]
    ):

        decimated_rec = DecimateRecording(rec, decimation_frame_step, decimation_frame_start=decimation_frame_start)
        decimated_parent_traces = parent_traces[decimation_frame_start::decimation_frame_step]

        for start_frame, end_frame in itertools.product(
            [0, 1, 5, 20],
            [len(decimated_parent_traces), 1, 5, 20],
        ):
            assert np.all(
                decimated_rec.get_traces(0, start_frame, end_frame) == decimated_parent_traces[start_frame:end_frame]
            )


if __name__ == "__main__":
    test_decimate()
