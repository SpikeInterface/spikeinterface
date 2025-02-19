"""
test for BaseSnippets are done with NumpySnippets.
but check only for BaseRecording general methods.
"""

import pytest
import numpy as np

from spikeinterface.core import generate_snippets
from spikeinterface.preprocessing.align_snippets import AlignSnippets


def test_AlignSnippets():
    duration = [4, 3]
    num_channels = 3
    nbefore = 20
    nafter = 44

    snippets, _ = generate_snippets(
        durations=duration, num_channels=num_channels, nbefore=nbefore, nafter=nafter, wf_folder=None
    )
    # simple realigment test
    alined_snippets = AlignSnippets(
        snippets, new_nbefore=10, new_nafter=44, mode="main_peak", interpolate=5, det_sign=0
    )
    spikes = alined_snippets.get_snippets(segment_index=0)

    # test interpolation
    snippets_ch_peak = AlignSnippets(
        snippets, new_nbefore=15, new_nafter=30, mode="ch_peak", interpolate=5, det_sign=-1
    )
    spikes = snippets_ch_peak.get_snippets(segment_index=0)

    # test interpolation
    snippets_interpolated = AlignSnippets(
        snippets, new_nbefore=15, new_nafter=20, mode="ch_peak", interpolate=5, det_sign=-1
    )
    spikes = snippets_interpolated.get_snippets(segment_index=0)


if __name__ == "__main__":
    test_AlignSnippets()
