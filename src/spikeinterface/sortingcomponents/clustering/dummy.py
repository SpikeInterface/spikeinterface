from __future__ import annotations

import numpy as np


class DummyClustering:
    """
    Stupid clustering.
    peak are clustered from there channel detection
    So peak['channel_index'] will be the peak_labels
    """

    _default_params = {}

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):
        labels = np.arange(recording.get_num_channels(), dtype="int64")
        peak_labels = peaks["channel_index"]
        return labels, peak_labels
