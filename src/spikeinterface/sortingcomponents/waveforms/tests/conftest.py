from __future__ import annotations

import pytest

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks


@pytest.fixture(scope="module")
def chunk_executor_kwargs():
    job_kwargs = dict(n_jobs=-1, chunk_size=10000, progress_bar=False)
    return job_kwargs


@pytest.fixture(scope="module")
def generated_recording():
    recording, sorting = generate_ground_truth_recording(
        durations=[10.0],
        sampling_frequency=32000.0,
        num_channels=32,
        num_units=10,
        seed=2205,
    )
    return recording


@pytest.fixture(scope="module")
def detected_peaks(generated_recording, chunk_executor_kwargs):
    recording = generated_recording
    peaks = detect_peaks(recording=recording, **chunk_executor_kwargs)
    return peaks
