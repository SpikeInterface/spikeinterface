import pytest

import spikeinterface as si
import spikeinterface.extractors as se

from spikeinterface.sortingcomponents.peak_detection import detect_peaks


@pytest.fixture(scope="package")
def chunk_executor_kwargs():
    job_kwargs = dict(n_jobs=-1, chunk_size=10000, progress_bar=False)
    return job_kwargs


@pytest.fixture(scope="package")
def mearec_recording():
    local_path = si.download_dataset(remote_path="mearec/mearec_test_10s.h5")
    recording, sorting = se.read_mearec(local_path)
    return recording


@pytest.fixture(scope="package")
def detected_peaks(mearec_recording, chunk_executor_kwargs):
    recording = mearec_recording
    peaks = detect_peaks(recording=recording, **chunk_executor_kwargs)
    return peaks
