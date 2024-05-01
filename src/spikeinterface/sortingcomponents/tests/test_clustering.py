import pytest
import numpy as np

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks, clustering_methods

from spikeinterface.core import get_noise_levels

from spikeinterface.sortingcomponents.tests.common import make_dataset

import time


def job_kwargs():
    return dict(n_jobs=1, chunk_size=10000, progress_bar=True, verbose=True, mp_context="spawn")


@pytest.fixture(name="job_kwargs", scope="module")
def job_kwargs_fixture():
    return job_kwargs()


@pytest.fixture(name="recording", scope="module")
def recording():
    rec, sorting = make_dataset()
    print(rec)
    return rec


def run_peaks(recording, job_kwargs):
    noise_levels = get_noise_levels(recording, return_scaled=False)
    return detect_peaks(
        recording,
        method="locally_exclusive",
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        noise_levels=noise_levels,
        **job_kwargs,
    )


@pytest.fixture(name="peaks", scope="module")
def peaks_fixture(recording, job_kwargs):
    return run_peaks(recording, job_kwargs)


def run_peak_locations(recording, peaks, job_kwargs):
    return localize_peaks(recording, peaks, method="center_of_mass", **job_kwargs)


@pytest.fixture(name="peak_locations", scope="module")
def peak_locations_fixture(recording, peaks, job_kwargs):
    return run_peak_locations(recording, peaks, job_kwargs)


@pytest.mark.parametrize("clustering_method", list(clustering_methods.keys()))
def test_find_cluster_from_peaks(clustering_method, recording, peaks, peak_locations):
    method_kwargs = {}
    if clustering_method in ("position", "position_and_pca"):
        method_kwargs["peak_locations"] = peak_locations
    if clustering_method in ("sliding_hdbscan", "position_and_pca"):
        method_kwargs["waveform_mode"] = "shared_memory"

    t0 = time.perf_counter()
    labels, peak_labels = find_cluster_from_peaks(
        recording, peaks, method=clustering_method, method_kwargs=method_kwargs
    )
    t1 = time.perf_counter()
    print(clustering_method, "found", len(labels), "clusters in ", t1 - t0)


if __name__ == "__main__":
    job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)
    recording, sorting = make_dataset()
    peaks = run_peaks(recording, job_kwargs)
    peak_locations = run_peak_locations(recording, peaks, job_kwargs)
    # method = "position_and_pca"
    method = "circus"

    test_find_cluster_from_peaks(method, recording, peaks, peak_locations)
