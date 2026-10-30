import pytest
import numpy as np

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks

from spikeinterface.sortingcomponents.tests.common import make_dataset


def _peaks_and_recording():
    recording, _ = make_dataset()

    peaks = detect_peaks(
        recording,
        method="locally_exclusive",
        method_kwargs=dict(peak_sign="neg", detect_threshold=5, exclude_sweep_ms=1.0),
        job_kwargs=dict(n_jobs=1, chunk_size=10000, progress_bar=True),
    )

    return recording, peaks


@pytest.fixture
def peaks_and_recording():
    return _peaks_and_recording()


def test_localize_peaks(peaks_and_recording):
    recording, peaks = peaks_and_recording

    # job_kwargs = dict(n_jobs=2, chunk_size=10000, progress_bar=True)
    job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)

    list_locations = []

    peak_locations = localize_peaks(recording, peaks, method="center_of_mass", job_kwargs=job_kwargs)
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("com", peak_locations))

    peak_locations = localize_peaks(recording, peaks, method="grid_convolution", job_kwargs=job_kwargs)
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("grid_convolution", peak_locations))

    peak_locations = localize_peaks(
        recording,
        peaks,
        method="monopolar_triangulation",
        method_kwargs=dict(optimizer="least_square"),
        job_kwargs=job_kwargs,
    )
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("least_square", peak_locations))

    peak_locations = localize_peaks(
        recording,
        peaks,
        method="monopolar_triangulation",
        method_kwargs=dict(optimizer="minimize_with_log_penality"),
        job_kwargs=job_kwargs,
    )
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("minimize_with_log_penality", peak_locations))

    peak_locations = localize_peaks(
        recording,
        peaks,
        method="monopolar_triangulation",
        method_kwargs=dict(
            optimizer="minimize_with_log_penality",
            enforce_decrease=True,
        ),
        job_kwargs=job_kwargs,
    )
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("minimize_with_log_penality", peak_locations))

    peak_locations = localize_peaks(
        recording,
        peaks,
        method="monopolar_triangulation",
        method_kwargs=dict(
            optimizer="minimize_with_log_penality",
            enforce_decrease=True,
            feature="energy",
        ),
        job_kwargs=job_kwargs,
    )
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("minimize_with_log_penality_energy", peak_locations))

    peak_locations = localize_peaks(
        recording,
        peaks,
        method_kwargs=dict(
            method="monopolar_triangulation",
            optimizer="minimize_with_log_penality",
            enforce_decrease=True,
            feature="peak_voltage",
        ),
        job_kwargs=job_kwargs,
    )
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("minimize_with_log_penality_v_peak", peak_locations))


@pytest.mark.parametrize("method", ["center_of_mass", "monopolar_triangulation", "grid_convolution"])
def test_localize_peaks_sparse(peaks_and_recording, method):
    recording, peaks = peaks_and_recording

    job_kwargs = dict(n_jobs=2, chunk_size=10000, progress_bar=True)

    # test sparse waveforms
    peak_locations = localize_peaks(
        recording,
        peaks,
        method_kwargs=dict(
            method=method,
        ),
        waveform_method="sparse",  # if method != "grid_convolution" else "dense",
        job_kwargs=job_kwargs,
    )
    assert peaks.size == peak_locations.shape[0]


@pytest.mark.parametrize("method", ["center_of_mass", "monopolar_triangulation", "grid_convolution"])
def test_localize_sparse_narrow(peaks_and_recording, method):
    """Test that a smaller sparsity in waveforms than localization is handled"""
    recording, peaks = peaks_and_recording

    job_kwargs = dict(n_jobs=2, chunk_size=10000, progress_bar=True)

    # test sparse waveforms
    peak_locations = localize_peaks(
        recording,
        peaks,
        method_kwargs=dict(
            method=method,
            radius_um=150,  # larger than waveform radius
        ),
        waveform_method="sparse",  # if method != "grid_convolution" else "dense",
        waveform_kwargs=dict(radius_um=50),  # smaller than localization radius
        job_kwargs=job_kwargs,
    )
    assert peaks.size == peak_locations.shape[0]


@pytest.mark.parametrize("method", ["center_of_mass", "monopolar_triangulation", "grid_convolution"])
def test_sparse_and_dense_are_close(peaks_and_recording, method):
    recording, peaks = peaks_and_recording

    job_kwargs = dict(n_jobs=2, chunk_size=10000, progress_bar=True)

    # test sparse waveforms
    radius_um = 150.0
    peak_locations_sparse = localize_peaks(
        recording,
        peaks,
        method_kwargs=dict(
            method=method,
        ),
        waveform_method="sparse",
        waveform_kwargs=dict(radius_um=radius_um),
        job_kwargs=job_kwargs,
    )
    peak_locations_dense = localize_peaks(
        recording,
        peaks,
        method_kwargs=dict(
            method=method,
        ),
        waveform_method="dense",
        job_kwargs=job_kwargs,
    )
    # Allow a 2um tolerance for the difference between sparse and dense localization results
    np.testing.assert_allclose(peak_locations_sparse["x"], peak_locations_dense["x"], rtol=0.01, atol=1)
    np.testing.assert_allclose(peak_locations_sparse["y"], peak_locations_dense["y"], rtol=0.01, atol=1)
    if "z" in peak_locations_sparse.dtype.names:
        np.testing.assert_allclose(peak_locations_sparse["z"], peak_locations_dense["z"], rtol=0.01, atol=1)


if __name__ == "__main__":
    import pytest

    # run the is close test only for center of mass
    peaks_and_recording_obj = _peaks_and_recording()
    test_sparse_and_dense_are_close(peaks_and_recording_obj, method="center_of_mass")
