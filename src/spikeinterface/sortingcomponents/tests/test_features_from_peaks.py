import pytest
import numpy as np

from spikeinterface.sortingcomponents.features_from_peaks import compute_features_from_peaks

from spikeinterface.core import get_noise_levels

from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from spikeinterface.sortingcomponents.tests.common import make_dataset


def test_features_from_peaks():
    recording, sorting = make_dataset()

    job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)

    noise_levels = get_noise_levels(recording, return_scaled=False)

    peaks = detect_peaks(
        recording,
        method="locally_exclusive",
        peak_sign="neg",
        detect_threshold=5,
        noise_levels=noise_levels,
        **job_kwargs,
    )

    feature_list = ["amplitude", "ptp", "center_of_mass", "energy"]
    feature_params = {
        "amplitude": {"all_channels": False, "peak_sign": "neg"},
        "ptp": {"all_channels": False},
        "center_of_mass": {"radius_um": 120.0},
        "energy": {"radius_um": 160.0},
    }
    features = compute_features_from_peaks(recording, peaks, feature_list, feature_params=feature_params, **job_kwargs)

    assert isinstance(features, tuple)

    assert len(features) == len(feature_list)

    # all features have the same shape[0] as peaks
    for one_feature in features:
        assert one_feature.shape[0] == peaks.shape[0]

    # split feature variable
    job_kwargs["n_jobs"] = 2
    amplitude, ptp, com, energy = compute_features_from_peaks(
        recording, peaks, feature_list, feature_params=feature_params, **job_kwargs
    )
    assert amplitude.ndim == 1  # because all_channels=False
    assert ptp.ndim == 1  # because all_channels=False
    assert com.ndim == 1
    assert "x" in com.dtype.fields
    assert energy.ndim == 1

    # amplitude and peak to peak with multi channels
    d = {"all_channels": True}
    (
        amplitude,
        ptp,
    ) = compute_features_from_peaks(
        recording, peaks, ["amplitude", "ptp"], feature_params={"amplitude": d, "ptp": d}, **job_kwargs
    )
    assert amplitude.shape[0] == amplitude.shape[0]
    assert amplitude.shape[1] == recording.get_num_channels()
    assert ptp.shape[0] == peaks.shape[0]
    assert ptp.shape[1] == recording.get_num_channels()


if __name__ == "__main__":
    test_features_from_peaks()
