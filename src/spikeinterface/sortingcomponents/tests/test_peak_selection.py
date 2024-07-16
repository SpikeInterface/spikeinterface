import pytest
import numpy as np


from spikeinterface.core import get_noise_levels

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks

from spikeinterface.sortingcomponents.tests.common import make_dataset


def test_select_peaks():
    recording, _ = make_dataset()

    # by_channel
    noise_levels = get_noise_levels(recording, return_scaled=False)
    peaks = detect_peaks(
        recording,
        method="by_channel",
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        chunk_size=10000,
        progress_bar=False,
        noise_levels=noise_levels,
    )

    peak_locations = localize_peaks(
        recording, peaks, method="center_of_mass", n_jobs=2, chunk_size=10000, progress_bar=True
    )

    n_peaks = 100
    select_kwargs = dict(n_peaks=n_peaks, noise_levels=noise_levels, peaks_locations=peak_locations)
    select_methods = [
        "uniform",
        "smart_sampling_amplitudes",
        "smart_sampling_locations",
        "smart_sampling_locations_and_time",
    ]
    for method in select_methods:
        selected_peaks = select_peaks(peaks, method=method, **select_kwargs)
        assert (
            selected_peaks.size <= n_peaks
        ), "selected_peaks is not the right size when return_indices=False, select_per_channel=False"

        selected_peaks = select_peaks(peaks, recording=recording, method=method, margin=(10, 10), **select_kwargs)
        assert (
            selected_peaks.size <= n_peaks
        ), "selected_peaks is not the right size when return_indices=False, select_per_channel=False"

        selected_peaks = select_peaks(peaks, method=method, select_per_channel=True, **select_kwargs)
        assert selected_peaks.size <= (
            n_peaks * recording.get_num_channels()
        ), "selected_peaks is not the right size when return_indices=False, select_per_channel=True"

        selected_peaks, selected_indices = select_peaks(peaks, method=method, return_indices=True, **select_kwargs)
        assert (
            selected_peaks.size <= n_peaks
        ), "selected_peaks is not the right size when return_indices=True, select_per_channel=False"
        assert np.all(
            selected_peaks == peaks[selected_indices]
        ), "selection_indices differ from selected_peaks when select_per_channel=False"

        selected_peaks, selected_indices = select_peaks(
            peaks, method=method, return_indices=True, select_per_channel=True, **select_kwargs
        )
        assert selected_peaks.size <= (
            n_peaks * recording.get_num_channels()
        ), "selected_peaks is not the right size when return_indices=True, select_per_channel=True"
        assert np.all(
            selected_peaks == peaks[selected_indices]
        ), "selection_indices differ from selected_peaks when select_per_channel=True"


if __name__ == "__main__":
    test_select_peaks()
