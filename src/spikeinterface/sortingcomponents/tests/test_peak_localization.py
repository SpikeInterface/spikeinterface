import pytest
import numpy as np

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks

from spikeinterface.sortingcomponents.tests.common import make_dataset


def test_localize_peaks():
    recording, _ = make_dataset()

    # job_kwargs = dict(n_jobs=2, chunk_size=10000, progress_bar=True)
    job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)

    peaks = detect_peaks(
        recording, method="locally_exclusive", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1, **job_kwargs
    )

    list_locations = []

    peak_locations = localize_peaks(recording, peaks, method="center_of_mass", **job_kwargs)
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("com", peak_locations))

    peak_locations = localize_peaks(recording, peaks, method="grid_convolution", **job_kwargs)
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("grid_convolution", peak_locations))

    peak_locations = localize_peaks(
        recording, peaks, method="monopolar_triangulation", optimizer="least_square", **job_kwargs
    )
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("least_square", peak_locations))

    peak_locations = localize_peaks(
        recording, peaks, method="monopolar_triangulation", optimizer="minimize_with_log_penality", **job_kwargs
    )
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("minimize_with_log_penality", peak_locations))

    peak_locations = localize_peaks(
        recording,
        peaks,
        method="monopolar_triangulation",
        optimizer="minimize_with_log_penality",
        enforce_decrease=True,
        **job_kwargs,
    )
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("minimize_with_log_penality", peak_locations))

    peak_locations = localize_peaks(
        recording,
        peaks,
        method="monopolar_triangulation",
        optimizer="minimize_with_log_penality",
        enforce_decrease=True,
        feature="energy",
        **job_kwargs,
    )
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("minimize_with_log_penality_energy", peak_locations))

    peak_locations = localize_peaks(
        recording,
        peaks,
        method="monopolar_triangulation",
        optimizer="minimize_with_log_penality",
        enforce_decrease=True,
        feature="peak_voltage",
        **job_kwargs,
    )
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("minimize_with_log_penality_v_peak", peak_locations))

    peak_locations = localize_peaks(recording, peaks, method="peak_channel", **job_kwargs)
    assert peaks.size == peak_locations.shape[0]
    list_locations.append(("peak_channel", peak_locations))

    # DEBUG
    # import MEArec
    # recgen = MEArec.load_recordings(recordings=local_path, return_h5_objects=True,
    # check_suffix=False,
    # load=['recordings', 'spiketrains', 'channel_positions'],
    # load_waveforms=False)
    # soma_positions = np.zeros((len(recgen.spiketrains), 3), dtype='float32')
    # for i, st in enumerate(recgen.spiketrains):
    # soma_positions[i, :] = st.annotations['soma_position']
    # import matplotlib.pyplot as plt
    # import spikeinterface.widgets as sw
    # from probeinterface.plotting import plot_probe
    # for title, peak_locations in list_locations:
    # probe = recording.get_probe()
    # fig, axs = plt.subplots(ncols=2, sharey=True)
    # ax = axs[0]
    # ax.set_title(title)
    # plot_probe(probe, ax=ax)
    # ax.scatter(peak_locations['x'], peak_locations['y'], color='k', s=1, alpha=0.5)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # #MEArec is "yz" in 2D
    # ax.scatter(soma_positions[:, 1], soma_positions[:, 2], color='g', s=20, marker='*')
    # ax = axs[1]
    # if 'z' in peak_locations.dtype.fields:
    # ax.scatter(peak_locations['z'], peak_locations['y'], color='k', s=1, alpha=0.5)
    # ax.set_xlabel('z')
    # ax.set_title(title)
    # plt.show()


if __name__ == "__main__":
    test_localize_peaks()
