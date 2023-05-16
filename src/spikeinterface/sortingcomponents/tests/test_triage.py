import pytest
import numpy as np

from spikeinterface import download_dataset

from spikeinterface.extractors import MEArecRecordingExtractor

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.clustering.triage import nearest_neighor_triage


def test_triage():
    scales = (1, 1, 10)
    threshold_quantile = 60
    x = np.random.normal(loc=-10.0, scale=10.0, size=(1000))
    z = np.random.normal(loc=20.0, scale=5.0, size=(1000))
    maxptps = np.random.uniform(low=1.0, high=2.0, size=(1000))

    idx_keep = nearest_neighor_triage(
        x,
        z,
        maxptps,
        scales=scales,
        threshold=threshold_quantile,
        c=1,
        ptp_weighting=True,
    )

    tx, tz, tmaxptps = x[idx_keep], z[idx_keep], maxptps[idx_keep]

    assert round(tmaxptps.size / maxptps.size, 2) == threshold_quantile / 100
    assert round(tx.size / x.size, 2) == threshold_quantile / 100
    assert round(tz.size / z.size, 2) == threshold_quantile / 100

    idx_keep = nearest_neighor_triage(
        x,
        z,
        maxptps,
        scales=scales,
        threshold=threshold_quantile,
        c=1,
        ptp_weighting=False,
    )

    tx, tz, tmaxptps = x[idx_keep], z[idx_keep], maxptps[idx_keep]

    assert round(tmaxptps.size / maxptps.size, 2) == threshold_quantile / 100
    assert round(tx.size / x.size, 2) == threshold_quantile / 100
    assert round(tz.size / z.size, 2) == threshold_quantile / 100

    # DEBUG
    # import MEArec
    # recgen = MEArec.load_recordings(recordings=local_path, return_h5_objects=True,
    # check_suffix=False,
    # load=['recordings', 'spiketrains', 'channel_positions'],
    # load_waveforms=False)
    # soma_positions = np.zeros((len(recgen.spiketrains), 3), dtype='float32')
    # for i, st in enumerate(recgen.spiketrains):
    #     soma_positions[i, :] = st.annotations['soma_position']
    # import matplotlib.pyplot as plt
    # import spikeinterface.widgets as sw
    # from probeinterface.plotting import plot_probe
    # for title, peak_locations in list_locations:
    #     probe = recording.get_probe()
    #     fig, axs = plt.subplots(ncols=2, sharey=True)
    #     ax = axs[0]
    #     ax.set_title(title)
    #     plot_probe(probe, ax=ax)
    #     ax.scatter(peak_locations['x'], peak_locations['y'], color='k', s=1, alpha=0.5)
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     # MEArec is "yz" in 2D
    #     ax.scatter(soma_positions[:, 1], soma_positions[:, 2], color='g', s=20, marker='*')
    #     if 'z' in peak_locations.dtype.fields:
    #         ax = axs[1]
    #         ax.scatter(peak_locations['z'], peak_locations['y'], color='k', s=1, alpha=0.5)
    #         ax.set_xlabel('z')
    # plt.show()


if __name__ == "__main__":
    test_triage()
