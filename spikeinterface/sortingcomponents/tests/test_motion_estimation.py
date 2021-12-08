import pytest
import numpy as np

from spikeinterface import download_dataset
from spikeinterface.extractors import MEArecRecordingExtractor


from spikeinterface.sortingcomponents import detect_peaks
from spikeinterface.sortingcomponents import (estimate_motion, make_motion_histogram,
    compute_pairwise_displacement, compute_global_displacement)


repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
remote_path = 'mearec/mearec_test_10s.h5'


def setup_module():
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    
    # detect and localize
    peaks = detect_peaks(recording,
                         method='locally_exclusive',
                         peak_sign='neg', detect_threshold=5, n_shifts=2,
                         chunk_size=10000, verbose=1, progress_bar=True,
                         localization_dict=dict(method='center_of_mass', local_radius_um=150, ms_before=0.1, ms_after=0.3),
                         #~ localization_dict=dict(method='monopolar_triangulation', local_radius_um=150, ms_before=0.1, ms_after=0.3, max_distance_um=1000),
                         )
    np.save('mearec_detected_peaks.npy', peaks)


def test_motion_functions():
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    
    peaks = np.load('mearec_detected_peaks.npy')
        
    bin_um = 2
    motion_histogram, temporal_bins, spatial_bins = make_motion_histogram(recording, peaks, bin_um=bin_um)
    # print(motion_histogram.shape, temporal_bins.size, spatial_bins.size)

    pairwise_displacement = compute_pairwise_displacement(motion_histogram, bin_um, method='conv2d')

    motion = compute_global_displacement(pairwise_displacement)

    # # DEBUG
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # extent = (temporal_bins[0], temporal_bins[-1], spatial_bins[0], spatial_bins[-1])
    # im = ax.imshow(motion_histogram.T, interpolation='nearest',
    #                     origin='lower', aspect='auto', extent=extent)

    # fig, ax = plt.subplots()
    # ax.scatter(peaks['sample_ind'] / recording.get_sampling_frequency(),peaks['y'], color='r')
    

    # fig, ax = plt.subplots()
    # extent = None
    # im = ax.imshow(pairwise_displacement, interpolation='nearest',
    #                     cmap='PiYG', origin='lower', aspect='auto', extent=extent)
    # im.set_clim(-40, 40)
    # ax.set_aspect('equal')
    # fig.colorbar(im)

    # fig, ax = plt.subplots()
    # ax.plot(temporal_bins[:-1], motion)
    # plt.show()


def test_estimate_motion_rigid():
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    print(recording)
    peaks = np.load('mearec_detected_peaks.npy')

    motion, temporal_bins, spatial_bins, extra_check = estimate_motion(recording, peaks, peak_locations=None,
                    direction='y', bin_duration_s=1., bin_um=10., margin_um=5,
                    method='decentralized_registration', method_kwargs={},
                    non_rigid_kwargs=None,
                    output_extra_check=True, progress_bar=True, verbose=True)
    # print(motion)
    # print(extra_check)
    print(spatial_bins)

    assert spatial_bins is None

    # # DEBUG
    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # ax.plot(temporal_bins[:-1], motion)

    # motion_histogram = extra_check['motion_histogram']
    # spatial_hist_bins = extra_check['spatial_hist_bins']
    # fig, ax = plt.subplots()
    # extent = (temporal_bins[0], temporal_bins[-1], spatial_hist_bins[0], spatial_hist_bins[-1])
    # im = ax.imshow(motion_histogram.T, interpolation='nearest',
    #                     origin='lower', aspect='auto', extent=extent)


    # fig, ax = plt.subplots()
    # pairwise_displacement = extra_check['pairwise_displacement_list'][0]
    # im = ax.imshow(pairwise_displacement, interpolation='nearest',
    #                     cmap='PiYG', origin='lower', aspect='auto', extent=None)
    # im.set_clim(-40, 40)
    # ax.set_aspect('equal')
    # fig.colorbar(im)

    # plt.show()


def test_estimate_motion_non_rigid():
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    print(recording)
    peaks = np.load('mearec_detected_peaks.npy')

    motion, temporal_bins, spatial_bins, extra_check = estimate_motion(recording, peaks, peak_locations=None,
                    direction='y', bin_duration_s=1., bin_um=10., margin_um=5,
                    method='decentralized_registration', method_kwargs={},
                    non_rigid_kwargs={'bin_step_um':50},
                    output_extra_check=True, progress_bar=True, verbose=True)
    # print(motion)
    # print(extra_check.keys())
    # print(spatial_bins)

    assert spatial_bins is not None
    assert len(spatial_bins) == motion.shape[1]

    # # # DEBUG
    # import matplotlib.pyplot as plt
    # probe = recording.get_probe()

    # from probeinterface.plotting import plot_probe
    # fig, ax = plt.subplots()
    # plot_probe(probe, ax=ax)

    # non_rigid_windows = extra_check['non_rigid_windows']
    # spatial_hist_bins = extra_check['spatial_hist_bins']
    # fig, ax = plt.subplots()
    # for w, win in enumerate(non_rigid_windows):
    #     ax.plot(win, spatial_hist_bins[:-1])
    #     ax.axhline(spatial_bins[w])

    # fig, ax = plt.subplots()
    # for w, win in enumerate(non_rigid_windows):
    #     ax.plot(temporal_bins[:-1], motion[:, w])

    # plt.show()


if __name__ == '__main__':
    # setup_module()
    # test_motion_functions()
    # test_estimate_motion_rigid()
    test_estimate_motion_non_rigid()
