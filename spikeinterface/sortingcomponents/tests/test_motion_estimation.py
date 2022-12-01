import pytest
from pathlib import Path
import numpy as np

from spikeinterface import download_dataset
from spikeinterface.extractors import MEArecRecordingExtractor

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.motion_estimation import (estimate_motion, make_2d_motion_histogram,
                                                                compute_pairwise_displacement, 
                                                                compute_global_displacement)

from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass

repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
remote_path = 'mearec/mearec_test_10s.h5'


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sortingcomponents"
else:
    cache_folder = Path("cache_folder") / "sortingcomponents"

DEBUG = False

if DEBUG:
    import matplotlib.pyplot as plt
    plt.ion()
    plt.show()


def setup_module():
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    
    cache_folder.mkdir(parents=True, exist_ok=True)

    # detect and localize
    peaks, peak_locations = detect_peaks(recording,
                                         method='locally_exclusive',
                                         peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                                         chunk_size=10000, verbose=1, progress_bar=True,
                                         pipeline_steps = [LocalizeCenterOfMass(recording, ms_before=0.1,
                                                                                ms_after=0.3, local_radius_um=150.)]
                         )
    np.save(cache_folder / 'mearec_peaks.npy', peaks)
    np.save(cache_folder / 'mearec_peak_locations.npy', peak_locations)


def test_motion_functions():
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)

    peaks = np.load(cache_folder / 'mearec_peaks.npy')
    peak_locations = np.load(cache_folder / 'mearec_peak_locations.npy')

    bin_um = 2
    motion_histogram, temporal_bins, spatial_bins = make_2d_motion_histogram(
        recording, peaks, peak_locations, bin_um=bin_um)
    # print(motion_histogram.shape, temporal_bins.size, spatial_bins.size)
    
    # conv + gradient_descent
    pairwise_displacement, pairwise_displacement_weight = compute_pairwise_displacement(
        motion_histogram, bin_um, method='conv')
    motion = compute_global_displacement(pairwise_displacement, convergence_method='gradient_descent')

    # phase_cross_correlation + gradient_descent_robust
    # not tested yet on GH because need skimage
    try:
        import skimage
        pairwise_displacement, pairwise_displacement_weight = compute_pairwise_displacement(
                        motion_histogram, bin_um, method='phase_cross_correlation')
        motion = compute_global_displacement(pairwise_displacement,
                        pairwise_displacement_weight=pairwise_displacement_weight, 
                        convergence_method='lsqr_robust')
    except ImportError:
        print("No skimage, not running phase_cross_correlation test.")

    if DEBUG:
        fig, ax = plt.subplots()
        extent = (temporal_bins[0], temporal_bins[-1], spatial_bins[0], spatial_bins[-1])
        im = ax.imshow(motion_histogram.T, interpolation='nearest',
                            origin='lower', aspect='auto', extent=extent)

        fig, ax = plt.subplots()
        ax.scatter(peaks['sample_ind'] / recording.get_sampling_frequency(), peak_locations['y'], color='r')

        fig, ax = plt.subplots()
        extent = None
        im = ax.imshow(pairwise_displacement, interpolation='nearest',
                            cmap='PiYG', origin='lower', aspect='auto', extent=extent)
        im.set_clim(-40, 40)
        ax.set_aspect('equal')
        fig.colorbar(im)

        fig, ax = plt.subplots()
        ax.plot(temporal_bins[:-1], motion)


def test_estimate_motion_rigid_decentralized():
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    
    peaks = np.load(cache_folder / 'mearec_peaks.npy')
    peak_locations = np.load(cache_folder / 'mearec_peak_locations.npy')

    motions = []
    for conv_engine in ("numpy", "torch"):
        motion, temporal_bins, spatial_bins, extra_check = estimate_motion(recording, peaks, peak_locations,
                                                                           direction='y', bin_duration_s=1., bin_um=10., 
                                                                           margin_um=5,
                                                                           method='decentralized_registration', 
                                                                           method_kwargs=dict(conv_engine=conv_engine),
                                                                           non_rigid_kwargs=None,
                                                                           output_extra_check=True, progress_bar=True, 
                                                                           verbose=True)
        motions.append(motion)

        if DEBUG:
            fig, ax = plt.subplots()
            ax.plot(temporal_bins, motion)

            motion_histogram = extra_check['motion_histogram']
            spatial_hist_bins = extra_check['spatial_hist_bins']
            fig, ax = plt.subplots()
            extent = (temporal_bins[0], temporal_bins[-1], spatial_hist_bins[0], spatial_hist_bins[-1])
            im = ax.imshow(motion_histogram.T, interpolation='nearest',
                                origin='lower', aspect='auto', extent=extent)

            fig, ax = plt.subplots()
            pairwise_displacement = extra_check['pairwise_displacement_list'][0]
            im = ax.imshow(pairwise_displacement, interpolation='nearest',
                                cmap='PiYG', origin='lower', aspect='auto', extent=None)
            im.set_clim(-40, 40)
            ax.set_aspect('equal')
            fig.colorbar(im)

    motion_numpy, motion_torch = motions
    assert (motion_numpy == motion_torch).all()


def test_estimate_motion_rigid_kilosort25():
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    
    peaks = np.load(cache_folder / 'mearec_peaks.npy')
    peak_locations = np.load(cache_folder / 'mearec_peak_locations.npy')

    motion, temporal_bins, spatial_bins, extra_check = estimate_motion(recording, peaks, peak_locations,
                                                                       direction='y', bin_duration_s=1., bin_um=10., 
                                                                       margin_um=5,
                                                                       method='kilosort25', 
                                                                       method_kwargs=dict(),
                                                                       non_rigid_kwargs=None,
                                                                       output_extra_check=True, progress_bar=True, 
                                                                       verbose=True)
    if DEBUG:
        fig, ax = plt.subplots()
        ax.plot(temporal_bins, motion)

        spikecounts_hists = extra_check['spikecounts_hists']
        target_hist = extra_check['target_hist']
        fig, axs = plt.subplots(ncols=len(spikecounts_hists))
        
        for temporal_bin, spikecounts_hist in enumerate(spikecounts_hists):
            ax = axs[temporal_bin]
            im = ax.imshow(spikecounts_hist.T, interpolation='nearest',
                        origin='lower', aspect='auto')
            ax.set_title(f"T{temporal_bin}")
            if temporal_bin > 0:
                ax.set_yticklabels([])
            ax.set_xticklabels([])

        fig, ax = plt.subplots()
        im = ax.imshow(target_hist, interpolation='nearest')
        fig.colorbar(im)
        ax.set_title("Target hist")


def test_estimate_motion_non_rigid_decentralized():
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    
    peaks = np.load(cache_folder / 'mearec_peaks.npy')
    peak_locations = np.load(cache_folder / 'mearec_peak_locations.npy')

    motions = []
    for conv_engine in ("numpy", "torch"):
        motion, temporal_bins, spatial_bins, extra_check = estimate_motion(recording, peaks, peak_locations=peak_locations,
                                                                           direction='y', bin_duration_s=1., bin_um=10., 
                                                                           margin_um=5,
                                                                           method='decentralized_registration', 
                                                                           method_kwargs=dict(conv_engine=conv_engine),
                                                                           non_rigid_kwargs={
                                                                               'bin_step_um': 50},
                                                                           output_extra_check=True, progress_bar=True, 
                                                                           verbose=True)
        motions.append(motion)


        assert spatial_bins is not None
        assert len(spatial_bins) == motion.shape[1]

        if DEBUG:
            probe = recording.get_probe()

            from probeinterface.plotting import plot_probe
            fig, ax = plt.subplots()
            plot_probe(probe, ax=ax)

            non_rigid_windows = extra_check['non_rigid_windows']
            spatial_hist_bins = extra_check['spatial_hist_bins']
            fig, ax = plt.subplots()
            for w, win in enumerate(non_rigid_windows):
                ax.plot(win, spatial_hist_bins[:-1])
                ax.axhline(spatial_bins[w])

            fig, ax = plt.subplots()
            for w, win in enumerate(non_rigid_windows):
                ax.plot(temporal_bins, motion[:, w])

    motion_numpy, motion_torch = motions
    assert (motion_numpy == motion_torch).all()


def test_estimate_motion_non_rigid_kilosort25():
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)

    peaks = np.load(cache_folder / 'mearec_peaks.npy')
    peak_locations = np.load(cache_folder / 'mearec_peak_locations.npy')

    motion, temporal_bins, spatial_bins, extra_check = estimate_motion(recording, peaks, peak_locations,
                                                                       direction='y', bin_duration_s=1., bin_um=10.,
                                                                       margin_um=5,
                                                                       method='kilosort25',
                                                                       method_kwargs=dict(),
                                                                       non_rigid_kwargs={'bin_step_um': 50},
                                                                       output_extra_check=True, progress_bar=True,
                                                                       verbose=True)
    if DEBUG:
        fig, ax = plt.subplots()
        ax.plot(temporal_bins, motion)

        spikecounts_hists = extra_check['spikecounts_hists']
        target_hist = extra_check['target_hist']
        fig, axs = plt.subplots(ncols=len(spikecounts_hists))

        for temporal_bin, spikecounts_hist in enumerate(spikecounts_hists):
            ax = axs[temporal_bin]
            im = ax.imshow(spikecounts_hist.T, interpolation='nearest',
                        origin='lower', aspect='auto')
            ax.set_title(f"T{temporal_bin}")
            if temporal_bin > 0:
                ax.set_yticklabels([])
            ax.set_xticklabels([])

        fig, ax = plt.subplots()
        im = ax.imshow(target_hist, interpolation='nearest')
        fig.colorbar(im)
        ax.set_title("Target hist")
    

if __name__ == '__main__':
    setup_module()
    # test_motion_functions()
    # test_estimate_motion_rigid_decentralized()
    # test_estimate_motion_non_rigid_decentralized()
    test_estimate_motion_rigid_kilosort25()
    test_estimate_motion_non_rigid_kilosort25()
