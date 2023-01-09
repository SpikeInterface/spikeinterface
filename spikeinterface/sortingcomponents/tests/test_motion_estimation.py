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



def test_estimate_motion():
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    
    peaks = np.load(cache_folder / 'mearec_peaks.npy')
    peak_locations = np.load(cache_folder / 'mearec_peak_locations.npy')



    # test many case and sub case
    all_cases = {
        # new york
        'rigid / decentralized / torch': dict(
            rigid=True,
            method='decentralized',
            conv_engine='torch'
            
        ),
        'rigid / decentralized / numpy': dict(
            rigid=False,
            method='decentralized',
            conv_engine='numpy',
            
            
        ),
        'non-rigid / decentralized / torch': dict(
            rigid=False,
            method='decentralized',
            conv_engine='torch',
        ),
        'non-rigid / decentralized / numpy': dict(
            rigid=False,
            method='decentralized',
            conv_engine='numpy',
        ),
        'non-rigid / decentralized / torch / time_horizon_s': dict(
            rigid=False,
            method='decentralized',
            conv_engine='torch',
            time_horizon_s=15.,
        ),
        'non-rigid / decentralized / torch / gradient_descent': dict(
            rigid=False,
            method='decentralized',
            conv_engine='torch',
            convergence_method='gradient_descent',
        ),


        # kilosort 2.5
        'rigid / iterative_template': dict(
            method='iterative_template',
            rigid=True,
        ),
        'non-rigid / iterative_template': dict(
            method='iterative_template',
            rigid=False,
        ),


    }

    motions = {}
    for name, cases_kwargs in all_cases.items():
        print(name)

        kwargs = dict(direction='y', bin_duration_s=1., bin_um=10., margin_um=5,
                      output_extra_check=True, progress_bar=False, verbose=False)
        kwargs.update(cases_kwargs)

        motion, temporal_bins, spatial_bins, extra_check = estimate_motion(recording, peaks, peak_locations, **kwargs)

        motions[name] = motion

        assert temporal_bins.shape[0] == motion.shape[0]
        assert spatial_bins.shape[0] == motion.shape[1]

        if cases_kwargs['rigid']:
            assert motion.shape[1] == 1
        else:
            assert motion.shape[1] > 1
            

        if DEBUG:
            fig, ax = plt.subplots()
            ax.plot(temporal_bins, motion)

            # motion_histogram = extra_check['motion_histogram']
            # spatial_hist_bins = extra_check['spatial_hist_bin_edges']
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

            plt.show()


    # same params with differents engine should be the same
    motion0, motion1 = motions['rigid / decentralized / torch'], motions['rigid / decentralized / numpy']
    assert (motion0 == motion1).all()

    motion0, motion1 = motions['non-rigid / decentralized / torch'], motions['non-rigid / decentralized / numpy']
    assert (motion0 == motion1).all()


if __name__ == '__main__':
    # setup_module()
    test_estimate_motion()

