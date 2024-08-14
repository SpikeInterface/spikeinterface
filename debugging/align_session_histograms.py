"""
TODO: some notes on this debugging script.
"""
import spikeinterface.full as si
from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion.motion_utils import \
    make_2d_motion_histogram, make_3d_motion_histograms
from scipy.optimize import minimize
from pathlib import Path
import alignment_utils
import pickle

# Generate a ground truth recording where every unit is firing a lot,
# with high amplitude, and is close to the spike, so all picked up.
# This just makes it easy to play around with the units, e.g.
# if specifying 5 units, 5 unit peaks are clearly visible, none are lost
# because their position is too far from probe.

# TODO: what to do about multi shanks? check what motion_correction does

# TODO: most noise in the histogram is from variation in poisson firing rate?
# what other sources of noise are there?
# TODO: robust and rescaling during poisson estimation
# (x, y) alignment!)

# Kilosort method
# Iterate over all
# Align to first, iteration
# robust and non-robust correlation!

# TODO tomorrow:
# 2) Add a robust xcorr
# 3) check the all-0s std case in breakpoint below.


si.set_global_job_kwargs(n_jobs=10)

SEED = 45
np.random.seed(SEED)

if __name__ == "__main__":

    default_unit_params_range = dict(
        alpha=(100.0, 500.0),
        depolarization_ms=(0.09, 0.14),
        repolarization_ms=(0.5, 0.8),
        recovery_ms=(1.0, 1.5),
        positive_amplitude=(0.1, 0.25),
        smooth_ms=(0.03, 0.07),
        spatial_decay=(20, 40),
        propagation_speed=(250.0, 350.0),
        b=(0.1, 1),
        c=(0.1, 1),
        x_angle=(0, np.pi),
        y_angle=(0, np.pi),
        z_angle=(0, np.pi),
    )

    default_unit_params_range["alpha"] = (100, 600)  # do this or change the margin on generate_unit_locations_kwargs
    default_unit_params_range["b"] = (0.5, 1) # and make the units fatter, easier to receive signal!
    default_unit_params_range["c"] = (0.5, 1)

    # TODO: this isn't making any sense, why does removing the highest-frequency
    # unit not kill any of the largest peaks in the histogram, which are not
    # based on amplitude at all?
    num_units = 15

    # TOOD: we should scale the firing rates no the amplitude idiot!?

    recordings_list, _ = generate_session_displacement_recordings(
        non_rigid_gradient=None,  # 0.05, TODO: note this will set nonlinearity to both x and y (the same)
        num_units=num_units,  # 100
        recording_durations=(100, 25, 100, 15, 10),  # TODO: checks on inputs
        recording_shifts=(
            (0, 0), (0, -15), (0, 15), (0, -50), (0, 75)  # TODO: options, to center vs. to specific session
        ),
        recording_amplitude_scalings=None,
#        {
#            "method": "by_amplitude_and_firing_rate",
#            "scalings": (np.r_[1, np.ones(num_units-1)], np.r_[0, 0, 0, np.ones(num_units-3)], np.r_[0, np.ones(num_units-1)], np.r_[0, np.ones(num_units-1)], np.r_[0, np.ones(num_units-1)])
#        },
    #    generate_sorting_kwargs=dict(firing_rates=(50, 100), refractory_period_ms=4.0),
    #    generate_templates_kwargs=dict(unit_params=default_unit_params_range, ms_before=1.5, ms_after=3),
        seed=SEED,
    #    generate_unit_locations_kwargs=dict(
    #        margin_um=0.0,  # if this is say 20, then units go off the edge of the probe and are such low amplitude they are not picked up.
    #        minimum_z=5.0,
    #        maximum_z=45.0,
    #        minimum_distance=18.0,
    #        max_iteration=100,
    #        distance_strict=False,
    #    ),
    )
    bin_um = 2.5  # TODO: pass this below

    PLOT = False # True

    # TODO: project all hist together perform PCA
    # project all hist onto first eig
    # align this!
    # OR, do on the histogram directly like eigenfaces/
    # TODO: perform a phase-shift to align time?

    if False:

        peaks_list = []
        peak_locations_list = []

        for recording in recordings_list:

            peaks, peak_locations = alignment_utils.prep_recording(
                recording, plot=PLOT,
            )
            peaks_list.append(peaks)
            peak_locations_list.append(peak_locations)

        # something relatively easy, only 15 units
        with open('all_recordings.pickle', 'wb') as handle:
            pickle.dump((recordings_list, peaks_list, peak_locations_list),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('all_recordings.pickle', 'rb') as handle:
        recordings_list, peaks_list, peak_locations_list = pickle.load(handle)

    alignment_utils.estimate_session_displacement_benchmarking(
        recordings_list, peaks_list, peak_locations_list, bin_um
    )

    if False:

        corrected_recordings_pois, all_motions_pois = alignment_utils.create_motion_recordings(
            all_recordings, motion_array_pois, all_temporal_bins, non_rigid_window_centers
        )

        corrected_recordings_mean, all_motions_mean = alignment_utils.create_motion_recordings(
            all_recordings, motion_array_mean, all_temporal_bins, non_rigid_window_centers
        )

        alignment_utils.correct_peaks_and_plot_histogram(
            corrected_recordings_pois, all_recordings, all_motions_pois, bin_um
        )

        alignment_utils.correct_peaks_and_plot_histogram(
            corrected_recordings_mean, all_recordings, all_motions_mean, bin_um
        )
