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

# Generate a ground truth recording where every unit is firing a lot,
# with high amplitude, and is close to the spike, so all picked up.
# This just makes it easy to play around with the units, e.g.
# if specifying 5 units, 5 unit peaks are clearly visible, none are lost
# because their position is too far from probe.

# TODO: what to do about multi shanks? check what motion_correction does

si.set_global_job_kwargs(n_jobs=10
                         )
if __name__ == "__main__":

    if False:
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

        rec_list, _ = generate_session_displacement_recordings(
            non_rigid_gradient=None,  # 0.05, TODO: note this will set nonlinearity to both x and y (the same)
            num_units=100,
            recording_durations=(50,),  # TODO: checks on inputs
            recording_shifts=(
                (0, 0),
            ),
            recording_amplitude_scalings=None,
        #    generate_sorting_kwargs=dict(firing_rates=(50, 100), refractory_period_ms=4.0),
        #    generate_templates_kwargs=dict(unit_params=default_unit_params_range, ms_before=1.5, ms_after=3),
            seed=None,
        #    generate_unit_locations_kwargs=dict(
        #        margin_um=0.0,  # if this is say 20, then units go off the edge of the probe and are such low amplitude they are not picked up.
        #        minimum_z=5.0,
        #        maximum_z=45.0,
        #        minimum_distance=18.0,
        #        max_iteration=100,
        #        distance_strict=False,
        #    ),
        )

    # if False:
    _, drift_rec, _ = si.generate_drifting_recording(num_units=123, duration=600)

    recording, motion_info = si.correct_motion(drift_rec, preset="kilosort_like", output_motion_info=True)

    from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks

    peaks = motion_info["peaks"]
    peak_locations = correct_motion_on_peaks(
        peaks,
        motion_info["peak_locations"],
        motion_info["motion"],
        recording,
    )

    si.plot_drift_raster_map(
        peaks=peaks,
        peak_locations=peak_locations,
        recording=recording,
        clim=(-300, 0)  # fix clim for comparability across plots
    )
    plt.show()


    if False:
        recording = rec_list[0]

        peaks = detect_peaks(recording, method="locally_exclusive")
        peak_locations = localize_peaks(recording, peaks, method="grid_convolution")

        si.plot_drift_raster_map(
            peaks=peaks,
            peak_locations=peak_locations,
            recording=recording,
            clim=(-300, 0)  # fix clim for comparability across plots
        )
        plt.show()


    if False:
        base_path = Path(r"X:\neuroinformatics\scratch\jziminski\ephys\inter-session-alignment\histogram_estimation")
        peaks = np.load(base_path / "peaks_corrected.npy")
        peak_locations = np.load(base_path / "peak_locations_corrected.npy")
        recording = si.load_extractor(base_path / "recording")

    est_dict = alignment_utils.get_all_hist_estimation(
        recording, peaks, peak_locations, bin_um=15,
    )

    alignment_utils.plot_all_hist_estimation(
        est_dict["chunked_session_hist"], est_dict["chunked_spatial_bins"],
    )

    alignment_utils.plot_chunked_session_hist(
        est_dict
    )

    if False:
        # TODO: to test, get a real recording, interpolate each recording
        # one up, one down a small amount.

        # -----------------------------------------------------------------------------
        # Over Entire Session
        # -----------------------------------------------------------------------------

        # TODO: figure out dynamic bin sizing based on 1-um histogram.
        #
        bin_um = 25  # TODO: maybe do some testing on benchmarks backed by some theory.
        WEIGHT_WITH_AMPLITUDE = False

        print("starting make hist")
        entire_session_hist, temporal_bin_edges, spatial_bin_edges = make_2d_motion_histogram(
                        recording,
                        peaks,
                        peak_locations,
                        weight_with_amplitude=WEIGHT_WITH_AMPLITUDE,
                        direction="y",
                        bin_s=recording.get_duration(segment_index=0),  # 1.0,
                        bin_um=bin_um,
                        hist_margin_um=50,
                        spatial_bin_edges=None,
                    )
        entire_session_hist = entire_session_hist[0]
        raw_entire_session_hist = entire_session_hist.copy()
        entire_session_hist /= np.max(entire_session_hist)
        centers = (spatial_bin_edges[1:] + spatial_bin_edges[:-1]) / 2
        plt.plot(centers, entire_session_hist)
        plt.show()

        # -----------------------------------------------------------------------------
        # CHUNKING!
        # -----------------------------------------------------------------------------

        # Estimating the temporal bin size from firing rates
        # based on law of large numbers for Poisson distribution
        # E[X] -> λ, STD = \sqrt{\dfrac{t \lambda}{n}}
        # maybe this is too conservative... 95% of chunks within 10% of lambda theoretically
        def estimate_chunk_size(firing_rate):
            l_pois = firing_rate
            l_exp = 1 / l_pois
            perc = 0.1
            s = (l_exp * perc) / 2  # 99% of values (this is very conservative)
            n = 1 / (s ** 2 / l_exp ** 2)
            t = n / l_pois
            return t, n

        # spikes per second
        raw_entire_session_hist /= recording.get_duration(segment_index=0)

        est_lambda = np.percentile(raw_entire_session_hist, 85)

        est_t, _ = estimate_chunk_size(est_lambda)
        print(est_t)
        chunk_session_hist, temporal_bin_edges, spatial_bin_edges = make_2d_motion_histogram(
                        recording,
                        peaks,
                        peak_locations,
                        weight_with_amplitude=WEIGHT_WITH_AMPLITUDE,
                        direction="y",
                        bin_s=est_t,  # Now make 25 histograms
                        bin_um=bin_um,
                        hist_margin_um=50,
                        spatial_bin_edges=None,
                    )

        m = chunk_session_hist.shape[0]  # TODO: n_hist
        n = chunk_session_hist.shape[1]  # TOOD: n_bin


        # TODO: try trimmed mean / median  and trimming bins for Poisson
        # TODO: try excluding entire histogram based on `session_std`.

        # TODO: could treat the std per bin as a vector and take norm (rather than avg here).
        session_std = np.sum(np.std(chunk_session_hist, axis=0)) / n
        print("Histogram STD:: ", session_std)

        # TODOs estimation
        # Try with amplitude scaling
        # Try with 3d histogram (same idea)
        # Try with (x, y) probe positions (note this will have to be per shank? no I dont think so)
        # TODO: exclude histograms at the level of entire histogram or bin
        # TODO: how to handle this multidimensional std. think more.
        # TODO: for now scale by max for interpretability
        # TODO: think about shank!
        # TODO: do LFP
        # TODO: handle nonlinear!

        # I think for now:
        # 1) think about dynamic bin sizing, formalisation
        # 2) figure out dynamic chunking, based on firing rates.
        # 2) Think about (robust) optimisation
        # 3) Do a minimal working version

        # 4) do some reading, try above approaches

        spatial_centers = (spatial_bin_edges[1:] + spatial_bin_edges[:-1]) / 2  # TODO: own function!
        temporal_centers = (temporal_bin_edges[1:] + temporal_bin_edges[:-1]) / 2

        for i in range(chunk_session_hist.shape[0]):
            plt.plot(spatial_centers, chunk_session_hist[i, :])
        plt.show()

        # -----------------------------------------------------------------------------
        # Sup of Chunks
        # -----------------------------------------------------------------------------

        max_hist = np.max(chunk_session_hist, axis=0)
        max_hist /= np.max(max_hist)

        # -----------------------------------------------------------------------------
        # Mean of Chunks
        # -----------------------------------------------------------------------------

        mean_hist = np.mean(chunk_session_hist, axis=0)
        mean_hist /= np.max(mean_hist)

        # -----------------------------------------------------------------------------
        # Median of Chunks
        # -----------------------------------------------------------------------------

        median_hist = np.median(chunk_session_hist, axis=0)  # interesting, this is probably dumb
        median_hist /= np.max(median_hist)

        # -----------------------------------------------------------------------------
        # Eigenvectors of Chunks
        # -----------------------------------------------------------------------------

        A = chunk_session_hist
        S = A.T @ A  # (num hist, num_bins)

        U,S, Vh = np.linalg.svd(S)

        # TODO: check why this is flipped
        first_eigenvalue = U[:, 0] * -1  # TODO: revise a little + consider another distance metric
        first_eigenvalue /= np.max(first_eigenvalue)

        # -----------------------------------------------------------------------------
        # Poisson Modelling
        # -----------------------------------------------------------------------------
        # Under assumption of independent bins and time points

        def obj_fun(lambda_, m, sum_k ):
            return -(sum_k * np.log(lambda_) - m * lambda_)

        poisson_estimate = np.zeros(chunk_session_hist.shape[1])  # TODO: var names
        for i in range(chunk_session_hist.shape[1]):

            ks = chunk_session_hist[:, i]

            m = ks.shape
            sum_k = np.sum(ks)

            # lol, this is painfully close to the mean...
            poisson_estimate[i] = minimize(obj_fun, 0.5, (m, sum_k), bounds=((1e-10, np.inf),)).x

        poisson_estimate /= np.max(poisson_estimate)

        # -----------------------------------------------------------------------------
        # Plotting Results
        # -----------------------------------------------------------------------------

        plt.plot(entire_session_hist)  # obs this is equal to mean hist
        plt.plot(max_hist)
        plt.plot(mean_hist)
        plt.plot(median_hist)
        plt.plot(first_eigenvalue)
        plt.plot(poisson_estimate)
        plt.legend(["entire", "chunk_max", "chunk mean", "chunk median", "chunk eigenvalue", "Poisson estimate"])
        plt.show()

        # After this try (x, y) alignment
        # estimate chunk size based on firing rate
        # figure out best histogram size based on x0x0x0
