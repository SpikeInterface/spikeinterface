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
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def run_benchmarking(
        shift=15,
        recording_durations=(100, 100),
        num_units=60,
        alpha=(100, 600),
        bin_um=2.5,
        seed=None,
):
    """

    """
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

    default_unit_params_range["alpha"] = alpha
    # default_unit_params_range["b"] = (0.5, 1)
    # default_unit_params_range["c"] = (0.5, 1)

    recordings_list, _ = generate_session_displacement_recordings(
        non_rigid_gradient=None,
        num_units=num_units,
        recording_durations=recording_durations,
        recording_shifts=(
            (0, 0), (0, shift)
        ),
        recording_amplitude_scalings=None,
        seed=seed,
    )

    peaks_list = []
    peak_locations_list = []

    for recording in recordings_list:
        peaks, peak_locations = alignment_utils.prep_recording(
            recording, plot=False,
        )
        peaks_list.append(peaks)
        peak_locations_list.append(peak_locations)

    alignment_results = alignment_utils.estimate_session_displacement_benchmarking(
        recordings_list, peaks_list, peak_locations_list, bin_um
    )

    for key in alignment_results["motion_arrays"].keys():
        arr = alignment_results["motion_arrays"][key]
        arr -= arr[0]
        alignment_results["motion_arrays"][key] = arr

    return alignment_results, recordings_list, peaks_list, peak_locations_list


def num_units_run_func(input):
    shift, num_units = input

    all_results = run_benchmarking(
        shift=shift,
        recording_durations=(100, 100),
        num_units=num_units,
        alpha=(100, 600),
        bin_um=2.5,
        seed=None,
    )

    filename = f"nu-{num_units}_shift-{shift}.pickle"
    output_file = method_output_path / filename

    if output_file.is_file():
        output_file.unlink()

    print(f"Writing: {filename}")
    with open(output_file, 'wb') as handle:
        pickle.dump(
            all_results, handle, protocol=pickle.HIGHEST_PROTOCOL
    )
