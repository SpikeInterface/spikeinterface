from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks

from spikeinterface.preprocessing.inter_session_alignment import (
    session_alignment,
)
from spikeinterface.widgets import plot_session_alignment, plot_activity_histogram_2d
import matplotlib.pyplot as plt

import spikeinterface.full as si
import numpy as np


si.set_global_job_kwargs(n_jobs=10)


if __name__ == '__main__':

    # --------------------------------------------------------------------------------------
    # Load / generate some recordings
    # --------------------------------------------------------------------------------------

    recordings_list, _ = generate_session_displacement_recordings(
        num_units=5,
        recording_durations=[25, 25, 25],
        recording_shifts=((0, 0), (0, -200), (0, 150)),
        non_rigid_gradient=None,
        seed=55,
    )

    peaks_list, peak_locations_list = session_alignment.compute_peaks_locations_for_session_alignment(
        recordings_list,
        detect_kwargs={"method": "locally_exclusive"},
        localize_peaks_kwargs={"method": "grid_convolution"},
    )

    # --------------------------------------------------------------------------------------
    # Do the estimation
    # --------------------------------------------------------------------------------------
    # For each session, an 'activity histogram' is generated. This can be `entire_session`
    # or the session can be chunked into segments and some summary generated taken over then.
    # This might be useful if periods of the recording have weird kinetics or noise.

    non_rigid_window_kwargs = session_alignment.get_non_rigid_window_kwargs()
    non_rigid_window_kwargs["rigid"] = False
#    non_rigid_window_kwargs["num_shifts_global"] = 500
#    non_rigid_window_kwargs["num_shifts_block"] = 24
 #   non_rigid_window_kwargs["win_step_um"] = 125
 #   non_rigid_window_kwargs["win_scale_um"] = 60

    estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
    estimate_histogram_kwargs["method"] = "chunked_median"
    estimate_histogram_kwargs["histogram_type"] = "activity_2d"
    estimate_histogram_kwargs["bin_um"] = 2
    estimate_histogram_kwargs["log_scale"] = True
    estimate_histogram_kwargs["weight_with_amplitude"] = False

    compute_alignment_kwargs = session_alignment.get_compute_alignment_kwargs()
    compute_alignment_kwargs["interpolate"] = False

    corrected_recordings_list, extra_info = session_alignment.align_sessions(
        recordings_list,
        peaks_list,
        peak_locations_list,
        alignment_order="to_session_1",  # "to_session_X" or "to_middle"
        non_rigid_window_kwargs=non_rigid_window_kwargs,
        estimate_histogram_kwargs=estimate_histogram_kwargs,
        compute_alignment_kwargs=compute_alignment_kwargs,
    )

    plot_session_alignment(
        recordings_list,
        peaks_list,
        peak_locations_list,
        extra_info["session_histogram_list"],
        **extra_info["corrected"],
        spatial_bin_centers=extra_info["spatial_bin_centers"],
        drift_raster_map_kwargs={"clim":(-250, 0), "scatter_decimate": 10}
    )
    plt.show()

    if estimate_histogram_kwargs["histogram_type"]  == "activity_2d":
        plot_activity_histogram_2d(
            extra_info["session_histogram_list"],
            extra_info["spatial_bin_centers"],
            extra_info["corrected"]["corrected_session_histogram_list"]
        )
        plt.show()
