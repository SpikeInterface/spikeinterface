from pathlib import Path
import spikeinterface.full as si
import numpy as np

base_path = Path(r"X:\neuroinformatics\scratch\jziminski\ephys\inter-session-alignment\test_motion_project_short\derivatives\1119617")

sessions = [
    "1119617_LSE1_shank12_g0",
    "1119617_posttest1_shank12_g0",
    "1119617_pretest1_shank12_g0",
]

recordings_list = []
peaks_list = []
peak_locations_list = []

for ses in sessions:

    ses_path = base_path / ses

    rec = si.load_extractor(ses_path / "preprocessing" / "si_recording")
    rec = si.astype(rec, np.float32)

    recordings_list.append(rec)
    peaks_list.append(np.load(ses_path / "motion_npy_files" / "peaks.npy" ))
    peak_locations_list.append(np.load(ses_path / "motion_npy_files" / "peak_locations.npy"))


estimate_histogram_kwargs = {
    "bin_um": 1,
    "method": "chunked_median",  # TODO: double check scaling
    "chunked_bin_size_s": "estimate",
    "log_scale": False,
    "smooth_um": 5,
}
compute_alignment_kwargs = {
    "num_shifts_block": None,  # TODO: can be in um so comaprable with window kwargs.
    "interpolate": False,
    "interp_factor": 10,
    "kriging_sigma": 1,
    "kriging_p": 2,
    "kriging_d": 2,
    "smoothing_sigma_bin": False,  # 0.5,
    "smoothing_sigma_window": False,  # 0.5,
    "akima_interp_nonrigid": False,

    "non_rigid_window_kwargs": {
        "win_shape": "gaussian",
        "win_step_um": 400,
        "win_scale_um": 400,
        "win_margin_um": None,
        "zero_threshold": None,
    },
}

import session_alignment
import plotting
import matplotlib.pyplot as plt

# TODO: add some print statements for progress
corrected_recordings_list, motion_objects_list, extra_info = session_alignment.align_sessions(
    recordings_list,
    peaks_list,
    peak_locations_list,
    alignment_order="to_session_1",
    rigid=False,
    estimate_histogram_kwargs=estimate_histogram_kwargs,
    compute_alignment_kwargs=compute_alignment_kwargs,
)

plotting.SessionAlignmentWidget(
    recordings_list,
    peaks_list,
    peak_locations_list,
    extra_info["session_histogram_list"],
    **extra_info["corrected"],
    spatial_bin_centers=extra_info["spatial_bin_centers"],
    drift_raster_map_kwargs={"clim":(-250, 0), "scatter_decimate": 10}  # TODO: option to fix this across recordings.
)

plt.show()
