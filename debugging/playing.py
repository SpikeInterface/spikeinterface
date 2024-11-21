
from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks

from spikeinterface.preprocessing.inter_session_alignment import (
    session_alignment,
    plotting_session_alignment,
)
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# Load / generate some recordings
# --------------------------------------------------------------------------------------

recordings_list, _ = generate_session_displacement_recordings(
    num_units=50,
    recording_durations=[50, 50, 50]
)

# --------------------------------------------------------------------------------------
# Compute the peaks / locations with your favourite method
# --------------------------------------------------------------------------------------
# Note if you did motion correction the peaks are on the motion object.
# There is a function 'session_alignment.align_sessions_after_motion_correction()
# you can use instead of the below.

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
# See `session_alignment.py` for docs on these settings.

non_rigid_window_kwarg = session_alignment.get_non_rigid_window_kwargs()
non_rigid_window_kwarg["rigid"] = True

corrected_recordings_list, extra_info = session_alignment.align_sessions(
    recordings_list,
    peaks_list,
    peak_locations_list,
    alignment_order="to_session_1",  # "to_session_X" or "to_middle"
    non_rigid_window_kwargs=non_rigid_window_kwarg,
)

plotting_session_alignment.SessionAlignmentWidget(
    recordings_list,
    peaks_list,
    peak_locations_list,
    extra_info["session_histogram_list"],
    **extra_info["corrected"],
    spatial_bin_centers=extra_info["spatial_bin_centers"],
    drift_raster_map_kwargs={"clim":(-250, 0), "scatter_decimate": 10}  # TODO: option to fix this across recordings.
)

plt.show()
