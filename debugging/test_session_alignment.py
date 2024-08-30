from __future__ import annotations

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
import alignment_utils  # TODO
import pickle
import session_alignment  # TODO
from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks
from spikeinterface.widgets.motion import DriftRasterMapWidget
from spikeinterface.widgets.base import BaseWidget
import plotting


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
import alignment_utils  # TODO
import pickle
import session_alignment  # TODO
from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks


# TODO: add different modes (to mean, to nth session...)
# TODO: document that the output is Hz

# TODO: major check, refactor and tidy up
# list out carefully all notes
# handle the case where the passed recordings are not motion correction recordings.

# 3) think about and add  new neurons that are introduced when shifted

# 4) add interpolation of the histograms prior to cross correlation
# 5) add robust cross-correlation
# 6) add trimmed methods
# 7) add better way to estimate chunk length.

# try and interpolate /smooth the xcorr. What about smoothing the activity histograms directly?
# look into te akima spline


MOTION = False  # True
SAVE = False
PLOT = False
BIN_UM = 1


if SAVE:
    scalings = [np.ones(25), np.r_[np.zeros(10), np.ones(15)]]
    recordings_list, _ = generate_session_displacement_recordings(
        non_rigid_gradient=0.1, # 0.05,
        num_units=25,
        recording_durations=(100, 100), #  100, 100),
        recording_shifts=(
            (0, 0), (0, 75) # , (0, -125), (0, 25),
        ),
        recording_amplitude_scalings=None, # {"method": "by_amplitude_and_firing_rate", "scalings": scalings},
        seed=42,
    )

    if not MOTION:
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
    else:
        # if False:
        # TODO: need to align spatial bin calculation between estimate motion and
        # estimate session methods so they are more easily interoperable. OR
        # just take spatial bin centers from interpoalte!
        recordings_list_new = []
        peaks_list = []
        peak_locations_list = []
        motion_info_list = []
        from spikeinterface.preprocessing.motion import correct_motion

        for i in range(len(recordings_list)):
            new_recording, motion_info = correct_motion(
                recordings_list[i], output_motion_info=True, estimate_motion_kwargs={
                    "rigid": True,
               #     "win_shape": "gaussian",
                #    "win_step_um": 50,
                 #   "win_margin_um": 0,
                }
            )
            recordings_list_new.append(new_recording)
            motion_info_list.append(motion_info)
        recordings_list = recordings_list_new

        with open('all_recordings_motion.pickle', 'wb') as handle:
            pickle.dump((recordings_list, motion_info_list),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

if MOTION:
    with open('all_recordings_motion.pickle', 'rb') as handle:
        recordings_list, motion_info_list = pickle.load(handle)
else:
    with open('all_recordings.pickle', 'rb') as handle:
        recordings_list, peaks_list, peak_locations_list = pickle.load(handle)

# TODO: need docs to be super clear from  estimate from existing motion,
# as will use motion correction nonrigid bins even if it is suboptimal.
if MOTION:
    from session_alignment import align_sessions_after_motion_correction

    non_rigid_window_kwargs = {
        "win_shape": "gaussian",
        "win_step_um": 50,
        "win_margin_um": 0,
    }
    corrected_recordings_list, motion_objects_list, extra_info = align_sessions_after_motion_correction(
        recordings_list, motion_info_list, rigid=False, override_nonrigid_window_kwargs=non_rigid_window_kwargs, chunked_bin_size_s="estimate", bin_um=BIN_UM  # non_rigid_window_kwargs
    )
    peaks_list = [info["peaks"] for info in motion_info_list]
    peak_locations_list = [info["peak_locations"] for info in motion_info_list]
else:
    non_rigid_window_kwargs = {
        "win_shape": "gaussian",
        "win_step_um": 25,
        "win_margin_um": 0,
    }
    corrected_recordings_list, motion_objects_list, extra_info = session_alignment.align_sessions(
        recordings_list,
        peaks_list,
        peak_locations_list,
        bin_um=BIN_UM,
        histogram_estimation_method="chunked_mean",
        alignment_method="mean_crosscorr",
        log_scale=True,
        rigid=False,
        non_rigid_window_kwargs=non_rigid_window_kwargs,
        chunked_bin_size_s="estimate"
    )

plotting.SessionAlignmentWidget(
    recordings_list,
    peaks_list,
    peak_locations_list,
    extra_info["histogram_info"]["session_histogram_list"],
    **extra_info["corrected"],
    spatial_bin_centers=extra_info["spatial_bin_centers"],
)

plt.show()
