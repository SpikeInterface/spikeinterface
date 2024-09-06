from __future__ import annotations

import spikeinterface.full as si
from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion.motion_utils import make_2d_motion_histogram, make_3d_motion_histograms
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
from spikeinterface.sortingcomponents.motion.motion_utils import make_2d_motion_histogram, make_3d_motion_histograms
from scipy.optimize import minimize
from pathlib import Path
import alignment_utils  # TODO
import pickle
import session_alignment  # TODO
from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks

# TODO: all of the nonrigid methods (and even rigid) could be having some strange affects on AP
# waveforms. definately needs looking into!

# TODO: ask about best way to chunk, as ofc the peak detection takes
# recording as inputs so cannot use get traces with chunks function as planned.
# TODO: expose trimmed versions, robust xcorr

# Note, the cross correlation is intrinsically limited because for large
# shifts the value is too reduced by the reduction in number of points.
# but, of course cannot scale by number of points due to instability at edges
# This is a major problem, e.g. see the strange results for:
"""
    scalings = [np.ones(25), np.r_[np.zeros(10), np.ones(15)]]
    recordings_list, _ = generate_session_displacement_recordings(
        non_rigid_gradient=None, # 0.05, # 0.05,
        num_units=55,
        recording_durations=(100, 100, 100, 100),
        recording_shifts=(
            (0, 0), (0, 250), (0, -150), (0, -210),
        ),
        recording_amplitude_scalings=None, # {"method": "by_amplitude_and_firing_rate", "scalings": scalings},
        generate_unit_locations_kwargs={"margin_um": 0, "minimum_z": 0, "maximum_z": 0},
        seed=42,
    )
"""

"""
TODO: in this case, it is not necessary to run peak detection across
      the entire recording, would probably be sufficient to
      take a few chunks, of size determined by firing frequency of
      the neurons in the recording (or just take user defined size).
      For now, run on the entire recording and discuss the best way to
      run on chunked sections with Sam.
"""

# with nonrigid shift. This is less of a problem when restricting to a small
# windwo for the nonrigid because even if it fails catistrophically the nonrigid
# error will only be max(non rigid shifts). But its still not good.

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

# TODO: think about the nonrigid alignment, it correlates
# over the entire window. is this wise? try cutting it down a bit?

# TODO: We this interpolate, smooth for xcorr in both rigid and non-rigid case. Think aboout this / check this is ok
# maybe we only want to apply the smoothings etc for nonrigid like KS motion correction

# TODO: try forcing all unit locations to actually
# be within the probe. Add some notes on this because it is confusing.

# 1) write argument checks
# 2) go through with a fine tooth comb, fix all outstanding issues, tidy up,
#    plot everything to make sure it is working prior to writing tests.

# 4) to an optimisation shift and scale instead of the current xcorr method.
# 5) finalise estimation of chunk size (skip for now, try a new alignment method)
#    and optimal bin size.
# 6) make some presets? should estimate a lot of parameters based on the data, especially for nonrigid.
#    these are all basically based on probe geometry.


# Note, shifting can move a unit closer to the channel e.g. if separated
# by 20 um which can increase the signal and make shift estimation harder.

# go through everything and plot to check before writing tests.
# there is a relationship between bin size and nonrigid bins. If bins are
# too small then nonrigid is very unstable. So either choosing a bigger bin
# size or smoothing over the histogram in relation to the number
# of nonrigid bins may make sense.

# the results for nonrigid are very dependent on chosen parameters,
# in particular the number of nonrigid windows, gaussian scale,
# smoothing of the histgram. An optimaisation method may also
# serve to help reduce the number of parameters to choose.

# what you really want is for the window size to adapt to how
# busy the histogram is.

# Suprisingly, the session that is aligned TO can have a
#  major  affect.

# problem with current:
# - xcorr is not the best for large shifts due to lower num overlapping samples
# -


MOTION = False  # True
SAVE = True
PLOT = False
BIN_UM = 1


if SAVE:
    scalings = [np.ones(25), np.r_[np.zeros(10), np.ones(15)]]
    recordings_list, _ = generate_session_displacement_recordings(
        non_rigid_gradient=0.05, # 0.05,  # 0.05, # 0.05,
        num_units=55,
        recording_durations=(100, 100, 100), # , 100),
        recording_shifts=(
            (0, 0),
            (0, 75),
            (0, -150),
        ),
        recording_amplitude_scalings=None,  # {"method": "by_amplitude_and_firing_rate", "scalings": scalings},
        generate_unit_locations_kwargs={"margin_um": 0, "minimum_z": 0, "maximum_z": 0},
        generate_templates_kwargs=dict(
            ms_before=1.5,
            ms_after=3.0,
            mode="sphere",  # this is key to maintaining consistent unit positions with shift
            unit_params=dict(
                alpha=(150.0, 500.0),
                spatial_decay=(10, 45),
            ),
        ),
        seed=42,
    )

    if not MOTION:
        peaks_list = []
        peak_locations_list = []

        for recording in recordings_list:
            peaks, peak_locations = alignment_utils.prep_recording(
                recording,
                plot=PLOT,
            )
            peaks_list.append(peaks)
            peak_locations_list.append(peak_locations)

        # something relatively easy, only 15 units
        with open("all_recordings.pickle", "wb") as handle:
            pickle.dump((recordings_list, peaks_list, peak_locations_list), handle, protocol=pickle.HIGHEST_PROTOCOL)
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
                recordings_list[i],
                output_motion_info=True,
                estimate_motion_kwargs={
                    "rigid": True,
                    #     "win_shape": "gaussian",
                    #    "win_step_um": 50,
                    #   "win_margin_um": 0,
                },
            )
            recordings_list_new.append(new_recording)
            motion_info_list.append(motion_info)
        recordings_list = recordings_list_new

        with open("all_recordings_motion.pickle", "wb") as handle:
            pickle.dump((recordings_list, motion_info_list), handle, protocol=pickle.HIGHEST_PROTOCOL)

if MOTION:
    with open("all_recordings_motion.pickle", "rb") as handle:
        recordings_list, motion_info_list = pickle.load(handle)
else:
    with open("all_recordings.pickle", "rb") as handle:
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
        recordings_list,
        motion_info_list,
        rigid=False,
        override_nonrigid_window_kwargs=non_rigid_window_kwargs,
        chunked_bin_size_s="estimate",
        bin_um=BIN_UM,  # non_rigid_window_kwargs
    )
    peaks_list = [info["peaks"] for info in motion_info_list]
    peak_locations_list = [info["peak_locations"] for info in motion_info_list]
else:
    estimate_histogram_kwargs = {
        "bin_um": BIN_UM,
        "method": "chunked_supremum",  # TODO: double check scaling
        "chunked_bin_size_s": "estimate",
        "log_scale": True,
        "smooth_um": 10,
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
            "win_step_um": 100,
            "win_scale_um": 100,
            "win_margin_um": None,
            "zero_threshold": None,
         },
    }

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
    drift_raster_map_kwargs={"clim":(-250, 0)}  # TODO: option to fix this across recordings.
)

plt.show()

# No, even two sessions is a mess
# TODO: working assumptions, maybe after rigid, make a template for nonrigid alignment
# as at the moment all nonrigid to eachother is a mess
