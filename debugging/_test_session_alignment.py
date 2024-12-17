from __future__ import annotations

from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
import matplotlib.pyplot as plt
import numpy as np
import pickle
from spikeinterface.preprocessing.inter_session_alignment import (
    session_alignment,
    plotting_session_alignment,
    alignment_utils
)
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
import spikeinterface.full as si


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

def _prep_recording(recording, plot=False):
    """
    :param recording:
    :return:
    """
    peaks = detect_peaks(recording, method="locally_exclusive")

    peak_locations = localize_peaks(recording, peaks, method="grid_convolution")

    if plot:
        si.plot_drift_raster_map(
            peaks=peaks,
            peak_locations=peak_locations,
            recording=recording,
            clim=(-300, 0),  # fix clim for comparability across plots
        )
        plt.show()

    return peaks, peak_locations

MOTION = True  # True
SAVE = True
PLOT = False
BIN_UM = 5


if SAVE:
    scalings = [np.ones(25), np.r_[np.zeros(10), np.ones(15)]]
    recordings_list, _ = generate_session_displacement_recordings(
        non_rigid_gradient=None, # 0.05,  # 0.05, # 0.05,
        num_units=55,
        recording_durations=(50, 50), # , 100),
        recording_shifts=(
            (0, 0),
            (0, 75),
        ),
        recording_amplitude_scalings=None,  # {"method": "by_amplitude_and_firing_rate", "scalings": scalings},
        generate_unit_locations_kwargs={"margin_um": 0, "minimum_z": 0, "maximum_z": 0},
        generate_templates_kwargs=dict(
            ms_before=1.5,
            ms_after=3.0,
            mode="sphere",  # this is key to maintaining consistent unit positions with shift
            unit_params=dict(
                alpha=(75, 125.0),  # firing rate
                spatial_decay=(10, 45),
            ),
        ),
        seed=42,
    )

    if not MOTION:
        peaks_list = []
        peak_locations_list = []

        for recording in recordings_list:
            peaks, peak_locations = _prep_recording(
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
                    "rigid": False,
                    "win_shape": "gaussian",
                    "win_step_um": 50,
                    "win_margin_um": 0,
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

estimate_histogram_kwargs = {
    "bin_um": BIN_UM,
    "method": "first_eigenvector",  # CHANGE NAME!!   # TODO: double check scaling
    "chunked_bin_size_s": "estimate",
    "log_scale": True,
    "depth_smooth_um": 10,
    "histogram_type": "activity_1d",  # "y_only", "locations_2d", "activity_2d""  TOOD: better names!
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
}
non_rigid_window_kwargs = {
    "rigid": False,
    "win_shape": "gaussian",
    "win_step_um": 250,
    "win_scale_um": 250,
    "win_margin_um": None,
    "zero_threshold": None,
}

if MOTION:
    corrected_recordings_list, extra_info = session_alignment.align_sessions_after_motion_correction(
        recordings_list,
        motion_info_list,
        align_sessions_kwargs={
            "alignment_order": "to_middle",
            "estimate_histogram_kwargs": estimate_histogram_kwargs,
            "compute_alignment_kwargs": compute_alignment_kwargs,
            "non_rigid_window_kwargs": non_rigid_window_kwargs,
        }
    )
    peaks_list = [info["peaks"] for info in motion_info_list]
    peak_locations_list = [info["peak_locations"] for info in motion_info_list]
else:
    corrected_recordings_list, extra_info = session_alignment.align_sessions(
        recordings_list,
        peaks_list,
        peak_locations_list,
        alignment_order="to_session_1",
        estimate_histogram_kwargs=estimate_histogram_kwargs,
        compute_alignment_kwargs=compute_alignment_kwargs,
    )

<<<<<<< HEAD

plotting_session_alignment.SessionAlignmentWidget(
    recordings_list,
    peaks_list,
    peak_locations_list,
    extra_info["session_histogram_list"],
    **extra_info["corrected"],
    spatial_bin_centers=extra_info["spatial_bin_centers"],
    drift_raster_map_kwargs={"clim":(-250, 0)}  # TODO: option to fix this across recordings.
)

=======
plotting_session_alignment.SessionAlignmentWidget(
    recordings_list,
    peaks_list,
    peak_locations_list,
    extra_info["session_histogram_list"],
    **extra_info["corrected"],
    spatial_bin_centers=extra_info["spatial_bin_centers"],
    drift_raster_map_kwargs={"clim":(-250, 0)}  # TODO: option to fix this across recordings.
)

>>>>>>> 978c5343c (Reformatting alignment methods and add 2D, need to tidy up.)
plt.show()

# TODO: estimate chunk size Hz needs to be scaled for time? is it not been done correctly?

<<<<<<< HEAD
# No, even two sessions is a mess
# TODO: working assumptions, maybe after rigid, make a template for nonrigid alignment
# as at the moment all nonrigid to eachother is a mess
if False:
=======
if False:
    # No, even two sessions is a mess
    # TODO: working assumptions, maybe after rigid, make a template for nonrigid alignment
    # as at the moment all nonrigid to eachother is a mess
>>>>>>> 978c5343c (Reformatting alignment methods and add 2D, need to tidy up.)
    A = extra_info["histogram_info_list"][2]["chunked_histograms"]

    mean_ = alignment_utils.get_chunked_hist_mean(A)
    median_ = alignment_utils.get_chunked_hist_median(A)
    supremum_ = alignment_utils.get_chunked_hist_supremum(A)
    poisson_ = alignment_utils.get_chunked_hist_poisson_estimate(A)
    eigenvector_ = alignment_utils.get_chunked_hist_eigenvector(A)

    plt.plot(mean_)
    plt.plot(median_)
    plt.plot(supremum_)
    plt.plot(poisson_)
    plt.plot(eigenvector_)
    plt.legend(["mean", "median", "supremum", "poisson", "eigenvector"])
    plt.show()