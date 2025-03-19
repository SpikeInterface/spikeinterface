"""
How to perform inter-session alignment
======================================

In this how-to, we will assess and correct changes in probe position across multiple experimental sessions
using `inter-session alignment`.

This is often valuable for chronic-recording experiments, where the goal is to track units across sessions


Running inter-session alignment
-------------------------------

In SpikeInterface, it is recommended to perform inter-session alignment
following within-session motion correction (if used) and before whitening.

Preprocessed recordings should first be stored in a list:

.. code-block:: python

    recordings_list = [prepro_session_1, prepro_session_2, ...]

Here, we will simulate such an experiment by generating a pair of sessions in
which the probe is displaced 200 micrometers (μm) along its y-axis (depth).
First, we will import all required packages and functions:
"""

import spikeinterface.full as si
from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
from spikeinterface.preprocessing.inter_session_alignment import session_alignment
from spikeinterface.widgets import plot_session_alignment, plot_activity_histogram_2d
import matplotlib.pyplot as plt


# %%
# and then generating the test recordings:

recordings_list, _ = generate_session_displacement_recordings(  # TODO: add to spikeinterface.full ?
    num_units=8,
    recording_durations=[10, 10],
    recording_shifts=((0, 0), (0, 200)),  # (x offset, y offset) pairs
    seed=42
)

# %%
# We won't explicitly preprocess these recordings in this how-to, but you can imagine
# preprocessing steps have already been run (e.g. filtering, common reference etc.).
#
# To run inter-session alignment, peaks must be detected and localised
# as the locations of firing neurons are used to anchor the sessions alignment.
#
# If you are **running inter-session alignment following motion correction**, the peaks will
# already be detected and localised. In this case, please jump to
# :ref:`inter-session alignment after motion correction <with_motion_correction>`.
#
# In this section we will assume motion correction was not run, so we need to compute the peaks:

peaks_list, peak_locations_list = session_alignment.compute_peaks_locations_for_session_alignment(
            recordings_list,
            detect_kwargs={"method": "locally_exclusive"},
            localize_peaks_kwargs={"method": "grid_convolution"},
)

# %%
# The peak locations (before correction) can be visualised with the plotting function:

plot_session_alignment(
    recordings_list,
    peaks_list,
    peak_locations_list,
)
plt.show()

# %%
# Now, we are ready to perform inter-session alignment. There are many options associated
# with this method—the simplest way to edit these is to fetch the default options
# with the getter function and make select changes as required:

estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
estimate_histogram_kwargs["histogram_type"] = "activity_2d"  # TODO: RENAME

corrected_recordings_list, extra_info = session_alignment.align_sessions(
    recordings_list,
    peaks_list,
    peak_locations_list,
    estimate_histogram_kwargs=estimate_histogram_kwargs
)

# %%
# To assess the performance of inter-session alignment, ``plot_session_alignment()``
# will plot both the original and corrected recordings:

plot_session_alignment(
    recordings_list,
    peaks_list,
    peak_locations_list,
    extra_info["session_histogram_list"],
    **extra_info["corrected"],
    spatial_bin_centers=extra_info["spatial_bin_centers"],
    drift_raster_map_kwargs={"clim":(-250, 0)}
)
plt.show()

# %%
# As we have used 2d histograms for alignment, we can also plot these with ``plot_activity_histogram_2d()``:

plot_activity_histogram_2d(
    extra_info["session_histogram_list"],
    extra_info["spatial_bin_centers"],
    extra_info["corrected"]["corrected_session_histogram_list"]
)
plt.show()

#
# .. _with_motion_correction:

# %%
# Inter-session alignment after motion correction
# -----------------------------------------------
#
# If motion correction has already been performed, it is possible to reuse the
# previously computed peaks and peak locations, avoiding the need for re-computation.
# We will use the special function` `align_sessions_after_motion_correction()`` for this case.
#
# Critically, the last preprocessing step prior to inter-session alignment should be motion correction.
# This ensures the correction for inter-session alignment will be **added directly to the motion correction**.
# This is beneficial as it avoids interpolating the data (i.e. shifting the traces) more than once.
#
# .. admonition:: Warning
#    :class: warning
#
#     To ensure that inter-session alignment adds the displacement directly to the motion-corrected recording
#     to avoid repeated interpolation, motion correction must be the final operation applied to the recording
#     prior to inter-session alignment.
#
#     You can verify this by confirming the recording is an ``InterpolateMotionRecording`` with:
#
#     .. code-block::
#         type(recording)``  # quick check, should print `InterpolateMotionRecording`
#
#         from spikeinterface.sortingcomponents.motion.motion_interpolation import InterpolateMotionRecording
#
#         assert isinstance(recording, InterpolateMotionRecording)  # error if not true
#
#
# ``align_sessions_after_motion_correction()`` will raise an error if the passed recordings
# are not all ``InterpolateMotionRecordings``.
#
# Let's first create some test data. We can create a recording with motion errors,
# then split it in two to simulate two separate sessions:

# Generate the recording with motion artefact
motion_recording = si.generate_drifting_recording(duration=100)[0]
total_duration = motion_recording.get_duration()
split_time = total_duration / 2

# Split in two to simulate two sessions
recording_part1 = motion_recording.time_slice(start_time=0, end_time=split_time)
recording_part2 = motion_recording.time_slice(start_time=split_time, end_time=total_duration)

# %%
# Next, motion correction is performed, storing the results in a list:

# perform motion correction on each session, storing the outputs in lists
recordings_list_motion = []
motion_info_list = []
for recording in [recording_part1, recording_part2]:

    rec, motion_info = si.correct_motion(recording, output_motion_info=True, preset="rigid_fast")

    recordings_list_motion.append(rec)
    motion_info_list.append(motion_info)

# %%
# Now, we are ready to use ``align_sessions_after_motion_correction()``
# to align the motion-corrected sessions. This function should always be used
# for aligning motion-corrected sessions, as it ensures the alignment
# parameters are properly matched.
#
# We can pass any arguments directly to ``align_sessions`` using the ``align_sessions_kwargs`` argument:

estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
estimate_histogram_kwargs["histogram_type"] = "activity_2d"  # TODO: RENAME

align_sessions_kwargs = {"estimate_histogram_kwargs": estimate_histogram_kwargs}

corrected_recordings_list_motion, _ = session_alignment.align_sessions_after_motion_correction(
    recordings_list_motion, motion_info_list, align_sessions_kwargs
)

# %%
# As above, the inter-session alignment can be assessed using ``plot_session_alignment()``.

# %%
# Inter-session alignment settings
# --------------------------------
#
# Estimate Histogram Kwargs
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The settings control how the activity histogram (used for alignment) is estimated
# for each session. LINK TO: get_estimate_histogram_kwargs
#
# The ``"bin_um"`` parameter controls the bin-size of the activity histogram.
# Along the probe's y-axis, spatial bins will be generated according to the bin-size.
# set with ``bin_um``.
#
# To compute the histogram, the session is split into chunks across time, and either
# the mean or median taken bin-size across chunks, to generate a summary histogram.
# The ``"method"`` parameter controls whether the mean (``"chunked_mean"``)
# or median (``"chunked_median"``) is used. The idea is to exclude periods of the
# recording which may be outliers due to noise or other signal contamination.
# ``"chunked_bin_size_s"`` sets the size of the temporal chunks. By default is
# ``"estimate"`` which estimates the chunk size based on firing frequency
# (see XXXX). Otherwise, can taFke a float for chunk size in seconds.
# The ``histogram_type`` can be ``"activity_1d`"` or ``"activity_2d"``,
# if 1D the firing rate x spatial bin histogram is generated. Otherwise
# a firing rate x amplitude x spatial bin histogram is generated.
#
# The histogram used can be obtained from the extra information:. Going back to
# the data from our previous example (before motion correction), we can plot
# the chunked histogram for the first session (0 index).
# Each line is a histogram estimated on a temporal chunk.

estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
estimate_histogram_kwargs["histogram_type"] = "activity_1d"  # TODO: RENAME  chunked_bin_size_s
estimate_histogram_kwargs["chunked_bin_size_s"] = 1.0

_, extra_info_rigid = session_alignment.align_sessions(
    recordings_list,
    peaks_list,
    peak_locations_list,
    estimate_histogram_kwargs=estimate_histogram_kwargs,
)

plt.plot(extra_info_rigid["histogram_info_list"][0]["chunked_histograms"].T)
plt.xlabel("Spatial bim (um)")
plt.show()

# %%
# Compute Alignment Kwargs
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Once the histograms have been generated for each session, the alignment
# between sessions is computed. ``compute_alignmen_kwargs()`` LINK are used to
# determine how this is performed. Alignment estimation proceeds similar
# to the Kilosort motion-correction method (see also Kilosort-like). Breifly,
# the cross-correlation of activity histograms is performed and the peak used
# as a linear estimate of the displacement. For-non rigid alignment, first linear alignment
# is performed, then the probe y-axis is binned and linear estimation performed in each bin.
# Then, shifts located at each bin center are interpolated acoss channels (see below).
#
# Most compute-alignment kwargs are similar to those used in motion correcion
# and can be read about HERE. Key arguments and those related to inter-session alignment
# include:
#
# ``"num_shifts_global"``: This is the number of shifts to perform cross-correlation across for linear alignment.
# Put differently, this is the maximum allowed displacement to consider for rigid alignment.
# ``"num_shifts_block"``: The number of shifts to perform cross-correlation across for non-linear alignment (within each spatial bin).
# ``"akima_interp_nonrigid"``: If ``True``, perform akima interpolation across non-rigid spatial bins (rather than linear).
# ``"min_crosscorr_threshold"``: To estimate alignment, normalised cross-correlation is performed. In some cases, particularly
# for non-rigid alignment, there may be little correlation within a bin. To stop aberrant shifts estimated on poor correlations,
# this sets a minimum value for the correlation used to estimate the shift. If less than this value, the shift will be set to zero.
#
# Compute Alignment Kwargs
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Non-rigid window kwargs determine how the non-rigid alignment is performed,
# in particular around how the y-axis of the probe is segmented into blocks
# (each which will be aligned using rigid alignment). TODO. ``compute_alignment_kwargs``
# are found here.
#
# We can see how the ``compute_alignment_kwargs`` control the non-rigid alignment
# by inspecting the output of inter-session alignment. By default, the non-rigid
# alignment is computed with Gaussian windows:

# TODO: RENAME

recordings_list, _ = generate_session_displacement_recordings(  # TODO: add to spikeinterface.full ?
    num_units=8,
    recording_durations=[10, 10],
    recording_shifts=((0, 0), (0, 200)),  # (x offset, y offset) pairs
    non_rigid_gradient=0.1,
    seed=42
)

peaks_list, peak_locations_list = session_alignment.compute_peaks_locations_for_session_alignment(
            recordings_list,
            detect_kwargs={"method": "locally_exclusive"},
            localize_peaks_kwargs={"method": "grid_convolution"},
)


non_rigid_window_kwargs = session_alignment.get_non_rigid_window_kwargs()
non_rigid_window_kwargs["rigid"] = True

_, extra_info_rigid = session_alignment.align_sessions(
    recordings_list,
    peaks_list,
    peak_locations_list,
    estimate_histogram_kwargs=estimate_histogram_kwargs,
    non_rigid_window_kwargs=non_rigid_window_kwargs,
)

plt.plot(extra_info_rigid["corrected"]["corrected_session_histogram_list"][0])
plt.plot(extra_info_rigid["corrected"]["corrected_session_histogram_list"][1])
plt.show()

# %%

non_rigid_window_kwargs = session_alignment.get_non_rigid_window_kwargs()
non_rigid_window_kwargs["rigid"] = False
non_rigid_window_kwargs["win_step_um"] = 200
non_rigid_window_kwargs["win_scale_um"] = 100

compute_alignment_kwargs = session_alignment.get_compute_alignment_kwargs()
compute_alignment_kwargs["akima_interp_nonrigid"] = True

_, extra_info_nonrigid = session_alignment.align_sessions(
    recordings_list,
    peaks_list,
    peak_locations_list,
    estimate_histogram_kwargs=estimate_histogram_kwargs,
    non_rigid_window_kwargs=non_rigid_window_kwargs,
)

# %%

plt.plot(extra_info_nonrigid["corrected"]["corrected_session_histogram_list"][0])
plt.plot(extra_info_nonrigid["corrected"]["corrected_session_histogram_list"][1])
plt.plot(extra_info_nonrigid["non_rigid_windows"].T)
plt.show()
