"""
How to perform inter-session alignment
======================================

In this tutorial we will assess and correct changes in probe position across
multiple experimental sessions using `inter-session alignment`.

This is often valuable for chronic-recording experiments, where the goal is to track units across sessions


Running inter-session alignment
-------------------------------

In SpikeInterface, it is recommended to perform inter-session alignment
following within-session motion correction (if used) and before whitening / sorting.
If you are running inter-session alignment after motion correction, see
:ref:`inter-session alignment after motion correction <with_motion_correction>`.

Preprocessed recordings should first be stored in a list:

.. code-block:: python

    recordings_list = [prepro_session_1, prepro_session_2, ...]

Here, we will simulate an experiment with two sessions by generating a pair of sessions in
which the probe is displaced 200 micrometers (μm) along its y-axis (depth).
First, we will import all required packages and functions:
"""

import spikeinterface.full as si
from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
from spikeinterface.preprocessing.inter_session_alignment import session_alignment
from spikeinterface.widgets import plot_session_alignment, plot_activity_histogram_2d
import matplotlib.pyplot as plt


# %%
# and then generate the test recordings:

recordings_list, _ = generate_session_displacement_recordings(  # TODO: add to spikeinterface.full ?
    num_units=8,
    recording_durations=[10, 10],
    recording_shifts=((0, 0), (0, 200)),  # (x offset, y offset) pairs
    seed=42
)

# %%
# We won't preprocess the simulated recordings in this tutorial, but you can imagine
# preprocessing steps have already been run (e.g. filtering, common reference etc.).
#
# To run inter-session alignment, peaks must be detected and localised
# as the locations of firing neurons are used to anchor the sessions' alignment.
#
# If you are **running inter-session alignment following motion correction**, the peaks will
# already be detected and localised. In this case, please jump to
# :ref:`inter-session alignment after motion correction <with_motion_correction>`.
#
# In this section of the tutorial, we will assume motion correction was not run, so we need to compute the peaks:

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
# we are now ready to perform inter-session alignment. There are many options associated
# with this method (see sections below). To edit the configurations, fetch the default options
# with the available getters function and make select changes as required:

estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
estimate_histogram_kwargs["histogram_type"] = "2d"

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
# Now, we are ready to use ``align_sessions_after_motion_correction()``.
# We can pass any arguments directly to ``align_sessions`` using the ``align_sessions_kwargs`` argument:

estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
estimate_histogram_kwargs["histogram_type"] = "2d"

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
# Below, the settings that control how inter-session alignment is performed
# are explored. These configs can be accessed by the getter functions
# ``get_estimate_histogram_kwargs``, ``get_compute_alignment_kwargs``,
# ``get_non_rigid_window_kwargs``, ``get_interpolate_motion_kwargs``.
#
# TODO: cannot add inter-session alignment to imports due to circular import error
#
# Estimate Histogram Kwargs
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The settings control how the activity histogram (used for alignment) is estimated
# for each session. They can be obtained with ``get_estimate_histogram_kwargs``.
#
# The ``"bin_um"`` parameter controls the bin-size of the activity histogram.
# Along the probe's y-axis, spatial bins will be generated according to this bin size.
#
# To compute the histogram, the session is split into chunks across time, and either
# the mean or median (bin-wise) taken across chunks. This generates the summary
# histgoram for that session to be used to estimate inter-session displacement.
#
# The ``"method"`` parameter controls whether the mean (``"chunked_mean"``)
# or median (``"chunked_median"``) is used. The idea of using the median is to
# reduce the effect periods of the recording which may be outliers
# due to noise or other signal contamination.
# ``"chunked_bin_size_s"`` sets the size of the temporal chunks. By default is
# ``"estimate"`` which estimates the chunk size based on firing frequency
# (see XXXX). Otherwise, can taFke a float for chunk size in seconds.
#
# The ``histogram_type`` can be ``"1d"` or ``"2d"``,
# if 1D the firing rate x spatial bin histogram is generated. Otherwise
# a firing rate x amplitude x spatial bin histogram is generated.
#
# We can visualise the histograms for each time chunk with:

estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
estimate_histogram_kwargs["histogram_type"] = "1d"
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
# Once the histograms have been generated for each session, the displacement
# between sessions is computed. ``get_compute_alignment_kwargs()`` set how this
# displacement is estimated.
#
# The estimation proceeds similar to the `Kilosort motion-correction <https://pubmed.ncbi.nlm.nih.gov/33859006/>`_
# method (see also the "kilosort-like" option in `:func:`correct_motion``.). Briefly, the cross-correlation
# of activity histograms is performed and the peak position used as a linear estimate of the displacement.
# For-non rigid alignment, first linear alignment is performed, then the probe y-axis is split into segments
# and linear estimation performed in each bin. Then, the displacement set at each bin center are interpolated acoss channels.
#
# Most compute-alignment kwargs are similar to those used in motion correction.
# Key arguments and those related to inter-session alignment
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
# Non-rigid window kwargs
# ~~~~~~~~~~~~~~~~~~~~~~~
# Non-rigid window kwargs determine how the non-rigid alignment is performed,
# in particular around how the y-axis of the probe is segmented into blocks
# (each which will be aligned using rigid alignment) are found here.
# (and see ``get_non_rigid_window_kwargs()``.
#
# We can see how the ``compute_alignment_kwargs`` control the non-rigid alignment
# by inspecting the output of inter-session alignment. First, we generate a
# pair of recordings with non-rigid displacement and perform rigid alignment:

recordings_list, _ = generate_session_displacement_recordings(
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
# Above, you can see there rigid alignemnt has well-matched one peak but
# the second peak is offset. Next, we can apply non-rigid alignment,
# and visualise the non-rigid segments that the probe is split into.
# Note that by default, Gaussian windows are used:

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

plt.plot(extra_info_nonrigid["corrected"]["corrected_session_histogram_list"][0])
plt.plot(extra_info_nonrigid["corrected"]["corrected_session_histogram_list"][1])
plt.plot(extra_info_nonrigid["non_rigid_windows"].T)
plt.show()

# %%
# It is notable that the non-rigid alignment is not perfect in this case. This
# is because for each bin, the displacement is computed and we can imagine
# its value being 'positioned' in the center of the bin. Then, the bin center
# values are interpolates across all channels. This leads to non-perfect alignment.
#
# This is in part because this is a simulated test case with only a few peaks and
# the spatial footprint of the APs is small. Nonetheless, the non-rigid window kwargs
# may be adjusting to maximise the performance of non-rigid alignment in the real world case.
