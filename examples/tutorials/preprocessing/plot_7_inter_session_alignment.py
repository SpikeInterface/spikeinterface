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
    num_units=5,
    recording_durations=[10, 10],
    recording_shifts=((0, 0), (0, 200)),  # (x offset, y offset) pairs
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

plot_session_alignment(  # TODO: is this signature confusing?
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
# As we have used 2d histograms for alignment, we can also plot these with ``plot_activity_histogram_2d()``.
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
recordings_list = []
motion_info_list = []
for recording in [recording_part1, recording_part2]:

    rec, motion_info = si.correct_motion(recording, output_motion_info=True, preset="rigid_fast")

    recordings_list.append(rec)
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

corrected_recordings_list, extra_info = session_alignment.align_sessions_after_motion_correction(
    recordings_list, motion_info_list, align_sessions_kwargs
)

# %%
# As above, the inter-session alignment can be assessed using ``plot_session_alignment()``.

# %%
# Inter-session alignment settings
# --------------------------------
#
#
# TODO: do a bit of the exploration of the outputs, how the inter-session alignment works
# and how changing the kwargs changes key features. Also double-check the
