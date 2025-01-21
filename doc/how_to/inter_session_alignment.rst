How to perform inter-session alignment
======================================

In this how-to, we will assess and correct changes in probe position across multiple experimental sessions
using 'inter-session alignment'.

This is often valuable chronic-recording experiments, where the goal is to track units across sessions

A full tutorial, including details on the many settings for this procedure, can be found `here <TODO: ADD LINK (INTERNAL)>`_.

Running inter-session alignment
-------------------------------

In SpikeInterface, it is recommended to perform inter-session alignment
following within-session motion correction (if used) and before whitening.

Preprocessed recordings should be stored in a list before being passed
to the `session_alignment` functions:

.. code:: python

    recordings_list = [prepro_session_1, prepro_session_2, ...]

Here, we will simulate such an experiment by generating a pair of sessions in
which the probe is displaced 200 micrometers (um) along its y-axis (depth).

The first step involves running all required imports:

.. code:: python

    from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
    import spikeinterface.full as si
    from spikeinterface.preprocessing.inter_session_alignment import (  # TODO: should add all of the below to spikeinterface.full? (CHECK)
        session_alignment,
    )
    from spikeinterface.widgets import plot_session_alignment, plot_activity_histogram_2d
    import matplotlib.pyplot as plt


and then generating the test recordings:

.. code:: python

    recordings_list, _ = generate_session_displacement_recordings(  # TODO: add to spikeinterface.full ?
        num_units=5,
        recording_durations=[10, 10],
        recording_shifts=((0, 0), (0, 200)),  # (x offset, y offset) pairs
    )

We won't explicitly preprocess these recordings in this how-to, but you can imagine
preprocessing steps have already been run (e.g. filtering, common reference etc.).

To run inter-session alignment, we need to detect peaks and compute the peak locations,
as the location of firing neurons are used as anchors to align the sessions.

If you are **running inter-session alignment following motion correction**, the peaks will
already be detected and localised. In this case, please jump to the
:ref:`alignment guide <_with_motion_correction>`.

In this section we will imagine motion correction was not run, so we need to compute the peaks:

.. code:: python

    peaks_list, peak_locations_list = session_alignment.compute_peaks_locations_for_session_alignment(
                recordings_list,
                detect_kwargs={"method": "locally_exclusive"},
                localize_peaks_kwargs={"method": "grid_convolution"},
    )

The peak locations (before correction) can be visualised with the plotting function:

.. code:: python

    plot_session_alignment(
        recordings_list,
        peaks_list,
        peak_locations_list,
        drift_raster_map_kwargs={"clim":(-250, 0)}  # fix the color limit across plots for easy comparison
    )
    plt.show()

Now, we are ready to perform inter-session alignment. There are many options associated
with this method—the simplest way to edit these is to fetch the default options
using the getter function as below:

.. code:: python

    estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
    estimate_histogram_kwargs["histogram_type"] = "activity_2d"  # TODO: RENAME

    corrected_recordings_list, extra_info = session_alignment.align_sessions(
        recordings_list,
        peaks_list,
        peak_locations_list,
        estimate_histogram_kwargs=estimate_histogram_kwargs
    )

To assess the performance of inter-session alignment, `plot_session_alignment()`
will plot both the original and corrected recordings:

.. code:: python

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

As we have used 2d histograms for alignment, we can also plot these with ``plot_activity_histogram_2d()``.

.. _with_motion_correction:

Inter-session alignment after motion correction
-----------------------------------------------

If motion correction has already been performed, it is possible to reuse the
previously computed peaks and peak locations, avoiding the need for re-computation.
We will use the special function `align_sessions_after_motion_correction()` for this case.

Critically, the last preprocessing step prior to inter-session alignment should be motion correction,
so the correction for inter-session displacement will be **added directly to the motion correction**.
This is beneficial as it avoids interpolating the data (i.e. shifting the traces) more than once.

.. admonition:: Warning
   :class: warning

    To ensure that inter-session alignment adds the displacement directly to the motion-corrected recording
    to avoid repeated interpolation, motion correction must be the final operation applied to the recording
    prior to inter-session alignment.

    You can verify this by confirming the recording is an ``InterpolateMotionRecording`` with:

    .. code:: python

        type(recording)  # quick check, should print `InterpolateMotionRecording`

        from spikeinterace.sortingcomponents.motion.motion_utils import InterpolateMotionRecording

        assert isinstance(recording, InterpolateMotionRecording)  # error if not true

    ``align_sessions_after_motion_correction()`` will raise an error if the passed recordings
    are not all `InterpolateMotionRecordings`.

Again, let's create some test data. We can create a recording with motion errors,
then split it in two to simulate two separate sessions:

.. code:: python

    # Generate the recording with motion artefact
    motion_recording, _ = si.generate_drifting_recording(duration=100)
    total_duration = motion_recording.get_duration()
    split_time = total_duration / 2

    # Split in two to simulate two sessions
    recording_part1 = motion_recording.time_slice(start_time=0, end_time=split_time)
    recording_part2 = motion_recording.time_slice(start_time=split_time, end_time=total_duration)


Next, motion correction is performed, storing the results in a list:

.. code:: python

    # perform motion correction on each session, storing the outputs in lists
    recordings_list = []
    motion_info_list = []
    for recording in [recording_part1, recording_part2]:

        rec, motion_info = si.correct_motion(recording, output_motion_info=True, preset="rigid_fast")

        recordings_list.append(rec)
        motion_info_list.append(motion_info)

Now, we are ready to use ``align_sessions_after_motion_correction()``
to align the motion-corrected sessions.

This function should always be used for aligning motion-corrected sessions,
as it ensures the alignment parameters are properly matched.

We can pass any arguments directly to ``align_sessions`` using the ``align_sessions_kwargs`` argument:

.. code:: python

    estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
    estimate_histogram_kwargs["histogram_type"] = "activity_2d"  # TODO: RENAME

    align_sessions_kwargs = {"estimate_histogram_kwargs": estimate_histogram_kwargs}

    corrected_recordings_list, extra_info = session_alignment.align_sessions_after_motion_correction(
        recordings_list, motion_info_list, align_sessions_kwargs
    )

As above, the inter-session alignment can be assessed using ``plot_session_alignment()``.
