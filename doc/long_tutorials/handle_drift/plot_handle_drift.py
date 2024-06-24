"""
===========================================
Handle motion/drift with spikeinterface NEW
===========================================

When running *in vivo* electrophysiology recordings, movement of the probe is
an inevitability, especially when the subjects are not head-fixed. SpikeInterface
includes a number of popular methods to compensate for probe motion during the
preprocessing step.

------------------------------------------
What is drift and where does it come from?
------------------------------------------

Movement of the probe means that the spikes recorded on the probe 'drift' along it.
Typically, this motion is vertical along the probe (along the 'y' axis) which
manifests as the units moving long the probe in space.

All common motion-correction methods address this vertical drift. Horizontal ('x')
or forward/backwards ('z') motion, that would appear as the amplitude of a unit
changing over time, are much harder to model and not handled in available motion-correction algorithms.
Fortunately, vertical drift is the most common form of motion as the probe is
more likely to move along the path it was inserted, rather than in other directions
where it is buffeted against the brain.

Vertical drift can come in two forms, 'rigid' and 'non-rigid'. Rigid drift
is drift caused by movement of the entire probe and the motion is the
same for all points along the probe. Non-rigid drift is instead caused by
local movement of parts of the brain along the probe, and can affect
the recording at only certain points along the probe.

------------------------------------------
How SpikeInterface handles drift
------------------------------------------

Spikeinterface offers a very flexible framework to handle drift as a
preprocessing step. In this tutorial we will cover the three main
drift-correction algorithms implemented in SpikeInterface with
a focus on running the methods and interpreting the output. For
more information on the theory and implementation of these methods,
see the :ref:`motion_correction` section of the documentation.

------------------------------------------
The drift correction steps
------------------------------------------

The easiest way to run drift correction in SpikeInterface is with the
high-level :py:func:`~spikeinterface.preprocessing.correct_motion()` function.
This function takes a preprocessed recording as input and then internally runs
several steps and returns a lazy recording that interpolates the traces on-the-fly
to compensate for the motion.

The
:py:func:`~spikeinterface.preprocessing.correct_motion()`
function provides a convenient wrapper around a number of sub-functions
that together implement the full drift correction algorithm.

Internally this function runs the following steps:

| **1.** ``localize_peaks()``
| **2.** ``select_peaks()`` (optional)
| **3.** ``estimate_motion()``
| **4.** ``interpolate_motion()``

All these sub-steps have many parameters which dictate the
speed and effectiveness of motion correction. As such, ``correct_motion``
provides three setting 'presets' which configure the motion correct
to proceed either as:

* **rigid_fast** - a fast, not particularly accurate correction assuming ridigt drift.
* **kilosort-like** - Mimics what is done in Kilosort (REF)
* **nonrigid_accurate** - A decentralised drift correction, introduced by the Paninski group (REF)

**Now, let's dive into running motion correction with these three
methods on a simulated dataset and interpreting the output.**

"""
# %%
#.. warning::
#    The below code uses multiprocessing. If you are on Windows, you may
#    need to place the code within a  ``if __name__ == "__main__":`` block.

# %%
# -------------------------------------------
# Setting up and preprocessing the recording
# -------------------------------------------
#
# First, we will import the modules we will need for this tutorial:

import matplotlib.pyplot as plt
import spikeinterface.full as si
from spikeinterface.generation.drifting_generator import generate_drifting_recording
from spikeinterface.preprocessing.motion import motion_options_preset
from spikeinterface.sortingcomponents.motion_interpolation import correct_motion_on_peaks
from spikeinterface.widgets import plot_peaks_on_probe

# %% read the file
# Next, we will generate a synthetic drifting recording. This recording will
# have 500 separate units with firing rates randomly distributed between
# 15 and 25 Hz. The recording will be in total 1000 seconds long.

# We will create a zigzag drift pattern on the recording, starting at
# 100 seconds and with a peak-to-peak period of 100 seconds (so we will
# have 9 zigzags through our recording). We also add some nonlinear
# drift in to the motion (i.e. is not the same across the entire probe).

num_units = 50 # 500,
duration = 100  # 1000,

_, raw_recording, _ = generate_drifting_recording(
    num_units=num_units,
    duration=duration,
    generate_sorting_kwargs=dict(firing_rates=(15, 25), refractory_period_ms=4.0),
    seed=42,
    generate_displacement_vector_kwargs=dict(motion_list=[
            dict(
                drift_mode="zigzag",
                non_rigid_gradient=None, # 0.1,
                t_start_drift=10.0,  # 100.0
                t_end_drift=None,
                period_s=10,  # 100
            ),
        ],
    )
)
print(raw_recording)

# %%
# Before performing motion correction, we will **preprocess** the recording
# with a bandpass filter and a common median reference.

filtered_recording = si.bandpass_filter(raw_recording, freq_min=300.0, freq_max=6000.0)
preprocessed_recording = si.common_reference(filtered_recording, reference="global", operator="median")

# %%
#.. warning::
#    It is better to not whiten the recording before motion estimation, as this
#    will give a better estimate of the peak locations. Whitening should
#    be performed after motion correction.

# %%
# ----------------------------------------
# Run motion correction with one function!
# ----------------------------------------
#
# Correcting for drift is easy! You just need to run a single function.
# We will now run motion correction on our recording using the three
# presets described above - **rigid_fast**, **kilosort_like** and
# **nonrigid_accurate**.
#
# Under the hood, each step, peak localisation, selection, motion estimation
# and interpolation expose a lot of options, making them highly flexible.
# The presets are simply a set of configurations which sets the motion
# correction steps to perform as described in the original methods.
# For example, we can print the full set of **kilosort_like** preset options:
print(motion_options_preset["kilosort_like"])

# %%
# Now, lets run motion correction with our three presets. We will
# set the ``job_kwargs`` to parallelize the job over a number of CPU cores.
# Motion correction is quite computationally intensive and so it is
# very useful to run it across a high numer of jobs to speed it up.

presets_to_run = ("rigid_fast", "kilosort_like", "nonrigid_accurate")

job_kwargs = dict(n_jobs=40, chunk_duration="1s", progress_bar=True)

results = {preset: {} for preset in presets_to_run}
for preset in presets_to_run:

    recording_corrected, motion_info = si.correct_motion(
        preprocessed_recording, preset=preset,  output_motion_info=True, **job_kwargs
    )
    results[preset]["motion_info"] = motion_info

# %%
#.. seealso::
#   It is often very useful to save the output of motion correction to
#   file, so they can be loaded later. This can be done by setting
#   the ``folder`` argument of ``correct_motion`` to a path to save in.
#   The ``motion_info`` can be loaded back with ``si.load_motion_info``.

# %%
# -------------------
# Plotting the results
# -------------------
#
# For all methods we have 4 plots. On the x-axis of all plots we have
# the (binned time). The plots display:
#   * **top left:** The estimated peak depth for every detected peaks.
#   * **top right:** The estimated peak depths after motion correction.
#   * **bottom left:** The average motion vector across depths and all motion across spatial depths (for non-rigid estimation).
#   * **bottom right:** if motion correction is non rigid, the motion vector across depths is plotted as a map, with the color code representing the motion in micrometers.
#
# These plots are quite complicated, so it is worth covering them in detail.
# For every detected action potential in our recording, we first estimate
# its depth (first panel) using a method from :py:func:`~spikeinterface.postprocessing.compute_unit_locations()`.
# Then, the probe motion is estimated the location of the detected peaks are
# adjusted to account for this (second panel). The motion estimation produces
# a measure of how much and in what direction the probe is moving at any given
# time bin (third panel). For non-rigid motion correction, the probe is divided
# into subsections - the motion vectors displayed are per subsection (i.e. per
# 'binned spatial depth') as well as the average. On the fourth panel, we see a
# more detailed representation of the motion vectors. We can see the motion plotted
# as a heatmap at each binned spatial depth across all time bins. We see it captures
# the zigzag pattern (alternating light and dark colors).

for preset in presets_to_run:
    fig = plt.figure(figsize=(7, 7))
    si.plot_motion_info(
        results[preset]["motion_info"],
        recording=recording_corrected,  # recording only used to get the real times
        figure=fig,
        depth_lim=(400, 600),
        color_amplitude=True,
        amplitude_cmap="inferno",
        scatter_decimate=10,
    )
    fig.suptitle(f"{preset=}")

# %%
# A few comments on the figures:
#   * The preset **'rigid_fast'** has only one motion vector for the entire probe because it is a "rigid" case.
#     The motion amplitude is globally underestimated because it averages across depths.
#     However, the corrected peaks are flatter than the non-corrected ones, so the job is partially done.
#     The big jump at=600s when the probe start moving is recovered quite well.
#   * The preset **kilosort_like** gives better results because it is a non-rigid case.
#     The motion vector is computed for different depths.
#     The corrected peak locations are flatter than the rigid case.
#     The motion vector map is still be a bit noisy at some depths (e.g around 1000um).
#   * The preset **nonrigid_accurate** seems to give the best results on this recording.
#     The motion vector seems less noisy globally, but it is not "perfect" (see at the top of the probe 3200um to 3800um).
#     Also note that in the first part of the recording before the imposed motion (0-600s) we clearly have a non-rigid motion:
#     the upper part of the probe (2000-3000um) experience some drifts, but the lower part (0-1000um) is relatively stable.
#     The method defined by this preset is able to capture this.

# %%
# -------------------------------------------------
# Correcting Peak Locations after Motion Correction
# -------------------------------------------------
#
# To understand how motion correction is applied to our data, it is
# important to understand the spikeinterface `peak` and `peak_locations`
# objects, explored further in the below dropdown.
#

# %%
# .. dropdown:: Dropdown title
#
#   Information about detected action potentials is represented in
#   SpikeInterface is ``peaks`` and ``peak_locations`` objects. The
#   ``peaks`` object is an array for every detected action potential in th
#    the dataset, containing its XXX, XX, XX, XX. it is created by the
#    XXXX function.
#
#   The ``peak_locations`` is a partner object to the ``peaks`` object
#   containing XXX. For every peak in ``peak`` there is a corresponding
#   location in ``peak_locations``. The peak locations is estimated
#   using the XXXX function. One way of correcting for motion is to
#   correct these peak locations directly, using the output of
#   ``si.correct_motion`` or ``si.estimate_motion``.

# %%
# The result of motion correction can be applied by interpolating the
# raw data to correct for the drift. Essentially, this shifts the signal
# across the probe depth by, at each channel, interpolating other channels
# XXXX. This is performed on the `corrected_recording` output from the
# `correct_motion` channel. This is useful for continuing with
# preprocessing and sorting with the corrected recording.
#
# The other way to apply the motion correction is to the ``peaks`` and
# ``peaks_location`` objects directly. This is done using the function
# ``correct_motion_on_peaks()``. Given a set of peaks, peak locations and
# the ``motion`` object output from ``correct_motion``, it will shift the
# location of the peaks according to the motion estimate, outputting a new
# ``peak_locations`` object. This is done to plot the peak locations in
# the next section.

# %%
#.. warning::
#   Note that the `peak_locations` output by `correct_motion`'s
#   `motion_info` is the ORIGINAL (uncorrected) peak locations. To get the corrected
#   peak locations, `correct_motion_on_peaks()` must be used!

for preset in presets_to_run:

    motion_info = results[preset]["motion_info"]

    peaks = motion_info["peaks"]

    original_peak_locations = motion_info["peak_locations"]

    corrected_peak_locations = correct_motion_on_peaks(peaks, original_peak_locations, motion_info['motion'], recording_corrected)  # TODO: what recording to use.

    widget = plot_peaks_on_probe(recording_corrected, [peaks, peaks], [original_peak_locations, corrected_peak_locations], ylim=(300,600))
    widget.figure.suptitle(preset)

# %%
# -------------------------
# Comparing the  Run Times
# -------------------------
#
# The different methods also have different speeds, the 'nonrigid_accurate'
# requires more computation time, particulary at the ``estimate_motion`` phase,
# as seen in the run times:

for preset in presets_to_run:
    print(preset)
    print(results[preset]["motion_info"]["run_times"])

# %%
# ------------------------
# Summary
# ------------------------
#
# That's it for our overall tour of correcting motion in
# SpikeInterface. If you'd like to explore more, see the API docs
# for `estimate_motion()`, `interpolate_motion`() (ay others?). Remember
# that correcting motion makes some assumptions on your datatype - always
# output and plot the motion correction information for your recordings,
# to make sure they are acting as expected!
