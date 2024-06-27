"""
===========================================
Handle probe drift with spikeinterface NEW
===========================================

Probe movement is an inevitability when running
*in vivo* electrophysiology recordings. Motion, caused by physical
movement of the probe or the sliding of brain tissue
deforming across the probe, can complicate the sorting
and analysis of units.

SpikeInterface offers a flexible framework to handle motion correction
as a preprocessing step. In this tutorial we will cover the three main
drift-correction algorithms implemented in SpikeInterface
(**rigid_fast**, **kilosort_like** and **nonrigid_accurate**) with
a focus on running the methods and interpreting the output.

For more information on the theory and implementation of these methods,
see the :ref:`motion_correction` section of the documentation.

---------------------
What is probe drift?
---------------------

The inserted probe can move from side-to-side (*'x' direction*),
up-or-down (*'y' direction*) or forwards-or-backwards (*'z' direction*).
Movement in the 'x' and 'z' direction is harder to model than vertical
drift (i.e. along the probe depth), and are not handled by most motion
correction algorithms. Fortunately, vertical drift which is most easily
handled is most pronounced as the probe is most likely to move along the path of insertion.

Vertical drift can come in two forms, *'rigid'* and *'non-rigid'*. Rigid drift
is drift caused by movement of the entire probe, and the motion is
similar across all channels along the probe depth. In contrast,
non-rigid drift is instead caused by local movements of neuronal tissue along the
probe, and can selectively affect subsets of channels.

--------------------------
The drift correction steps
--------------------------

The easiest way to run drift correction in SpikeInterface is with the
high-level :py:func:`~spikeinterface.preprocessing.correct_motion()` function.
This function takes a recording as input and returns a motion-corrected
recording object. As with all other preprocessing steps, the correction (in this
case interpolation of the data to correct the detected motion) is lazy and applied on-the-fly when data is needed).

The :py:func:`~spikeinterface.preprocessing.correct_motion()`
function implements motion correction algorithms in a modular way
wrapping a number of subfunctions that together implement the
full drift correction algorithm.

These drift-correction modules are:

| **1.** ``localize_peaks()`` (detect spikes and localize their position on the probe)
| **2.** ``select_peaks()`` (optional, select a subset of peaks to use to estimate motion)
| **3.** ``estimate_motion()`` (estimate motion using the detected spikes)
| **4.** ``interpolate_motion()`` (perform interpolation on the raw data to account for the estimated drift).

All these sub-steps have many parameters which dictate the
speed and effectiveness of motion correction. As such, ``correct_motion``
provides three setting 'presets' which configure the motion correct
to proceed either as:

* **rigid_fast** - a fast, not particularly accurate correction assuming rigid drift.
* **kilosort-like** - Mimics what is done in Kilosort.
* **nonrigid_accurate** - A decentralized drift correction (DREDGE), introduced by the Paninski group.

When using motion correction in your analysis, please make sure to
:ref:`cite the appropriate paper for your chosen method<cite-motion-correction>`.


**Now, let's dive into running motion correction with these three
methods on a simulated dataset.**

"""

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
# Next, we will generate a synthetic, drifting recording. This recording will
# have 100 separate units with firing rates randomly distributed between
# 15 and 25 Hz.
#
# We will create a zigzag drift pattern on the recording, starting at
# 100 seconds and with a peak-to-peak period of 100 seconds (so we will
# have 9 zigzags through our recording). We also add some non-linearity
# to the imposed motion.

# %%
#.. note::
#    This tutorial can take a long time to run with the default arguments.
#    If you would like to run this locally, you may want to edit ``num_units``
#    and ``duration`` to smaller values (e.g. 25 and 100 respectively).
#
#    Also note, the below code uses multiprocessing. If you are on Windows, you may
#    need to place the code within a  ``if __name__ == "__main__":`` block.


num_units = 100  # 250 still too many I think!
duration = 1000

_, raw_recording, _ = generate_drifting_recording(
    num_units=num_units,
    duration=duration,
    generate_sorting_kwargs=dict(firing_rates=(15, 25), refractory_period_ms=4.0),
    seed=42,
    generate_displacement_vector_kwargs=dict(motion_list=[
            dict(
                drift_mode="zigzag",
                non_rigid_gradient=None, # 0.1,
                t_start_drift=int(duration/10),
                t_end_drift=None,
                period_s=int(duration/10),
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
# We can run these presents with the ``preset`` argument of
# :py:func:`~spikeinterface.preprocessing.correct_motion()`. Under the
# hood, the presets define a set of parameters by set how to run the
# 4 submodules that make up motion correction (described above).
print(motion_options_preset["kilosort_like"])

# %%
# Now, lets run motion correction with our three presets. We will
# set the ``job_kwargs`` to parallelize the job over a number of CPU cores—motion
# correction is computationally intensive and will run faster with parallelization.

presets_to_run = ("rigid_fast", "kilosort_like", "nonrigid_accurate")

job_kwargs = dict(n_jobs=40, chunk_duration="1s", progress_bar=True)

results = {preset: {} for preset in presets_to_run}
for preset in presets_to_run:

    corrected_recording, motion_info = si.correct_motion(
        preprocessed_recording, preset=preset,  output_motion_info=True, **job_kwargs
    )
    results[preset]["motion_info"] = motion_info

# %%
#.. seealso::
#   It is often very useful to save ``motion_info`` to a
#   file, so it can be loaded and visualized later. This can be done by setting
#   the ``folder`` argument of
#   :py:func:`~spikeinterface.preprocessing.correct_motion()` to a path to write
#   the motion output to. The ``motion_info`` can be loaded back with
#   ``si.load_motion_info``.

# %%
# -------------------
# Plotting the results
# -------------------
#
# Next, let's plot the results of our motion estimation using the ``plot_motion_info()``
# function. The plot contains 4 panels, on the x-axis of all plots we have
# the (binned time). The plots display:
#   * **top left:** The estimated peak depth for every detected peak.
#   * **top right:** The estimated peak depths after motion correction.
#   * **bottom left:** The average motion vector across depths and all motion across spatial depths (for non-rigid estimation).
#   * **bottom right:** if motion correction is non rigid, the motion vector across depths is plotted as a map, with the color code representing the motion in micrometers.

for preset in presets_to_run:

    fig = plt.figure(figsize=(7, 7))

    si.plot_motion_info(
        results[preset]["motion_info"],
        recording=corrected_recording,  # the recording is only used to get the real times
        figure=fig,
        depth_lim=(400, 600),
        color_amplitude=True,
        amplitude_cmap="inferno",
        scatter_decimate=10,            # Only plot every 10th peak
    )
    fig.suptitle(f"{preset=}")

# %%
# These plots are quite complicated, so it is worth covering them in detail.
# For every detected spike in our recording, we first estimate
# its depth (first panel) using a method from
# :py:func:`~spikeinterface.postprocessing.compute_unit_locations()`.
#
# Then, the probe motion is estimated the location of the detected peaks are
# adjusted to account for this (second panel).
#
# The motion estimation produces
# a measure of how much and in what direction the probe is moving at any given
# time bin (third panel). For non-rigid motion correction, the probe is divided
# into subsections - the motion vectors displayed are per subsection (i.e. per
# 'binned spatial depth') as well as the average.
#
# On the fourth panel, we see a
# more detailed representation of the motion vectors. We can see the motion plotted
# as a heatmap at each binned spatial depth across all time bins. It captures
# the zigzag pattern (alternating light and dark colors) of the injected motion.

# %%
# A few comments on the figures:
#   * The preset **'rigid_fast'** has only one motion vector for the entire probe because it is a 'rigid' case.
#     The motion amplitude is globally underestimated because it averages across depths.
#     However, the corrected peaks are flatter than the non-corrected ones, so the job is partially done.
#     The big jump at=600s when the probe start moving is recovered quite well.
#   * The preset **kilosort_like** gives better results because it is a non-rigid case.
#     The motion vector is computed for different depths.
#     The corrected peak locations are flatter than the rigid case.
#     The motion vector map is still be a bit noisy at some depths (e.g around 1000um).
#   * The preset **nonrigid_accurate** seems to give the best results on this recording.
#     The motion vector seems less noisy globally, but it is not 'perfect' (see at the top of the probe 3200um to 3800um).
#     Also note that in the first part of the recording before the imposed motion (0-600s) we clearly have a non-rigid motion:
#     the upper part of the probe (2000-3000um) experience some drifts, but the lower part (0-1000um) is relatively stable.
#     The method defined by this preset is able to capture this.

# %%
# -------------------------------------------------
# Correcting Peak Locations after Motion Correction
# -------------------------------------------------
#
# The result of motion correction can be applied to the data in two ways.
# The first is by interpolating the raw traces to correct for the estimated drift.
# This changes the data in the
# recording by shifting the signal across channels, and is given in the
# `corrected_recording` output from :py:func:`~spikeinterface.preprocessing.correct_motion()`.
# This is useful in most cases, for continuing
# with preprocessing and sorting with the corrected recording.
#
# The second way is to apply the results of motion correction directly
# to the ``peak_locations`` object. If you are not familiar with
# SpikeInterface's ``peak`` and ``peak_locations`` objects,
# these are explored further in the below dropdown.

# %%
# .. dropdown:: 'Peaks' and 'Peak Locations' in SpikeInterface
#
#   Information about detected spikes is represented in
#   SpikeInterface's ``peaks`` and ``peak_locations`` objects. The
#   ``peaks`` object is an array for containing the
#   sample index, channel index (where its signal
#   is strongest), amplitude and recording segment index for every detected spike
#   in the dataset. It is created by the
#   :py:func:`~spikeinterface.sortingcomponents.peak_detection.detect_peaks()`
#   function.
#
#   The ``peak_locations`` is a partner object to the ``peaks`` object,
#   and contains the estimated location (``"x"``, ``"y"``) of the spike. For every spike in
#   ``peaks`` there is a corresponding location in ``peak_locations``.
#   The peak locations is estimated using the
#   :py:func:`~spikeinterface.sortingcomponents.peak_localization.localise_peaks()`
#   function.

# %%
# The other way to apply the motion correction is to the ``peaks`` and
# ``peaks_location`` objects directly. This is done using the function
# ``correct_motion_on_peaks()``. Given a set of peaks, peak locations and
# the ``motion`` object output from :py:func:`~spikeinterface.preprocessing.correct_motion()`,
# it will shift the location of the peaks according to the motion estimate, outputting a new
# ``peak_locations`` object. This is done to plot the peak locations in
# the next section.
#


# %%
#.. warning::
#   Note that the ``peak_locations`` output by
#   :py:func:`~spikeinterface.preprocessing.correct_motion()`
#   (in the ``motion_info`` object) is the original (uncorrected) peak locations.
#   To get the corrected peak locations, ``correct_motion_on_peaks()`` must be used!

for preset in presets_to_run:

    motion_info = results[preset]["motion_info"]

    peaks = motion_info["peaks"]

    original_peak_locations = motion_info["peak_locations"]

    corrected_peak_locations = correct_motion_on_peaks(peaks, original_peak_locations, motion_info['motion'], corrected_recording)

    widget = plot_peaks_on_probe(corrected_recording, [peaks, peaks], [original_peak_locations, corrected_peak_locations], ylim=(300,600))
    widget.figure.suptitle(preset)

# %%
# -------------------------
# Comparing the  Run Times
# -------------------------
#
# The different methods also have different speeds, the 'nonrigid_accurate'
# requires more computation time, in particular at the ``estimate_motion`` phase,
# as seen in the run times:

for preset in presets_to_run:
    print(preset)
    print(results[preset]["motion_info"]["run_times"])

# %%
# ------------------------
# Summary
# ------------------------
#
# That's it for our tour of motion correction in
# SpikeInterface. Remember that correcting motion makes some
# assumptions on your data (e.g. number of channels, noise in the recording)—always
# plot the motion correction information for your
# recordings, to make sure the correction is behaviour as expected!
