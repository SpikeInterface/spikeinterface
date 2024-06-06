"""
Handle motion/drift with spikeinterface NEW
===========================================

When running *in vivo* electrophysiology recordings, movement of the probe is
an inevitability, especially when the subjects are not head-fixed. SpikeInterface
includes a number of popular methods to compensate for probe motion during the
preprocessing step.

### What is drift and where does it come from?

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

### How SpikeInterface handles drift

Spikeinterface offers a very flexible framework to handle drift as a
preprocessing step. In this tutorial we will cover the three main
drift-correction algorithms implemented in SpikeInterface with
a focus on running the methods and interpreting the output. For
more information on the theory and implementation of these methods,
see the :ref:`motion_correction` section of the documentation.

### The drift correction steps

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
speed and effectiveness of motion correction. As such, `correct_motion`
provides three setting 'presets' which configure the motion correct
to proceed either as:

* `rigid_fast` - a fast, not particularly accurate correction assuming ridigt drift.
* `kilosort-like` - Mimics what is done in Kilosort (REF)
* `nonrigid_accurate` - A decentralised drift correction, introduced by the Paninski group (REF)

**Now, let's dive into running motion correction with these three
methods on a simulated dataset and interpreting the output.

"""



# %%
#.. seealso::
#    hello world

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import shutil
import spikeinterface.full as si

# %% read the file
from spikeinterface.extractors import toy_example
from spikeinterface.generation.drifting_generator import generate_drifting_recording

# TODO: add a note that it must be run in a if __name__ == "__main__" block.
# TODO: is there currently any way to compute accuracy of method based on
# drift-corrected vs. original static recording?

_, raw_rec, _ = generate_drifting_recording(
    num_units=300,
    duration=1000,
    generate_sorting_kwargs=dict(firing_rates=(15, 25), refractory_period_ms=4.0),
    seed=42,
)
print(raw_rec)

# %%
# We preprocess the recording with bandpass filter and a common median reference.
# Note, that it is better to not whiten the recording before motion estimation
# to get a better estimate of peak locations!

def preprocess_chain(rec):
    rec = si.bandpass_filter(rec, freq_min=300.0, freq_max=6000.0)
    rec = si.common_reference(rec, reference="global", operator="median")
    return rec


rec = preprocess_chain(raw_rec)

job_kwargs = dict(n_jobs=40, chunk_duration="1s", progress_bar=True)

# %%
# Run motion correction with one function!
# ----------------------------------------
#
# Correcting for drift is easy! You just need to run a single function.
# We will try this function with 3 presets.
#
# Internally a preset is a dictionary of dictionaries containing all parameters for every steps.
#
# Here we also save the motion correction results into a folder to be able to load them later.

# internally, we can explore a preset like this
# every parameter can be overwritten at runtime
from spikeinterface.preprocessing.motion import motion_options_preset

print(motion_options_preset["kilosort_like"])

# %%
# lets try theses 3 presets
some_presets = ("rigid_fast", "kilosort_like", "nonrigid_accurate")
results = {preset: {} for preset in some_presets}  # TODO: RENAME VAR

# %%
# and compute motion with 3 presets

for preset in some_presets:
    print("Computing with", preset)

    recording_corrected, motion_info = si.correct_motion(  # TODO: RECORDING_CORRECTED UNUSED
        rec, preset=preset,  output_motion_info=True, **job_kwargs
    )
    results[preset]["motion_info"] = motion_info

# %%
# Plot the results
# ----------------
#
# We load back the results and use the widgets module to explore the estimated drift motion.
#
# For all methods we have 4 plots:
#   * **top left:** time vs estimated peak depth
#   * **top right:** time vs peak depth after motion correction
#   * **bottom left:** the average motion vector across depths and all motion across spatial depths for non-rigid estimation)
#   * **bottom right:** if motion correction is non rigid, the motion vector across depths is plotted as a map, with the color code representing the motion in micrometers.
#
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

for preset in some_presets:
    fig = plt.figure(figsize=(14, 8))
    si.plot_motion(
        results[preset]["motion_info"],
        figure=fig,
        depth_lim=(400, 600),
        color_amplitude=True,
        amplitude_cmap="inferno",
        scatter_decimate=10,
    )
    fig.suptitle(f"{preset=}")

# %%
# Plot peak localization
# ----------------------
#
# We can also use the internal extra results (peaks and peaks location) to check if putative
# clusters have a lower spatial spread after the motion correction.
#
# Here we plot the estimated peak locations (left) and the corrected peak locations
# (on right) on top of the probe.
# The color codes for the peak amplitudes.
#
# We can see here that some clusters seem to be more compact on the 'y' axis, especially
# for the preset "nonrigid_accurate".
#
# Be aware that there are two ways to correct for the motion:
#   1. Interpolate traces and detect/localize peaks again  (`interpolate_recording()`)
#   2. Compensate for drifts directly on peak locations (`correct_motion_on_peaks()`)
#
# Case 1 is used before running a spike sorter and the case 2 is used here to display the results.

from spikeinterface.sortingcomponents.motion_interpolation import correct_motion_on_peaks

for preset in some_presets:

    fig, axs = plt.subplots(ncols=2, figsize=(12, 8), sharey=True)

    ax = axs[0]
    si.plot_probe_map(rec, ax=ax)

    motion_info = results[preset]["motion_info"]

    peaks = motion_info["peaks"]
    sr = rec.get_sampling_frequency()
    time_lim0 = 0
    time_lim1 = 50
    mask = (peaks["sample_index"] > int(sr * time_lim0)) & (peaks["sample_index"] < int(sr * time_lim1))
    sl = slice(None, None, 5)
    amps = np.abs(peaks["amplitude"][mask][sl])
    amps /= np.quantile(amps, 0.95)
    c = plt.get_cmap("inferno")(amps)

    color_kargs = dict(alpha=0.2, s=2, c=c)

    loc = motion_info["peak_locations"]
    ax.scatter(loc["x"][mask][sl], loc["y"][mask][sl], **color_kargs)

    loc2 = correct_motion_on_peaks(
        motion_info["peaks"],
        motion_info["peak_locations"],
        rec.sampling_frequency,
        motion_info["motion"],
        motion_info["temporal_bins"],
        motion_info["spatial_bins"],
        direction="y",
    )

    ax = axs[1]
    si.plot_probe_map(rec, ax=ax)
    ax.scatter(loc2["x"][mask][sl], loc2["y"][mask][sl], **color_kargs)

    ax.set_ylim(400, 600)
    fig.suptitle(f"{preset=}")

# %%
# Accuracy and Run Times
# ----------------------
#
# Presets and related methods have differents accuracies but also computation speeds.
# It is good to have this in mind!

run_times = []
for preset in some_presets:
    run_times.append(results[preset]["motion_info"]["run_times"])
keys = run_times[0].keys()

bottom = np.zeros(len(run_times))
fig, ax = plt.subplots()
for k in keys:
    rtimes = np.array([rt[k] for rt in run_times])
    if np.any(rtimes > 0.0):
        ax.bar(some_presets, rtimes, bottom=bottom, label=k)
    bottom += rtimes
ax.legend()
