"""
How SpikeInterface Handles Time
================================

Extracellular electrophysiology commonly involves synchronization of events across many timestreams.
For example, an experiment may involve displaying stimuli to an animal and recording the stimuli-evoked
neuronal responses. Accurate timing representation is critical for proper synchronization during analysis.

This tutorial explores how SpikeInterface stores time information and how to use it in your analysis
to ensure spike event timing is accurate.

Prerequisites:
--------------
A basic understanding of digital sampling (e.g., sampling frequency) is assumed.
For a refresher, review the explanation below.

"""

# %%
# .. dropdown:: Digital Sampling
#
#    What we are fundamentally interested in is recording the electrical waveforms found in the brain.
#    These 'continuous' signals change over time. To represent them in finite-memory computers, we sample
#    the continuous signals at discrete points in time.
#
#    When sampling, a key question is: *How fast should we sample the continuous data?* We could sample
#    the signal every :math:`N` seconds. For instance, if we sample 4 times per second, this is called the
#    'sampling frequency' (:math:`f_s`), expressed in Hertz (Hz), or samples per second.
#
#    An alternative way to think of this is to ask: *How often do we sample the signal?* In this example,
#    we sample every 0.25 seconds, which is called the 'sampling step' (:math:`t_s`), the inverse of the sampling frequency.
#
#    .. image:: handle-times-sampling-image.png
#       :alt: Image of continuous signal (1 second) with dots indicating samples collected at 0, 0.25, 0.5, and 0.75 seconds.
#       :width: 400px
#       :align: center
#
#    In real-world applications, signals are sampled much faster. For example, Neuropixels samples at
#    30 kHz (30,000 samples per second) with a sampling step of :math:`\frac{1}{30000} \approx 0.0000333` seconds.
#
#    Computers typically represent time as a long array of numbers, referred to here as a
#    'time array' (e.g., ``[0, 0.25, 0.5, 0.75, ...]``).
#
#
# Overview of Time Representations in SpikeInterface
# ---------------------------------------------------
# When you load a recording into SpikeInterface, it is associated with a time array.
# Depending on the data format, this may be loaded from metadata. If no time metadata is available,
# times are generated based on the sampling rate and the number of samples.

# %%
import spikeinterface.full as si

# Generate a recording for this example
recording, sorting = si.generate_ground_truth_recording(durations=[10])

# Print recording details
print("Number of samples:", recording.get_num_samples())
print("Sampling frequency:", recording.get_sampling_frequency())
print("Time vector:", recording.get_times())

# %%
# Retrieving information about times
# ----------------------------------
#
# In this example, no time metadata was associated with the recording, so a default time array was
# generated based on the number of samples and sampling frequency. The time array starts at 0 seconds
# and continues to :math:`10 - 1/f_s` seconds (where :math:`f_s` is the sampling frequency),
# in steps of :math:`1/f_s`. The recording duration is 10 seconds.
#
# There are a number of helper functions we can use to get information about the times,
# including:

print("Time vector (s):", recording.get_times())
print("Start time (s):", recording.get_start_time())
print("End time (s):", recording.get_end_time())
print("Duration (s):", recording.get_duration())

# %%
# Shifting the Start Time
# -----------------------
# You may want to change the start time of your recording. This can be done using the `shift_times()`
# method, which adds an offset in seconds to all timestamps in place.

# %%
recording.shift_times(100.15)
print(recording.get_times())  # Time now starts at 100.15 seconds

recording.shift_times(-50.15)
print(recording.get_times())  # Time now starts at 50 seconds

# %%
# Manually Setting a Time Vector
# ------------------------------
#
# It is also possible to manually set a time vector on your recording. This can be useful if you
# have true sample timestamps that were not loaded from metadata. The benefit of the full time
# vector is it can include small sampling rate drifts that are not captured when computing
# the time from number of samples and sampling frequency alone.

# %%
import numpy as np

fs = recording.get_sampling_frequency()

times = 50 + np.arange(recording.get_num_samples()) / fs  # Starts at 50 seconds
drift = np.cumsum(np.linspace(0, 0.001 / fs, recording.get_num_samples()))

true_times = times + drift

recording.set_times(true_times)
print(recording.get_times())

# %%
# Use ``reset_times()`` to discard explicit timestamps and start-time offsets in memory.
# Times are regenerated from the sampling frequency, starting at zero for every segment;
# timestamp files on disk are unchanged.

# %%
# .. warning::
#
#    If your time vector is regularly sampled at the recording's sampling frequency,
#    it is better to shift the default times rather than set your own time vector,
#    to save memory.

# %%
# Retrieving Timepoints from Sample Index
# ---------------------------------------
# SpikeInterface provides methods to convert between time points and sample indices.

# %%
sample_index = recording.time_to_sample_index(120)
print("Sample index:", sample_index)

timepoint = recording.sample_index_to_time(125000)
print("Time (s):", timepoint)


# %%
# Multi-segment times
# -------------------
# Each segment has its own timestamps and start time. Specify ``segment_index``
# when retrieving times or durations from a multi-segment recording.

recording_segments = si.generate_recording(durations=[10, 20], t_starts=[0, 20])

print("First segment start (s):", recording_segments.get_start_time(segment_index=0))
print("Second segment start (s):", recording_segments.get_start_time(segment_index=1))
print("First segment duration (s):", recording_segments.get_duration(segment_index=0))

# %%
# ``get_total_duration()`` sums the segment durations, excluding gaps between segments.

print("Total duration (s):", recording_segments.get_total_duration())


# %%
# Sorting Times
# -------------
# The sorting can also have times attached to it. By default, it will inherit the
# times of the attached recording. However, a sorting object cam have its own unique time stamps.
#
# A sorting has its own time offset: ``sorting.shift_times()`` shifts spike times
# in seconds without changing sample indices or the recording. Shifting the recording
# after registration does not automatically apply the same shift to the sorting.

sorting.register_recording(recording)

sorting.shift_times(1.0)

print("Sorting start time (s):", sorting.get_start_time())
print("Recording start time (s):", recording.get_start_time())

# %%
# Spike trains are returned as sample indices by default. Using ``return_times=True``
# returns spike times in seconds. Here we show the first five spike times.

unit_id_to_show = sorting.unit_ids[0]
print(sorting.get_unit_spike_train(unit_id_to_show, return_times=True)[:5])

# %%
# Shifting the sorting back by 51 seconds moves its start time to zero.
# The same five spikes now have timestamps 51 seconds earlier.

sorting.shift_times(-51.0)
print(sorting.get_unit_spike_train(unit_id_to_show, return_times=True)[:5])

# %%
# Aligning Events Across Timestreams
# -----------------------------------
# Aligning electrophysiology recordings with other data streams (e.g., behavioral data)
# is an important step in analysis. This is often done by acquiring a synchronization
# pulse on an additional channel.
#
# While SpikeInterface does not include built-in features for time-alignment,
# these resources may be helpful:
#
# * `SpikeGLX User Manual <https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#procedure-to-calibrate-sample-rates>`_
# * `OpenEphys Synchronization Guide <https://open-ephys.github.io/gui-docs/Tutorials/Data-Synchronization.html>`_
# * `NWB Temporal Alignment <https://neuroconv.readthedocs.io/en/main/user_guide/temporal_alignment.html>`_
