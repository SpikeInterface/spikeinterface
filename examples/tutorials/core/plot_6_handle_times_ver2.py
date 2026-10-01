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
#    30 kHz (30,000 samples per second) with a sampling step of :math:`\frac{1}{30000} = 0.0003` seconds.
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
print(f"Number of samples: {recording.get_num_samples()}")
print(f"Sampling frequency: {recording.get_sampling_frequency()}")
print(f"Time vector: {recording.get_times()}")

# %%
# In this example, no time metadata was associated with the recording, so a default time array was
# generated based on the number of samples and sampling frequency. The time array starts at 0 seconds
# and continues to 10 seconds (10 * `sampling frequency` samples) in steps of the sampling step size,
# :math:`\dfrac{1}{\text{sampling frequency}}`.

# %%
# Shifting the Start Time
# -----------------------
# You may want to change the start time of your recording. This can be done using the `shift_start_time()`
# method, which adjusts the first time point of the recording.

# %%
recording.shift_start_time(100.15)
print(recording.get_times())  # Time now starts at 100.15 seconds

recording.shift_start_time(-50.15)
print(recording.get_times())  # Time now starts at 50 seconds

# %%
# Setting Time Vector Changes Spike Times
# ---------------------------------------
# If we use the sorting object with the default times, the spike times will reflect those times.
# You can register the recording to change the spike times accordingly.

# %%
unit_id_to_show = sorting.unit_ids[0]
spike_times_orig = sorting.get_unit_spike_train(unit_id_to_show, return_times=True)

# Register the recording to adjust spike times
sorting.register_recording(recording)

spike_times_new = sorting.get_unit_spike_train(unit_id_to_show, return_times=True)

# %%
# Manually Setting a Time Vector
# ------------------------------
# It is also possible to manually set a time vector on your recording. This can be useful if you
# have true sample timestamps that were not loaded from metadata.

# %%
import numpy as np

times = np.linspace(0, 10, recording.get_num_samples())
offset = np.cumsum(np.linspace(0, 0.1, recording.get_num_samples()))
true_times = times + offset

recording.set_times(true_times)
print(recording.get_times())

# %%
# .. warning::
#
#    For regularly spaced time vectors, it is better to shift the default times rather
#    than set your own time vector to save memory.

# %%
# Retrieving Timepoints from Sample Index
# ---------------------------------------
# SpikeInterface provides methods to convert between time points and sample indices.

# %%
sample_index = recording.time_to_sample_index(5.0)
print(sample_index)

timepoint = recording.sample_index_to_time(125000)
print(timepoint)

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
