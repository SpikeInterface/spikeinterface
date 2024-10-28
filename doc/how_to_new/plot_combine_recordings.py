"""
====================================
Combine recordings in SpikeInterface
====================================

In this tutorial, we will walk through combining multiple recording objects. Sometimes this occurs due to hardware
settings (e.g., Intan software has a default setting of new files every 1 minute) or the experimenter decides to
split their recording into multiple files for different experimental conditions. If the probe has not been moved,
however, then during sorting it would likely make sense to combine these individual recording objects into one
recording object.

------------
Why Combine?
------------

Combining your data into a single recording allows you to have consistent labels (`unit_ids`) across the whole recording.

Spike sorters seek to sort spikes within a recording into groups of units. Thus if multiple `Recording` objects have the
exact same probe location within some tissue and are occurring continuously in time, the units between the `Recordings` will
be the same. But if we sort each recording separately, the unit ids given by the sorter will not be the same between each
`Sorting`, and so we will need extensive post-processing to try to figure out which units are actually the same between
each `Sorting`. By combining everything into one `Recording`, all spikes will be sorted into the same pool of units.

---------------------------------------
Combining recordings continuous in time
---------------------------------------

Some file formats (e.g., Intan) automatically create new files every minute or few minutes (with a setting that can be user
controlled). Other times an experimenter separates their recording for experimental reasons. SpikeInterface provides two
tools for bringing together these files into one `Recording` object.
"""

# %%
# ------------------------
# Concatenating Recordings
# ------------------------

# First, let's cover concatenating recordings together. This will generate a mono-segment
# recording object. Let's load a set of Intan files. 0 is the amplifier data for Intan

import spikeinterface as si  # This is only core
import spikeinterface.extractors as se

recording_one, _ = si.generate_ground_truth_recording(durations=[25])
recording_two, _ = si.generate_ground_truth_recording(durations=[25])

print(recording_one)

print(recording_two)

# %%
# Next, we will concatenate these recordings together.

concatenated_recording = si.concatenate_recordings([recording_one, recording_two])

print(concatenated_recording)

# %%
# If we know that we will deal with a lot of files, we can actually work our
# way through a series of them relatively quickly by doing

list_of_recs = [si.generate_ground_truth_recording(durations=[25])[0] for _ in range(4)]
list_of_recordings = []
for rec in list_of_recs:
    list_of_recordings.append(rec)
recording = si.concatenate_recordings(list_of_recordings)

# %%
# -----------------
# Append Recordings
# -----------------
#
# If you wish to keep each recording as a separate segment identity (e.g. if doing baseline, stim, poststim) you can use
# `append` instead of `concatenate`. This has the benefit of allowing you to keep different parts of data
# separate, but it is important to note that not all sorters can handle multi-segment objects.

recording = si.append_recordings([recording_one, recording_two])

print(recording)

# %%
# --------
# Pitfalls
# --------
#
# It's important to remember that these operations are directional. So:

recording_forward = si.concatenate_recordings([recording_one, recording_two])
recording_backward = si.concatenate_recordings([recording_two, recording_one])

# %%
# This is important because your spike times will be relative to the start of your recording.
