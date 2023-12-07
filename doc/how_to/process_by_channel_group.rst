Processing a Recording by Channel Group (e.g. Shank)
===========================================================

In this tutorial, we will walk through how to preprocess and sorting a recording
per channel group. A channel group is a subset of channels grouped by some
feature - a typical example is grouping channels by shank in a Neuropixels recording.

First, we can check the channels on our recording are grouped as we expect. For example,
in the below example we have a X-shank neuropixels recording. We can visualise the
channel groupings with XXX.


On our SpikeInterface recording, we can also access the channel groups per-channel

1) channel groups
2) index channel ids by groups

Why would you want to preprocess or sort by group?

1)
2)
3)

Splitting a Recording by Channel Group
--------------------------------------

We can split a recording into mutliply recordings, one for each channel group, with the `split_by` method.

Here, we split a recording by channel group to get a `split_recording_dict`. This is a dictionary
containing the recordings, split by channel group:

```
split_recording_dict = recording.split_by("group")
print(split_recording_dict)
XXXXX
```

Now, we are ready to preprocess and sort by channel group.

Preprocessing a Recording by Channel Group
------------------------------------------

The essense of preprocessing by channel group is to first split the recording
into separate recordings, perform the preprocessing steps, then aggregate
the channels back together.

Here, we loop over the split recordings, preprocessing each shank group
individually. At the end, we use the `aggregate_channels` function
to combine the per-shank recordings back together.

```
preprocessed_recordings = []
for recording in split_recordings_dict.values():


    shifted_recording = spre.phase_shift(recording)

    filtered_recording = spre.bandpass_filter(shifted_recording)

    referenced_recording = spre.common_reference(filtered_recording)

    preprocessed_recordings.append(referenced_recording)

combined_preprocessed_recording = aggregate_channels(preprocessed_recordings)

```

It is strongly recommended to use the above structure to preprocess by group.
A discussion of the subtleties of the approach may be found in the below
Notes section for the interested reader.

# Preprocessing channel in depth

Preprocessing and aggregation of channels is very flexible. Under the hood,
`aggregate_channels` keeps track of when a recording was split. When `get_traces()`
is called, the preprocessing is still performed per-group, even though the
recording is now aggregated.

However, to ensure data is preprocessed by shank, the preprocessing step must be
applied per-group. For example, the below example will NOT preprocess by shank:

```
split_recording = recording.split_by("group")
split_recording_as_list = list(**split_recording.values())
combined_recording = aggregate_channels(split_recording_as_list)

# will NOT preprocess by channel group.
filtered_recording = common_reference(combined_recording)

```

Similarly, in the below example the first preprocessing step (bandpass filter)
would applied by group (although, this would have no effect in practice
as this preprocessing step is always performed per channel). However,
common referencing (which is effected by splitting by group) will
not be applied per group:

```
split_recording = recording.split_by("group")

filtered_recording = []
for recording in split_recording.values()
    filtered_recording.append(spre.bandpass_filtered(recording))

combined_recording = aggregate_channels(filtered_recording)

# As the recording has been combined, common referencing
# will NOT be applied per channel group.
referenced_recording = spre.common_reference(combined_recording).
```

Finally, it is not recommended to apply `aggregate_channels` more than once
as this will slow down `get_traces()` and may result in unpredictable behaviour.


Sorting a Recording by Channel Group
------------------------------------

Sorting a recording can be performed in two ways. One, is to split the
recording by group and use `run_sorter` (LINK) separately on each preprocessed
channel group.

Altearntively, SpikeInterface proves a conveniecne function XXX
to this.

Example

Done!
