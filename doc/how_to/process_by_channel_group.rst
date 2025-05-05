Process a recording by channel group
====================================

In this tutorial, we will walk through how to preprocess and sort a recording
separately for *channel groups*. A channel group is a subset of channels grouped by some
feature - for example a multi-shank Neuropixels recording in which the channels
are grouped by shank.

**Why preprocess by channel group?**

Certain preprocessing steps depend on the spatial arrangement of the channels.
For example, common average referencing (CAR) averages over channels (separately for each time point)
and subtracts the average. In such a scenario it may make sense to group channels so that
this averaging is performed only over spatially close channel groups.

**Why sort by channel group?**

When sorting, we may want to completely separately channel groups so we can
consider their signals in isolation. If recording from a long
silicon probe, we might want to sort different brain areas separately,
for example using a different sorter for the hippocampus, the thalamus, or the cerebellum.


Splitting a Recording by Channel Group
--------------------------------------

In this example, we create a 384-channel recording with 4 shanks. However this could
be any recording in which the channel are grouped in some way, for example
a multi-tetrode recording with channel groups representing the channels on each individual tetrodes.

First, let's import the parts of SpikeInterface we need into Python, and generate our toy recording:

.. code-block:: python

    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre
    from spikeinterface import aggregate_channels
    from probeinterface import generate_tetrode, ProbeGroup
    import numpy as np

    # Create a toy 384 channel recording with 4 shanks (each shank contain 96 channels)
    recording, _ = se.toy_example(duration=[1.00], num_segments=1, num_channels=384)
    four_shank_groupings = np.repeat([0, 1, 2, 3], 96)
    recording.set_property("group", four_shank_groupings)

    print(recording.get_channel_groups())
    """
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    """

We can split a recording into multiple recordings, one for each channel group, with the :py:func:`~split_by` method.

.. code-block:: python

    split_recording_dict = recording.split_by("group")

Splitting a recording by channel group returns a dictionary containing separate recordings, one for each channel group:

.. code-block:: python

    print(split_recording_dict)
    """
    {0: ChannelSliceRecording: 96 channels - 30.0kHz - 1 segments - 30,000 samples - 1.00s - float32 dtype
                           10.99 MiB, 1: ChannelSliceRecording: 96 channels - 30.0kHz - 1 segments - 30,000 samples - 1.00s - float32 dtype
                           10.99 MiB, 2: ChannelSliceRecording: 96 channels - 30.0kHz - 1 segments - 30,000 samples - 1.00s - float32 dtype
                           10.99 MiB, 3: ChannelSliceRecording: 96 channels - 30.0kHz - 1 segments - 30,000 samples - 1.00s - float32 dtype
                           10.99 MiB}
    """

Preprocessing a Recording by Channel Group
------------------------------------------

If a preprocessing function is given a dictionary of recordings, it will apply the preprocessing
seperately to each recording in the dict, and return a dictionary of preprocessed recordings.
Hence we can pass the ``split_recording_dict`` in the same way as we would pass a single recording
to any preprocessing function.

.. code-block:: python

    shifted_recordings = spre.phase_shift(split_recording_dict)
    filtered_recording = spre.bandpass_filter(shifted_recording)
    referenced_recording = spre.common_reference(filtered_recording)

We can then aggregate the recordings back together using the ``aggregate_channels`` function

.. code-block:: python

    combined_preprocessed_recording = aggregate_channels(referenced_recording)

Now, when ``combined_preprocessed_recording`` is used in sorting, plotting, or whenever
calling its :py:func:`~get_traces` method, the data will have been
preprocessed separately per-channel group (then concatenated
back together under the hood).

.. note::

    The splitting and aggregation of channels for preprocessing is flexible.
    Under the hood, :py:func:`~aggregate_channels` keeps track of when a recording was split. When
    :py:func:`~get_traces` is called, the preprocessing is still performed per-group,
    even though the recording is now aggregated.

    To ensure data is preprocessed by channel group, the preprocessing step must be
    applied separately to each split channel group recording.
    For example, the below example will NOT preprocess by channel group:

    .. code-block:: python

        split_recording = recording.split_by("group")
        split_recording_as_list = list(**split_recording.values())
        combined_recording = aggregate_channels(split_recording_as_list)

        # will NOT preprocess by channel group.
        filtered_recording = common_reference(combined_recording)


    In general, it is not recommended to apply :py:func:`~aggregate_channels` more than once.
    This will slow down :py:func:`~get_traces` calls and may result in unpredictable behaviour.


Sorting a Recording by Channel Group
------------------------------------

We can also sort a recording for each channel group separately. It is not necessary to preprocess
a recording by channel group in order to sort by channel group.

There are two ways to sort a recording by channel group. First, we can split the preprocessed
recording (or, if it was already split during preprocessing as above, skip the :py:func:`~aggregate_channels` step
directly use the :py:func:`~split_recording_dict`).

**Option 1: Manual splitting**

In this example, similar to above we loop over all preprocessed recordings that
are grouped by channel, and apply the sorting separately. We store the
sorting objects in a dictionary for later use.

.. code-block:: python

    split_preprocessed_recording = preprocessed_recording.split_by("group")

    sortings = {}
    for group, sub_recording in split_preprocessed_recording.items():
        sorting = run_sorter(
            sorter_name='kilosort2',
            recording=split_preprocessed_recording,
            output_folder=f"folder_KS2_group{group}"
            )
        sortings[group] = sorting

**Option 2 : Automatic splitting**

Alternatively, SpikeInterface provides a convenience function to sort the recording by property:

.. code-block:: python

     aggregate_sorting = run_sorter_by_property(
        sorter_name='kilosort2',
        recording=preprocessed_recording,
        grouping_property='group',
        working_folder='working_path'
    )
