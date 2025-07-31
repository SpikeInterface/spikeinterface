.. _process_by_group:

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
    good_channels_recording = spre.detect_and_remove_bad_channels(filtered_recording)

If needed, we could aggregate the recordings back together using the ``aggregate_channels`` function.
Note: you do not need to do this to sort the data (see :ref:`sorting-by-channel-group`).

.. code-block:: python

    combined_preprocessed_recording = aggregate_channels(good_channels_recording)

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

.. _sorting-by-channel-group:

Sorting a Recording by Channel Group
------------------------------------

We can also sort a recording for each channel group separately. It is not necessary to preprocess
a recording by channel group in order to sort by channel group.

There are two ways to sort a recording by channel group. First, we can pass a dictionary to the
``run_sorter`` function. Since the preprocessing-by-group method above returns a dict, we can
simply pass this output. Alternatively, for more control, we can loop over the recordings
ourselves.

**Option 1 : Automatic splitting (Recommended)**

Simply pass the split recording to the ``run_sorter`` function, as if it was a non-split recording.
This will return a dict of sortings, with the same keys as the dict of recordings that were
passed to ``run_sorter``.

.. code-block:: python

    split_recording = raw_recording.split_by("group")
    # is a dict of recordings

    # do preprocessing if needed
    pp_recording = spre.bandpass_filter(split_recording)

    dict_of_sortings = run_sorter(
        sorter_name='kilosort4',
        recording=pp_recording,
        folder='my_kilosort4_sorting'
    )


**Option 2: Manual splitting**

In this example, we loop over all preprocessed recordings that
are grouped by channel, and apply the sorting separately. We store the
sorting objects in a dictionary for later use.

You might do this if you want extra control e.g. to apply bespoke steps
to different groups.

.. code-block:: python

    split_preprocessed_recording = preprocessed_recording.split_by("group")

    sortings = {}
    for group, sub_recording in split_preprocessed_recording.items():
        sorting = run_sorter(
            sorter_name='kilosort2',
            recording=split_preprocessed_recording,
            folder=f"folder_KS2_group{group}"
            )
        sortings[group] = sorting


Creating a SortingAnalyzer by Channel Group
-------------------------------------------

The code above generates a dictionary of recording objects and a dictionary of sorting objects.
When making a :ref:`SortingAnalyzer <modules/core:SortingAnalyzer>`, we can pass these dictionaries and
a single analyzer will be created, with the recordings and sortings appropriately aggregated.

The dictionary of recordings and dictionary of sortings must have the same keys. E.g. if you
use ``split_by("group")``, the keys of your dict of recordings will be the values of the ``group``
property of the recording. Then the dict of sortings should also have these keys.
Note that if you use the internal functions, like we do in the code-block below, you don't need to
keep track of keys yourself. SpikeInterface will do this for you automatically.

The code for create ``SortingAnalyzer`` from dicts of recordings and sortings is very similar to that for
creating a sorting analyzer from a single recording and sorting:

.. code-block:: python

    dict_of_recordings = preprocessed_recording.split_by("group")
    dict_of_sortings = run_sorter(sorter_name="mountainsort5", recording = dict_of_recordings)

    analyzer = create_sorting_analyzer(sorting=dict_of_sortings, recording=dict_of_recordings)


The code above creates a *single* sorting analyzer called :code:`analyzer`. You can select the units
from one of the "group"s as follows:

.. code-block:: python

    aggretation_keys = analyzer.get_sorting_property("aggregation_key")
    unit_ids_group_0 = analyzer.unit_ids[aggretation_keys == 0]
    group_0_analzyer = analyzer.select_units(unit_ids = unit_ids_group_0)
