How to Combine Recordings in SpikeInterface
===========================================

In this tutorial we will walk through combining multiple recording objects. This can happen when the probe is not
moved within the tissue, but an overall recording session is split into several subsessions (e.g. pre-stim, stim, post-stim).
Thus one may wish to combine these individual recording objects into one recording object before sorting.

**Why Combine?**

Spike sorters seek to sort spikes within a recording into groups of units. Thus if multiple :code:`Recording` objects have the
exact same probe location within tissue and are occurring continuously in time the units between the :code:`Recordings` will
be the same. But if we sort each recording separately the unit ids given by the sorter will not be the same between each
:code:`Sorting` and so we will need extensive post-processing to try to figure out which units are actually the same between
each :code:`Sorting`. By combining everything into one :code:`Recording` all spikes will be sorted into the same pool of units.

Combining recordings continuous in time
---------------------------------------

Some file formats (e.g. Intan) automatically create new files every minute or few minutes (with a setting that can be user
controlled). Other times an experimenter separates their recording for experimental reasons. SpikeInterface provides two
tools for bringing together these files into one :code:`Recording` object.

Concatenating Recordings
^^^^^^^^^^^^^^^^^^^^^^^^

First let's cover concatenating recordings together. This will generate a mono-segment recording object. Let's load a set of
Intan files.

.. code-block:: python

    import spikeinterface as si # this is only core
    import spikeinterface.extractors as se

    intan_rec_one = se.read_intan('./intan_example_01.rhd', stream_id='0') # 0 is the amplifier data for Intan
    intan_rec_two = se.read_intan('./intan_example_02.rhd', stream_id='0')

    print(intan_rec_one)

    """
    IntanRecordingExtractor 64 Channels - 30.0kHz - 1 segments - 1,800,000 samples - 60 seconds - uint16 dtype - xx
    """

    concatenated_recording = si.concatenate_recordings([intan_rec_one, intan_rec_two])

    print(concatenated_recording)

    """
    ConcatenatedRecordingExtractor 64 Channels - 30.0kHz - 1 segments - 3,600,000 samples - 120 seconds - uint16 dtype -xx
    """

If we know that we will deal with a lot of files we can actually work our way through a series of them relatively quickly by doing
the following:

.. code-block:: python

    list_of_files = ['file1.rhd', 'file2.rhd', 'file3.rhd', 'file4.rhd']
    list_of_recordings = []
    for file in list_of_files:
        list_of_recordings.append(se.read_intan(file, stream_id='0'))
    recording = si.concatenate_recordings(list_of_recordings)


Append Recordings
^^^^^^^^^^^^^^^^^

If you wish to keep each recording as a separate segment identity (e.g. if doing baseline, stim, poststim) you can use
:code:`append` instead of :code:`concatenate`.

#TODO Why would we really want to do this @everyone

If we use the same Intan files as above (:code:`intan_rec_one` and :code:`intan_rec_two`) we can see what happens if we
append them instead of concatenate them.

.. code-block:: python

    recording = si.append_recordings([intan_rec_one, intan_rec_two])

    print(recording)

    """
    AppendRecordingExtractor 64 Channels - 30.0khz - 2 segments - 3,600,000 samples - 120 seconds - xx
    """
