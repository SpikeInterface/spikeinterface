Combine Recordings in SpikeInterface
====================================

In this tutorial we will walk through combining multiple recording objects. Sometimes this occurs due to hardware
settings (e.g. Intan software has a default setting of new files every 1 minute) or the experimenter decides to
split their recording into multiple files for different experimental conditions. If the probe has not been moved,
however, then during sorting it would likely make sense to combine these individual reocrding objects into one
recording object.

**Why Combine?**

Combining your data into a single recording allows you to have consistent labels (:code:`unit_ids`) across the whole recording.

Spike sorters seek to sort spikes within a recording into groups of units. Thus if multiple :code:`Recording` objects have the
exact same probe location within some tissue and are occurring continuously in time the units between the :code:`Recordings` will
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
    IntanRecordingExtractor: 64 Channels - 30.0kHz - 1 segments - 1,800,000 samples
                             60.00s (1.00 minutes) uint16 dtype - 219.73 MiB
    """

    print(intan_rec_two)

    """
    IntanRecordingExtractor: 64 Channels - 30.0Khz - 1 segments - 1,800,000 samples
                             60.00s (1.00 minutes) - uin16 dtype - 219.73 MiB
    """

    concatenated_recording = si.concatenate_recordings([intan_rec_one, intan_rec_two])

    print(concatenated_recording)

    """
    ConcatenateSegmentRecording: 64 Channels - 30.0kHz - 1 segments - 3,600,000 samples
                                 120.00s (2.00 minutes) - uint16 dtype - 429.47 MiB
    """

As we can see if we take the sample number (1,800,000) or time (60.00s) of each recording and add them together
we get the concatenated sample number (3,600,000) and time (120.00s).

If we know that we will deal with a lot of files we can actually work our way through a series of them relatively quickly by doing
the following:

.. code-block:: python


    # make sure to use the appropriate paths for adapting to your own pipeline
    # adapt the extractor for your desired file format as well
    list_of_files = ['file1.rhd', 'file2.rhd', 'file3.rhd', 'file4.rhd']
    list_of_recordings = []
    for file in list_of_files:
        list_of_recordings.append(se.read_intan(file, stream_id='0'))
    recording = si.concatenate_recordings(list_of_recordings)


Append Recordings
^^^^^^^^^^^^^^^^^

If you wish to keep each recording as a separate segment identity (e.g. if doing baseline, stim, poststim) you can use
:code:`append` instead of :code:`concatenate`. This has the benefit of allowing you to keep different parts of data
separate, but it is important to note that not all sorters can handle multi-segment objects.

If we use the same Intan files as above (:code:`intan_rec_one` and :code:`intan_rec_two`) we can see what happens if we
append them instead of concatenate them.

.. code-block:: python

    recording = si.append_recordings([intan_rec_one, intan_rec_two])

    print(recording)

    """
    AppendSegmentRecording: 64 Channels - 30.0khz - 2 segments - 3,600,000 samples
                            120.00s (2.00 minutes) - uint16 dtype - 439.47 MiB
    Segments:
    Samples:    1,800,000 | 1,800,000
    Durations:  60.00s (1.00 minutes) | 60.00s (1.00 minutes)
    Memory:     219.17 MiB | 219.17 MiB
    """

In this case we see that our recording has two segments instead of one segment. The total sample number (3,600,00)
and the total time (120.00s), however are still the same as our example above. We can see that each segment is
exactly equivalent to one of the :code:`IntanRecordingExtractor`'s above.


Pitfalls
--------

It's important to remember that these operations are directional. So,

.. code-block:: python

    recording_forward = si.concantenate_recordings([intan_rec_one, intan_rec_two])
    recording_backward = si.concantenate_recordings([intan_rec_two, intan_rec_one])

    recording_forward != recording_backward


This is important because your spike times will be relative to the start of your recording.
