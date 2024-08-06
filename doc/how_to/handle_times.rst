How SpikeInterface handles Time
=================================

Extracellular electrophysiology commonly involves synchronisation of events
across many timestreams. For example, an experiment may involve
displaying a stimuli to an animal and recording the stimuli-evoked
neuronal responses. It is critical that timings is represented in
a clear way across data streams so they may be properly syncronised during
analysis.

Below, we will explore the ways that SpikeInterface represents time
information and how you can use it in your analysis to ensure the timing
of your spike events is represented faithfully. The ways that will
be explored are **providing no times**, **providing start times**
and **providing the full time array**.

A familiarity with the terms used in digital sampling (e.g. sampling
frequency) will be assumed below. If you are not familiar with these concepts,
please see the below dropdown for a refresher.

.. dropdown:: Digital Sampling

    What we are fundamentally interested in is recording the electrical
    waveforms found in the brain. In the real world, these are 'continuous'
    signals changing through time. To represent these real signals on our
    finite-memory computers, we must 'sample' the real-world continuous signals
    at discrete time points.

    When sampling our signal the fundamental question we need to ask is - how fast?
    A natural approach is to sample our signal every X seconds. For the sake of example,
    let's say we decide to sample our signal 4 times per second. This is known as the
    'sampling frequency' (sometimes denoted $f_s$) and is expressed in Hertz (Hz) i.e.
    'samples-per-second'.

    Another way of thinking about the same idea is to ask - how often are we
    sampling our signal? In our example, we are sampling the signal once every 0.25 seconds.
    This is known as the 'sampling step' (sometimes denoted $t_s$) and is the inverse of
    the sampling frequency.

    [PICTURE]

    In the real world, we will sample our signal much faster. For example, Neuropixels
    samples at 30 kHz (30,000 samples per second), with a
    sampling step of 1/30000 = 0.0003 seconds!

    On our computer, we typically repesent time as a long array of numbers.
    For example, XXXX. Often, it is easier to access this array using 'indices'
    rather than the raw time units. MORE ON THIS. It is useful to remember
    these below quick conversion between time and indices that are often
    used in code:

    ``time * sampling_frequency = index`` and
    ``index * sampling_step = index * (1/sampling_frequency = time``.

------------------------------------------------------------------
An Overview of the possible Time representations in SpikeInterface
------------------------------------------------------------------

When loading a raw recording, SpikeInterface only has access to the
sampling frequency and raw data samples of the data. Only in rare
cases are the time-stamps of extracellular electrophysiological data
are stored alongside the recordings and can be directly loaded
(TODO: is any of that true?)

In some applications, having the exact time is very important.
For example, NeuroPixels. In this case want to include the
exact times. How do you do this? where is this information stored?

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Providing no times to spikeinterface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this case, all times used will be generated from the sample index
and sampling frequency. For example, if a spike peak is detected at
index 300 of the raw data, and the sampling frequency is 30 kHz,
the peak time will be 300 * (1/30000) = 0.01 s.

Note that time is represented in spikeinterface in XXXX datatype, and so
resolution will be this. For example, a spike at index 1000 of the raw
data will be represented as 1000 * (1/30000) = 0.03333XXX seconds.

For a multi-segment recording, each segment will be fixed to start at
0 seconds (TODO: is this true)? If there is a peak at index 300 in
segment one and index in the peak at index 300 in segment two, both
segments action potential peaks will be given a spike time of 0.1 s,
and their segment tracked to distinguish them. TODO: CHECK

# TODO: yes follow a code example!
```sorting.get_unit_spike_train(0, return_times=True, segment_index=1)```

^^^^^^^^^^^^^^^^^^^^^^^^^
Providing the start time
^^^^^^^^^^^^^^^^^^^^^^^^^

### TODO :::: This section make no sense because you can't set
`t_start` directly as far as I can tell, it is only available on
a few extractors.

Alternatively, the start-time of each segment can be provided to
spikeinterface. The same method (index * (1 / sampling_frequency))
will be used to determine event spike, but now an initial offset
will be added using the `t_start` variable. TODO: reword, less confusing.

For example, imagine you have two recording sessions, recorded at
30 kHz sampling frequency, both 30 minutes long. You record one session,
wait 10 minutes, then record another session. In this example the will imagine
these two sessions are loaded into spikeinterface as two segments on a
single recording (however, the process would be the same if you had loaded
the sessions separately into two recording objects).

If you wanted to represent the sessions as starting relative to the
first session, and keeping the true timings of the recording, you can
do:

```
recording.select_segment(1).set_times(XXX)
```

Alternatively, if you wanted to ignore the 10 minute gap, you could do:

```
recording.select_segment(1).set_times(XXXX)
```
``` print spike times```
sorting.get_unit_spike_train(0, return_times=True, segment_index=1)
# TODO: it says use time vector but this doesn't return times.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Providing the full time array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, may want to get full time array. Where do you get this from?
Loaded - some more information on where from

`recording.has_time_vector()`.

we can set times e.g. as above.

1) set the times
2) print the times. Note how they are different to the above case!

Note that for some extractors, e.g. read_openephys you can load the
syncrhonized timestamps directly (load_sync_timestamps param). It would be
important to mention! Section on when files are loaded autoamticaly!

--------------------------
Accessing time information
--------------------------

Cover the The two time array functions.

`sample_index_to_time`
