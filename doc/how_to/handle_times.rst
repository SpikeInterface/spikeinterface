How SpikeInterface handles time
=================================

Extracellular electrophysiology commonly involves synchronisation of events
across many timestreams. For example, an experiment may involve
displaying a stimuli to an animal and recording the stimuli-evoked
neuronal responses. It is critical that timings are represented
accurately so they may be properly synchronised during
analysis.

Below, we will explore the ways that SpikeInterface stores time
information and how you can use it in your analysis to ensure the timing
of your spike events are accurate.

A familiarity with the terms used in digital sampling (e.g. sampling
frequency) will be assumed below. If you are not familiar with these concepts,
please see the below dropdown for a refresher.

.. dropdown:: Digital Sampling

    What we are fundamentally interested in is recording the electrical
    waveforms found in the brain. In the real world, these are 'continuous'
    signals changing through time. To represent these real signals on our
    finite-memory computers, we must 'sample' the real-world continuous signals
    at discrete time points.

    When sampling our signal the fundamental question we need to ask is - how fast should
    we sample the continuous data? A natural approach is to sample our signal every $N$ seconds.
    For example, let's say we decide to sample our signal 4 times per second.
    This is known as the 'sampling frequency' (sometimes denoted $f_s$) and is expressed
    in Hertz (Hz) i.e. 'samples-per-second'.

    Another way of thinking about the same idea is to ask - how often are we
    sampling our signal? In our example, we are sampling the signal once every 0.25 seconds.
    This is known as the 'sampling step' (sometimes denoted $t_s$) and is the inverse of
    the sampling frequency.

    .. image:: handle_times_files/handle-times-sampling-image.png
       :alt: Image of continuous signal (1 second) with dots indicating samples collected at 0, 0.25, 0.5 and 0.75 seconds.
       :width: 400px
       :align: center

    In the real world, we will sample our signal much faster. For example, Neuropixels
    samples at 30 kHz (30,000 samples per second), with a
    sampling step of 1/30000 = 0.0003 seconds!

    Using computers, we typically represent time as a long array of numbers,
    here we refer to this as a 'time array' in seconds.
    For example, `[0, 0.25, 0.5, 0.75, ...]`.

------------------------------------------------------------------
An Overview of the possible Time representations in SpikeInterface
------------------------------------------------------------------

When you load a recording into SpikeInterface, it will be automatically
associated with a time array. Depending on your data format, this might
be loaded from metadata on your raw recording. If there is no time metadata
on your raw recording, the times will be generated based on your sampling
rate and number of samples.

**[TODO: concrete example of a datatype that loads time also on the neo side]**

You can use the :meth:`get_times() <spikeinterface.core.BaseRecording.get_times>`
method to inspect the time array associated with your recording.

.. code-block:: python

    import spikeinterface.full as si

    # Generate a recording for this example
    recording, sorting = si.generate_ground_truth_recording(durations=[10])

    print(f"number of samples: {recording.get_num_samples()}")
    print(f"sampling frequency: {recording.get_sampling_frequency()}"

    print(
        recording.get_times()
    )

Here, we see that as no time metadata is associated with the loaded recording,
a default time array is generated from the number of samples and sampling frequency.
The times starts at :math:`0` seconds and continues until :math:`10` seconds
(:math:`10 \cdot \text{sampling frequency}` samples) in steps of sampling step size,
:math:`\frac{1}{\text{sampling frequency}}`.

If timings are obtained from metadata during file loading, you may find that the first timepoint is
not zero, or the times may not be separated by exactly :math:`\frac{1}{\text{sampling frequency}}`
but may be irregular due to small drifts in sampling rate during acquisition (so called 'clock drift').

^^^^^^^^^^^^^^^^^^^^^^^
Shifting the start time
^^^^^^^^^^^^^^^^^^^^^^^

Having loaded your recording object and inspected the associated
times, you may want to change the start time of your recording.

For example, your recording may not have metadata attached and you
want to shift the default times to start at the
true (real world) start time of the recording, or relative to some
other event (e.g. behavioural trial start time).

Alternatively, you may want to change the start time of the metadata-loaded
recording for similar reasons.

To do this, you can use the `shift_start_time()` function to shift
the first timepoint of the recording. Shifting by a positive value will
increase the start time, while shifting by a negative value will decrease
the start time.

.. code-block:: python

    recording.shift_start_time(100.15)

    print(recording.get_times())  # time now start at 100.15 seconds

    recording.shift_start_time(-50.15)

    print(recording.get_times())  # time now start at 50 seconds

**TODO: link to new function and test when other PR is merged**


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Setting time vector changes spike times
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we sort out recording, the spike times will reflect the times
set on the recording. In our case, because we already have the
sorting object based on the default times, we will set the new
recording object on the sorting.

.. code-block:: python

    unit_id_to_show = sorting.unid_ids[0]

    spike_times_orig = sorting.get_unit_spike_train(unit_id_to_show, return_times=True)

    sorting.register_recording(recording)

    spike_times_new = sorting.get_unit_spike_train(unit_id_to_show, return_times=True)


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Manually setting a time vector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to manualyl set an entire time vector on your recording.
This might be useful in case you have the true sample timestamps of your
recording but these were not automatically loaded from metadata.

You can associate any time vector with your recording (as long as it contains
as many samples as the recording itself) using
:meth:`set_times() <spikeinterface.core.BaseRecording.set_times>`

.. code-block:: python

    times = np.linspace(0, 10, recording.get_num_samples()))
    offset = np.cumsum(
        np.linspace(0, 0.1, recording.get_num_samples())
    )
    true_times = times + offset

    recording.set_times(true_times)

    recording.get_times()

.. warning::

   In the case of regularly spaced time vectors, it is recommended
   to shift the default times rather than set your own time vector,
   as this will require more memory under the hood.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Retrieving timepoints from sample index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SpikeInterface provides two convenience methods for obtaining the
timepoint in seconds given an index of the time array.

Use
:meth:`time_to_sample_index() <spikeinterface.core.BaseRecording.time_to_sample_index>`
to go from time to the sample index:

.. code-block:: python

    sample_index = recording.time_to_sample_index(5.0)

    print(sample_index)


and
:meth:`sample_index_to_to_time() <spikeinterface.core.BaseRecording.sample_index_to_to_time>`
to can retrieve the index given a timepoint:


.. code-block:: python

    timepoint = recording.sample_index_to_to_time(125000)

    print(timepoint)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Aligning events across timestreams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The alignment of electrophysiology recording time to other data streams (e.g. behaviour)
is an important step in electrophysiology analysis. To achieve this,it is common to acquire
a synchronisation ('sync') pulse on an additional channel.

At present SpikeInterface does not include features for time-alignment,
but some useful articles on how to approach this can be found on the following pages:

* `SpikeGLX <https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#procedure-to-calibrate-sample-rates>`_,
* `OpenEphys <https://open-ephys.github.io/gui-docs/Tutorials/Data-Synchronization.html>`_,
* `NWB <https://neuroconv.readthedocs.io/en/main/user_guide/temporal_alignment.html>`_
