Core module
===========

Overview
--------

The :py:mod:`spikeinterface.core` module provides the basic classes and tools of the SpikeInterface ecosystem.

Several Base classes are implemented here and inherited throughout the SI code-base.
The core classes are: :py:class:`~spikeinterface.core.BaseRecording` (for raw data),
:py:class:`~spikeinterface.core.BaseSorting` (for spike-sorted data), and
:py:class:`~spikeinterface.core.WaveformExtractor` (for waveform extraction and postprocessing).

There are additional classes to allow to retrieve events (:py:class:`~spikeinterface.core.BaseEvent`) and to
handle unsorted waveform cutouts, or *snippets*, which are recorded by some acquisition systems
(:py:class:`~spikeinterface.core.BaseSnippets`).


All classes support:
  * metadata handling
  * data on-demand (lazy loading)
  * multiple segments, where each segment is a contiguous piece of data (recording, sorting, events).


Recording
---------

The :py:class:`~spikeinterface.core.BaseRecording` class serves as the basis of all
:code:`Recording` classes.
It interfaces with the raw traces and has the following features:

* retrieve raw and scaled traces from each segment
* keep info about channel_ids VS channel indices
* handle probe information
* store channel properties
* store object annotations
* enable grouping, splitting, and slicing
* handle time information

Here we assume :code:`recording` is a :py:class:`~spikeinterface.core.BaseRecording` object
with 16 channels:

.. code-block:: python

    channel_ids = recording.channel_ids
    num_channels = recording.get_num_channels()
    sampling_frequency = recording.sampling_frequency

    # get number of samples/duration
    num_samples_segment = recording.get_num_samples(segment_index=0)
    ### NOTE ###
    # 'segment_index' is required for multi-segment objects
    num_total_samples = recording.get_total_samples()
    total_duration = recording.get_total_duration()

    # retrieve raw traces between frames 100 and 200
    traces = recording.get_traces(start_frame=100, end_frame=200, segment_index=0)
    # retrieve raw traces only for the first 4 channels
    traces_slice = recording.get_traces(start_frame=100, end_frame=200, segment_index=0,
                                        channel_ids=channel_ids[:4])
    # retrieve traces after scaling to uV
    # (requires 'gain_to_uV' and 'offset_to_uV' properties)
    traces_uV = recording.get_traces(start_frame=100, end_frame=200, segment_index=0,
                                     return_scaled=True)
    # set/get a new channel property (e.g. "quality")
    recording.set_property(key="quality", values=["good"] * num_channels)
    quality_values = recording.get_property("quality")
    # get all available properties
    property_keys = recording.get_property_keys()

    # set/get an annotation
    recording.annotate(date="Recording acquired today")
    recording.get_annotation(key="date")

    # get new recording with the first 10s of the traces
    recording_slice_frames = recording.frame_slice(start_frame=0,
                                                   end_frame=int(10*sampling_frequency))
    # get new recording with the first 4 channels
    recording_slice_chans = recording.channel_slice(channel_ids=channel_ids[:4])
    # remove last two channels
    recording_rm_chans = recording.remove_channels(channel_ids=channel_ids[-2:])

    # set channel grouping (assume we have 4 groups of 4 channels, e.g. tetrodes)
    groups = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    recording.set_channel_groups(groups)
    # split by property
    recording_by_group = recording.split_by("group")
    # 'recording_by_group' is a dict with group as keys (0,1,2,3) and channel
    # sliced recordings as values

    # set times (for synchronization) - assume our times start at 300 seconds
    timestamps = np.arange(num_samples) / sampling_frequency + 300
    recording.set_times(times=timestamps, segment_index=0)

**Note**:
Raw data formats often store data as integer values for memory efficiency. To give these integers meaningful physical units (uV), you can apply a gain and an offset.
Many devices have their own gains and offsets necessary to convert their data and these values are handled by SpikeInterface for its extractors. This
is triggered by the :code:`return_scaled` parameter in :code:`get_traces()`, (see above example), which will return the traces in uV.


Sorting
-------

The :py:class:`~spikeinterface.core.BaseSorting` class serves as the basis for all :code:`Sorting` classes.
It interfaces with a spike-sorted output and has the following features:

* retrieve spike trains of different units
* keep info about unit_ids VS unit indices
* store channel properties
* store object annotations
* enable selection of sub-units
* handle time information


Here we assume :code:`sorting` is a :py:class:`~spikeinterface.core.BaseSorting` object
with 10 units:

.. code-block:: python

    unit_ids = sorting.unit_ids
    num_channels = sorting.get_num_units()
    sampling_frequency = sorting.sampling_frequency

    # retrieve spike trains for a unit (returned as sample indices)
    unit0 = unit_ids[0]
    spike_train = sorting.get_unit_spike_train(unit_id=unit0, segment_index=0)
    # retrieve spikes between frames 100 and 200
    spike_train_slice = sorting.get_unit_spike_train(unit_id=unit0,
                                                     start_frame=100, end_frame=200,
                                                     segment_index=0)
    ### NOTE ###
    # 'segment_index' is required for multi-segment objects

    # set/get a new unit property (e.g. "quality")
    sorting.set_property(key="quality", values=["good"] * num_units)
    quality_values = sorting.get_property("quality")
    # get all available properties
    property_keys = sorting.get_property_keys()

    # set/get an annotation
    sorting.annotate(date="Spike sorted today")
    sorting.get_annotation(key="date")

    # get new sorting within the first 10s of the spike trains
    sorting_slice_frames = sorting.frame_slice(start_frame=0,
                                               end_frame=int(10*sampling_frequency))
    # get new sorting with only the first 4 units
    sorting_select_units = sorting.select_units(unit_ids=unit_ids[:4])

    # register 'recording' from the previous example and get the spike trains in seconds
    sorting.register_recording(recording=recording)
    spike_train_s = sorting.get_unit_spike_train(unit_id=unit0, segment_index=0,
                                                 return_times=True)
    ### NOTE ###
    # When running spike sorting in SpikeInterface, the recording is  automatically registered. If
    # times are not set, the samples are divided by the sampling frequency


Internally, any sorting object can construct 2 internal caches:
  1. a list (per segment) of dict (per unit) of numpy.array. This cache is useful when accessing spike trains on a unit
     per unit basis across segments.
  2. a unique numpy.array with structured dtype aka "spikes vector". This is useful for processing by small chunks of
     time, like for extracting amplitudes from a recording.


WaveformExtractor
-----------------

The :py:class:`~spikeinterface.core.WaveformExtractor` class is the core object to combine a
:py:class:`~spikeinterface.core.BaseRecording` and a :py:class:`~spikeinterface.core.BaseSorting` object.
Waveforms are very important for additional analyses, and the basis of several postprocessing and quality metrics
computations.

The :py:class:`~spikeinterface.core.WaveformExtractor` allows us to:

* extract waveforms
* sub-sample spikes for waveform extraction
* compute templates (i.e. average extracellular waveforms) with different modes
* save waveforms in a folder (in numpy / `Zarr <https://zarr.readthedocs.io/en/stable/tutorial.html>`_) for easy retrieval
* save sparse waveforms or *sparsify* dense waveforms
* select units and associated waveforms

In the default format (:code:`mode='folder'`) waveforms are saved to a folder structure with waveforms as
:code:`.npy` files.
In addition, waveforms can also be extracted in-memory for fast computations (:code:`mode='memory'`).
Note that this mode can quickly fill up your RAM... Use it wisely!
Finally, an existing :py:class:`~spikeinterface.core.WaveformExtractor` can be saved also in :code:`zarr` format.

.. code-block:: python

    # extract dense waveforms on 500 spikes per unit
    we = extract_waveforms(recording=recording,
                           sorting=sorting,
                           sparse=False,
                           folder="waveforms",
                           max_spikes_per_unit=500
                           overwrite=True)
    # same, but with parallel processing! (1s chunks processed by 8 jobs)
    job_kwargs = dict(n_jobs=8, chunk_duration="1s")
    we = extract_waveforms(recording=recording,
                           sorting=sorting,
                           sparse=False,
                           folder="waveforms_parallel",
                           max_spikes_per_unit=500,
                           overwrite=True,
                           **job_kwargs)
    # same, but in-memory
    we_mem = extract_waveforms(recording=recording,
                               sorting=sorting,
                               sparse=False,
                               folder=None,
                               mode="memory",
                               max_spikes_per_unit=500,
                               **job_kwargs)

    # load pre-computed waveforms
    we_loaded = load_waveforms(folder="waveforms")

    # retrieve waveforms and templates for a unit
    waveforms0 = we.get_waveforms(unit_id=unit0)
    template0 = we.get_template(unit_id=unit0)

    # compute template standard deviations (average is computed by default)
    # (this can also be done within the 'extract_waveforms')
    we.precompute_templates(modes=("std",))

    # retrieve all template means and standard deviations
    template_means = we.get_all_templates(mode="average")
    template_stds = we.get_all_templates(mode="std")

    # save to Zarr
    we_zarr = we.save(folder="waveforms_zarr", format="zarr")

    # extract sparse waveforms (see Sparsity section)
    # this will use 50 spikes per unit to estimate the sparsity within a 40um radius from that unit
    we_sparse = extract_waveforms(recording=recording,
                                  sorting=sorting,
                                  folder="waveforms_sparse",
                                  max_spikes_per_unit=500,
                                  method="radius",
                                  radius_um=40,
                                  num_spikes_for_sparsity=50)


**IMPORTANT:** to load a waveform extractor object from disk, it needs to be able to reload the associated
:code:`sorting` object (the :code:`recording` is optional, using :code:`with_recording=False`).
In order to make a waveform folder portable (e.g. copied to another location or machine), one can do:

.. code-block:: python

    # create a "processed" folder
    processed_folder = Path("processed")

    # save the sorting object in the "processed" folder
    sorting = sorting.save(folder=processed_folder / "sorting")
    # extract waveforms using relative paths
    we = extract_waveforms(recording=recording,
                           sorting=sorting,
                           folder=processed_folder / "waveforms",
                           use_relative_path=True)
    # the "processed" folder is now portable, and the waveform extractor can be reloaded
    # from a different location/machine (without loading the recording)
    we_loaded = si.load_waveforms(folder=processed_folder / "waveforms",
                                  with_recording=False)


Event
-----

The :py:class:`~spikeinterface.core.BaseEvent` class serves as basis for all :code:`Event` classes.
It allows one to retrieve events and epochs (e.g. TTL pulses).
Internally, events are represented as numpy arrays with a structured dtype. The structured dtype
must contain the :code:`time` field, which represents the event times in seconds. Other fields are
optional.

Here we assume :code:`event` is a :py:class:`~spikeinterface.core.BaseEvent` object
with events from two channels:

.. code-block:: python

    channel_ids = event.channel_ids
    num_channels = event.get_num_channels()
    # get structured dtype for the first channel
    event_dtype = event.get_dtype(channel_ids[0])
    print(event_dtype)
    # >>> dtype([('time', '<f8'), ('duration', '<f8'), ('label', '<U100')])

    # retrieve events (with structured dtype)
    events = event.get_events(channel_id=channel_ids[0], segment_index=0)
    # retrieve event times
    event_times = event.get_event_times(channel_id=channel_ids[0], segment_index=0)
    ### NOTE ###
    # 'segment_index' is required for multi-segment objects


Snippets
--------

The :py:class:`~spikeinterface.core.BaseSnippets` class serves as basis for all :code:`Snippets`
classes (currently only :py:class:`~spikeinterface.core.NumpySnippets` and
:code:`WaveClusSnippetsExtractor` are implemented).

It represents unsorted waveform cutouts. Some acquisition systems, in fact, allow users to set a
threshold and only record the times at which a peak was detected and the waveform cut out around
the peak.

**NOTE**: while we support this class (mainly for legacy formats), this approach is a bad practice
and is highly discouraged! Most modern spike sorters, in fact, require the raw traces to perform
template matching to recover spikes!

Here we assume :code:`snippets` is a :py:class:`~spikeinterface.core.BaseSnippets` object
with 16 channels:

.. code-block:: python

    channel_ids = snippets.channel_ids
    num_channels = snippets.get_num_channels()
    # retrieve number of snippets
    num_snippets = snippets.get_num_snippets(segment_index=0)
    ### NOTE ###
    # 'segment_index' is required for multi-segment objects
    # retrieve total number of snippets across segments
    total_snippets = snippets.get_total_snippets()

    # retrieve snippet size
    nbefore = snippets.nbefore # samples before peak
    nsamples_per_snippet = snippets.snippet_len # total
    nafter = nsamples_per_snippet - nbefore # samples after peak

    # retrieve sample/frame indices
    frames = snippets.get_frames(segment_index=0)
    # retrieve snippet cutouts
    snippet_cutouts = snippets.get_snippets(segment_index=0)
    # retrieve snippet cutouts on first 4 channels
    snippet_cutouts_slice = snippets.get_snippets(channel_ids=channel_ids[:4],
                                                  segment_index=0)


Handling probes
---------------

In order to handle probe information, SpikeInterface relies on the
`probeinterface <https://probeinterface.readthedocs.io/en/main/>`_ package.
Either a :py:class:`~probeinterface.Probe` or a  :py:class:`~probeinterface.ProbeGroup` object can
be attached to a recording and it loads probe information (particularly channel locations and
sometimes groups).
ProbeInterface also has a library of available probes, so that you can download
and attach an existing probe to a recording with a few lines of code. When a probe is attached to
a recording, the :code:`location` property is automatically set. In addition, the
:code:`contact_vector` property will carry detailed information of the probe design.


Here we assume that :code:`recording` has 64 channels and it has been recorded by a
`ASSY-156-P-1 <https://gin.g-node.org/spikeinterface/probeinterface_library/src/master/cambridgeneurotech/ASSY-156-P-1/ASSY-156-P-1.png>`_ probe from
`Cambridge Neurotech <https://www.cambridgeneurotech.com/>`_ and wired via an Intan RHD2164 chip to the acquisition device.
The probe has 4 shanks, which can be loaded as separate groups (and spike sorted separately):

.. code-block:: python

    import probeinterface as pi

    # download probe
    probe = pi.get_probe(manufacturer='cambridgeneurotech', probe_name='ASSY-156-P-1')
    # add wiring
    probe.wiring_to_device('ASSY-156>RHD2164')

    # set probe
    recording_w_probe = recording.set_probe(probe)
    # set probe with group info and return a new recording object
    recording_w_probe = recording.set_probe(probe, group_mode="by_shank")
    # set probe in place, ie, modify the current recording
    recording.set_probe(probe, group_mode="by_shank", in_place=True)

    # retrieve probe
    probe_from_recording = recording.get_probe()
    # retrieve channel locations
    locations = recording.get_channel_locations()
    # equivalent to recording.get_property("location")

Probe information is automatically propagated in SpikeInterface, for example when slicing a recording by channels or
applying preprocessing.

Note that several :code:`read_***` functions in the :py:mod:`~spikeinterface.extractors` module
automatically load the probe from the files (including, SpikeGLX, Open Ephys - only NPIX plugin, Maxwell, Biocam,
and MEArec).


Sparsity
--------

In several cases, it is not necessary to have waveforms on all channels. This is especially true for high-density
probes, such as Neuropixels, because the waveforms of a unit will only appear on a small set of channels.
Sparsity is defined as the subset of channels on which waveforms (and related information) are defined. Of course,
sparsity is not global, but it is unit-specific.

**NOTE** As of version :code:`0.99.0` the default for a :code:`extract_waveforms()` has :code:`sparse=True`, i.e. every :code:`waveform_extractor`
will be sparse by default. Thus for users that wish to have dense waveforms they must set :code:`sparse=False`. Keyword arguments
can still be input into the :code:`extract_wavforms()` to generate the desired sparsity as explained below.

Sparsity can be computed from a :py:class:`~spikeinterface.core.WaveformExtractor` object with the
:py:func:`~spikeinterface.core.compute_sparsity` function:

.. code-block:: python

    # in this case 'we' should be a dense waveform_extractor
    sparsity = compute_sparsity(we, method="radius", radius_um=40)

The returned :code:`sparsity` is a :py:class:`~spikeinterface.core.ChannelSparsity` object, which has convenient
methods to access the sparsity information in several ways:

* | :code:`sparsity.unit_id_to_channel_ids` returns a dictionary with unit ids as keys and the list of associated
  | channel_ids as values
* | :code:`sparsity.unit_id_to_channel_indices` returns a similar dictionary, but instead with channel indices as
  | values (which can be used to slice arrays)

There are several methods to compute sparsity, including:

* | :code:`method="radius"`: selects the channels based on the channel locations. For example, using a
  | :code:`radius_um=40`, will select, for each unit, the channels which are within 40um of the channel with the
  | largest amplitude (*the extremum channel*). **This is the recommended method for high-density probes**
* | :code:`method="best_channels"`:  selects the best :code:`num_channels` channels based on their amplitudes. Note that
  | in this case the selected channels might not be close to each other.
* | :code:`method="threshold"`: selects channels based on an SNR threshold (given by the :code:`threshold` argument)
* | :code:`method="by_property"`: selects channels based on a property, such as :code:`group`. This method is recommended
  | when working with tetrodes.

The computed sparsity can be used in several postprocessing and visualization functions. In addition, a "dense"
:py:class:`~spikeinterface.core.WaveformExtractor` can be saved as "sparse" as follows:

.. code-block:: python

    we_sparse = we.save(sparsity=sparsity, folder="waveforms_sparse")

The :code:`we_sparse` object will now have an associated sparsity (:code:`we.sparsity`), which is automatically taken
into consideration for downstream analysis (with the :py:meth:`~spikeinterface.core.WaveformExtractor.is_sparse`
method). Importantly, saving sparse waveforms, especially for high-density probes, dramatically reduces the size of the
waveforms folder.

.. _save_load:


Saving, loading, and compression
--------------------------------

The Base SpikeInterface objects (:py:class:`~spikeinterface.core.BaseRecording`,
:py:class:`~spikeinterface.core.BaseSorting`, and
:py:class:`~spikeinterface.core.BaseSnippets`) hold full information about their history to maintain provenance.
Each object is in fact internally represented as a dictionary (:code:`si_object.to_dict()`) which can be used to
re-instantiate the object from scratch (this is true for all objects except in-memory ones, see :ref:`in_memory`).

The :code:`save()` function allows to easily store SI objects to a folder on disk.
:py:class:`~spikeinterface.core.BaseRecording` objects are stored in binary (.raw) or
`Zarr <https://zarr.readthedocs.io/en/stable/tutorial.html>`_ (.zarr) format and
:py:class:`~spikeinterface.core.BaseSorting` and :py:class:`~spikeinterface.core.BaseSnippets` object in numpy (.npz)
format. With the actual data, the :code:`save()` function also stores the provenance dictionary and all the properties
and annotations associated to the object.
The save function also supports parallel processing to speed up the writing process.

From a SpikeInterface folder, the saved object can be reloaded with the :code:`load_extractor()` function.
This saving/loading features enables us to store SpikeInterface objects efficiently and to distribute processing.

.. code-block:: python

    # n_jobs is related to the number of processors you want to use
    # n_jobs=-1 indicates to use all available
    job_kwargs = dict(n_jobs=8, chunk_duration="1s")
    # save recording to folder in binary (default) format
    recording_bin = recording.save(folder="recording", **job_kwargs)
    # save recording to folder in zarr format (.zarr is appended automatically)
    recording_zarr = recording.save(folder="recording", format="zarr", **job_kwargs)
    # save snippets to NPZ
    snippets_saved = snippets.save(folder="snippets")
    # save sorting to NPZ
    sorting_saved = sorting.save(folder="sorting")

**NOTE:** the Zarr format by default applies data compression with :code:`Blosc.Zstandard` codec with BIT shuffling.
Any other Zarr-compatible compressors and filters can be applied using the :code:`compressor` and :code:`filters`
arguments. For example, in this case we apply `LZMA <https://numcodecs.readthedocs.io/en/stable/lzma.html>`_
and use a `Delta <https://numcodecs.readthedocs.io/en/stable/delta.html>`_ filter:


.. code-block:: python

    from numcodecs import LZMA, Delta

    compressor = LZMA()
    filters = [Delta(dtype="int16")]

    recording_custom_comp = recording.save(folder="recording", format="zarr",
                                           compressor=compressor, filters=filters,
                                           **job_kwargs)


Parallel processing and job_kwargs
----------------------------------

The :py:mod:`~spikeinterface.core` module also contains the basic tools used throughout SpikeInterface for parallel
processing of recordings.
In general, parallelization is achieved by splitting the recording in many small time chunks and processing
them in parallel (for more details, see the :py:class:`~spikeinterface.core.ChunkRecordingExecutor` class).

Many functions support parallel processing (e.g., :py:func:`~spikeinterface.core.extract_waveforms`, :code:`save`,
and many more). All of these functions, in addition to other arguments, also accept the so-called **job_kwargs**.
These are a set of keyword arguments which are common to all functions that support parallelization:

* chunk_duration or chunk_size or chunk_memory or total_memory
    - chunk_size: int
        Number of samples per chunk
    - chunk_memory: str
        Memory usage for each job (e.g. '100M', '1G')
    - total_memory: str
        Total memory usage (e.g. '500M', '2G')
    - chunk_duration : str or float or None
        Chunk duration in s if float or with units if str (e.g. '1s', '500ms')
* n_jobs: int
    Number of jobs to use. With -1 the number of jobs is the same as number of cores.
    A float like 0.5 means half of the availables core.
* progress_bar: bool
    If True, a progress bar is printed
* mp_context: "fork" | "spawn" | None, default: None
        "fork" or "spawn". If None, the context is taken by the recording.get_preferred_mp_context().
        "fork" is only safely available on LINUX systems.

The default **job_kwargs** are :code:`n_jobs=1, chunk_duration="1s", progress_bar=True`.

Any of these arguments, can be overridden by manually passing the argument to a function
(e.g., :code:`extract_waveforms(..., n_jobs=16)`). Alternatively, **job_kwargs** can be set globally
(for each SpikeInterface session), with the :py:func:`~spikeinterface.core.set_global_job_kwargs` function:

.. code-block:: python

    global_job_kwargs = dict(n_jobs=16, chunk_duration="5s", progress_bar=False)
    set_global_job_kwargs(**global_job_kwargs)
    print(get_global_job_kwargs())
    # >>> {'n_jobs': 16, 'chunk_duration': '5s', 'progress_bar': False}

.. _in_memory:

Object "in-memory"
------------------

While most of the times SpikeInterface objects will be loaded from a file, sometimes it is convenient to construct
in-memory objects (for example, for testing a new method) or "manually" add some information to the pipeline
workflow.

In order to do this, one can use the :code:`Numpy*` classes, :py:class:`~spikeinterface.core.NumpyRecording`,
:py:class:`~spikeinterface.core.NumpySorting`, :py:class:`~spikeinterface.core.NumpyEvent`, and
:py:class:`~spikeinterface.core.NumpySnippets`. These object behave exactly like normal SpikeInterface objects,
but they are not bound to a file.

Also note the class :py:class:`~spikeinterface.core.SharedMemorySorting` which is very similar to
:py:class:`~spikeinterface.core.NumpySorting` but with an underlying SharedMemory which is useful for
parallel computing.

In this example, we create a recording and a sorting object from numpy objects:

.. code-block:: python

    import numpy as np

    # in-memory recording
    sampling_frequency = 30_000.
    duration = 10.
    num_samples = int(duration * sampling_frequency)
    num_channels = 16
    random_traces = np.random.randn(num_samples, num_channels)

    recording_memory = NumpyRecording(traces_list=[random_traces])
    # with more elements in `traces_list` we can make multi-segment objects

    # in-memory sorting
    num_units = 10
    num_spikes_unit = 1000
    spike_trains = []
    labels = []
    for i in range(num_units):
        spike_trains_i = np.random.randint(low=0, high=num_samples, size=num_spikes_unit)
        labels_i = [i] * num_spikes_unit
        spike_trains += spike_trains_i
        labels += labels_i

    sorting_memory = NumpySorting.from_times_labels(times=spike_trains, labels=labels,
                                                    sampling_frequency=sampling_frequency)


Any sorting object can be transformed into a :py:class:`~spikeinterface.core.NumpySorting` or
:py:class:`~spikeinterface.core.SharedMemorySorting` easily like this:

.. code-block:: python

    # turn any sortinto into NumpySorting
    sorting_np = sorting.to_numpy_sorting()

    # or to SharedMemorySorting for parallel computing
    sorting_shm = sorting.to_shared_memory_sorting()


.. _multi_seg:

Manipulating objects: slicing, aggregating
-------------------------------------------

:py:class:`~spikeinterface.core.BaseRecording` (and :py:class:`~spikeinterface.core.BaseSnippets`)
and :py:class:`~spikeinterface.core.BaseSorting` objects can be sliced on the time or channel/unit axis.

These operations are completely lazy, as there is no data duplication. After slicing or aggregating,
the new objects will be *views* of the original ones.

.. code-block:: python

    # here we load a very long recording and sorting
    recording = read_spikeglx(folder_path='np_folder')
    sorting =read_kilosort(folder_path='ks_folder')

    # keep one channel of every tenth channel
    keep_ids = recording.channel_ids[::10]
    sub_recording = recording.channel_slice(channel_ids=keep_ids)

    # keep between 5min and 12min
    fs = recording.sampling_frequency
    sub_recording = recording.frame_slice(start_frame=int(fs * 60 * 5), end_frame=int(fs * 60 * 12))
    sub_sorting = sorting.frame_slice(start_frame=int(fs * 60 * 5), end_frame=int(fs * 60 * 12))

    # keep only the first 4 units
    sub_sorting = sorting.select_units(unit_ids=sorting.unit_ids[:4])


We can also aggregate (or stack) multiple recordings on the channel axis using
the :py:func:`~spikeinterface.core.aggregate_channels`. Note that for this operation the recordings need to have the
same sampling frequency, number of segments, and number of samples:

.. code-block:: python

    recA_4_chans = read_binary('fileA.raw')
    recB_4_chans = read_binary('fileB.raw')
    rec_8_chans = aggregate_channels([recA_4_chans, recB_4_chans])

We can also aggregate (or stack) multiple sortings on the unit axis using the
:py:func:`~spikeinterface.core.aggregate_units` function:

.. code-block:: python

    sortingA = read_npz_sorting('sortingA.npz')
    sortingB = read_npz_sorting('sortingB.npz')
    sorting_20_units = aggregate_units([sortingA, sortingB])


Working with multiple segments
------------------------------

Multi-segment objects can result from running different recording phases (e.g., baseline, stimulation, post-stimulation)
without moving the underlying probe (e.g., just clicking play/pause on the acquisition software). Therefore, multiple
segments are assumed to record from the same set of neurons.

We have several functions to manipulate segments of SpikeInterface objects. All these manipulations are lazy.


.. code-block:: python

    # recording2: recording with 2 segments
    # recording3: recording with 3 segments

    # `append_recordings` will append all segments of multiple recordings
    recording5 = append_recordings([recording2, recording3])
    # `recording5` will have 5 segments

    # `concatenate_recordings` will make a mono-segment recording by virtual concatenation
    recording_mono = concatenate_recordings([recording2, recording5])

    # `split_recording` will return a list of mono-segment recordings out of a multi-segment one
    recording_mono_list = split_recording(recording5)
    # `recording_mono_list` will have 5 elements with 1 segment

    # `select_segment_recording` will return a user-defined subset of segments
    recording_select1 = select_segment_recording(recording5, segment_indices=3)
    # `recording_select1` will have 1 segment (the 4th one)
    recording_select2 = select_segment_recording(recording5, segment_indices=[0, 4])
    # `recording_select2` will have 2 segments (the 1st and last one)



The same functions are also available for
:py:class:`~spikeinterface.core.BaseSorting` objects
(:py:func:`~spikeinterface.core.append_sortings`,
:py:func:`~spikeinterface.core.concatenate_sortings`,
:py:func:`~spikeinterface.core.split_sorting`,
:py:func:`~spikeinterface.core.select_segment_sorting`).


**Note** :py:func:`~spikeinterface.core.append_recordings` and:py:func:`~spikeinterface.core.concatenate_recordings`
have the same goal, aggregate recording pieces on the time axis but with 2 different strategies! One is keeping the
multi segments concept, the other one is breaking it!
See this example for more detail :ref:`example_segments`.



Recording tools
---------------

The :py:mod:`spikeinterface.core.recording_tools` submodule offers some utility functions on top of the recording
object:

  * :py:func:`~spikeinterface.core.get_random_data_chunks`: retrieves some random chunks of data:
  * :py:func:`~spikeinterface.core.get_noise_levels`: estimates the channel noise levels
  * :py:func:`~spikeinterface.core.get_chunk_with_margin`: gets traces with a left and right margin
  * :py:func:`~spikeinterface.core.get_closest_channels`: returns the :code:`num_channels` closest channels to each specified channel
  * :py:func:`~spikeinterface.core.get_channel_distances`: returns a square matrix with channel distances
  * :py:func:`~spikeinterface.core.order_channels_by_depth`: gets channel order in depth


Template tools
--------------

The :py:mod:`spikeinterface.core.template_tools` submodule includes functionalities on top of the
:py:class:`~spikeinterface.core.WaveformExtractor` object to retrieve important information about the *templates*:

  * | :py:func:`~spikeinterface.core.get_template_amplitudes`: returns the amplitude of the template for each unit on
    | every channel
  * | :py:func:`~spikeinterface.core.get_template_extremum_channel`: returns the channel id (or index) where the
    | template has the largest amplitude
  * | :py:func:`~spikeinterface.core.get_template_extremum_channel_peak_shift`: returns the misalignment in samples
    | (peak shift) of each template with respect to the center of the waveforms (:py:attr:`~spikeinterface.core.WaveformExtractor.nbefore`)
  * | :py:func:`~spikeinterface.core.get_template_extremum_amplitude`: returns the amplitude of the template for each
    | unit on the extremum channel



Generate toy objects
--------------------

The :py:mod:`~spikeinterface.core` module also offers some functions to generate toy/simulated data.
They are useful to make examples, tests, and small demos:

.. code-block:: python

    # recording with 2 segments and 4 channels
    recording = generate_recording(num_channels=4, sampling_frequency=30000.,
                                   durations=[10.325, 3.5], set_probe=True)

    # sorting with 2 segments and 5 units
    sorting = generate_sorting(num_units=5, sampling_frequency=30000., durations=[10.325, 3.5],
                               firing_rate=15, refractory_period=1.5)

    # snippets of 60 samples on 2 channels from 5 units
    snippets = generate_snippets(nbefore=20, nafter=40, num_channels=2,
                                 sampling_frequency=30000., durations=[10.325, 3.5],
                                 set_probe=True,  num_units=5)


There are also some more advanced functions to generate sorting objects with varioues "mistakes"
(mainly for testing purposes):

  * :py:func:`~spikeinterface.core.synthesize_random_firings`
  * :py:func:`~spikeinterface.core.clean_refractory_period`
  * :py:func:`~spikeinterface.core.inject_some_duplicate_units`
  * :py:func:`~spikeinterface.core.inject_some_split_units`
  * :py:func:`~spikeinterface.core.synthetize_spike_train_bad_isi`


Downloading test datasets
-------------------------

The `NEO <https://github.com/NeuralEnsemble/python-neo>`_ package is maintaining a collection of many
electrophysiology file formats: https://gin.g-node.org/NeuralEnsemble/ephy_testing_data

The :py:func:`~spikeinterface.core.download_dataset` function is capable of downloading and caching locally dataset
from this repository. The function depends on the :code:`datalad` python package, which internally depends on
:code:`git` and :code:`git-annex`.

The :py:func:`~spikeinterface.core.download_dataset`  is very useful to perform local tests on small files from
various formats:

.. code-block:: python

    # Spike" format
    local_file_path = download_dataset(remote_path='spike2/130322-1LY.smr')
    rec = read_spike2(local_file_path)

    # MEArec format
    local_file_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
    rec, sorting = read_mearec(local_file_path)

    # SpikeGLX format
    local_folder_path = download_dataset(remote_path='/spikeglx/multi_trigger_multi_gate')
    rec = read_spikeglx(local_folder_path)
