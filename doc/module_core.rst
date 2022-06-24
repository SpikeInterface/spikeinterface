Core module
===========

Overview
--------

The :py:mod:`spikeinterface.core` module provides the basic classes and tools of the SpikeInterface ecosystem.

Several Base classes are implemented here and inherited throughout the SI code-base.

All classes support multiple segments. Each segment is a contiguous piece of data (recording, sorting, events).


Recording
---------

The :py:class:`~spikeinterface.core.BaseRecording` class serves as basis for all
:code:`RecordingExtractors`.
It represents an extracellular recording and has the following features:

* retrieve raw and scaled traces from each segment
* keep info about channel_ids VS channel indices
* handle probe information
* store channel properties
* store object annotations
* enable grouping, splitting, and slicing
* handle segment operations (e.g. concatenation)
* handle time information


Sorting
-------

The :py:class:`~spikeinterface.core.BaseSorting` class serves as basis for all :code:`SortingExtractors`.
It represents a spike sorted output and has the following features:

* retrieve spike trains for different units
* keep info about unit_ids VS unit indices
* store channel properties
* store object annotations
* enable selection of sub-units
* handle time information


Event
-----

The :py:class:`~spikeinterface.core.BaseEvent` class serves as basis for all :code:`Event` classes.
It represents a events during the recording (e.g. TTL pulses) and has the following features:

* retrieve events and/or epochs from files
* enable grouping, splitting, and slicing (TODO)
* handle segment operations (e.g. concatenation) (TODO)

WaveformExtractor
-----------------

The :py:class:`~spikeinterface.core.WaveformExtractor` class is the core of postprocessing a spike sorting output.
It combines a paired recording-sorting objects to extract waveforms.
It allows to:

* retrieve waveforms
* control spike subsampling for waveforms
* compute templates (i.e. average extracellular waveforms)
* save waveforms in a folder for easy retrieval


Saving and loading
------------------

All SI objects hold full information about their history to endure provenance. Each object is in fact internally
represented as a dictionary (:code:`si_object.to_dict()`) which can be used to reload the object from scratch.

The :code:`save()` function allows to easily store SI objects to a folder on disk.
:py:class:`~spikeinterface.core.BaseRecording` objects are stored in binary (.raw) format  and
:py:class:`~spikeinterface.core.BaseSorting` object in numpy (.npz) format. With the actual data, the :code:`save()`
function also stores the provenance dictionary and all the properties and annotations associated to the object.

From a saved SI folder, an SI object can be reloaded with the :code:`si.load_extractor()` function.

This saving/loading features enables to store SI objects efficiently and to distribute processing.


Parallel processing
-------------------

The :py:mod:`~spikeinterface.core` module also contains the basic tools used throughout SI for parallel processing.
To discover more about it, checkout the :py:class:`~spikeinterface.core.ChunkRecordingExecutor` class.
