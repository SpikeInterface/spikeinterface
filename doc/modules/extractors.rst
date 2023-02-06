Extractors module
=================

Overview
--------

The :py:mod:`~spikeinterface.extractors` module contains :code:`RecordingExtractor` and :code:`SortingExtractor` classes
to interface with a large variety of acquisition systems and spike sorting outputs.

Most of the :code:`RecordingExtractor` classes are implemented by wrapping the
`NEO rawio implementation <https://github.com/NeuralEnsemble/python-neo/tree/master/neo/rawio>`_.

Most of the :code:`SortingExtractor` are instead directly implemented in SpikeInterface.


Although SI is object-oriented (class-based), each object can also be loaded with  a convenient :code:`read()` function:

Supported File Formats
----------------------

Currently, we support many popular file formats for both raw and sorted extracellular datasets.
Given the standardized, modular design of our recording and sorting extractors,
adding new file formats is straightforward so we expect this list to grow in future versions.

Most of format are supported on top on `NEO <https://github.com/NeuralEnsemble/python-neo>`_

Raw Data Formats
----------------

For raw data formats, we currently support:

* **BlackRock** - BlackRockRecordingExtractor
* **Binary** - BinaryRecordingExtractor
* **Biocam HDF5** - BiocamRecordingExtractor
* **CED** - CEDRecordingExtractor
* **Intan** - IntanRecordingExtractor
* **Klusta** - KlustaRecordingExtractor
* **MaxOne** - MaxOneRecordingExtractor
* **MCSH5** - MCSH5RecordingExtractor
* **MEArec** - MEArecRecordingExtractor
* **Mountainsort MDA** - MdaRecordingExtractor
* **Neurodata Without Borders** - NwbRecordingExtractor
* **Neuroscope** - NeuroscopeRecordingExtractor
* **NIX** - NIXIORecordingExtractor
* **Neuralynx** - NeuralynxRecordingExtractor
* **Open Ephys Legacy** - OpenEphysLegacyRecordingExtractor
* **Open Ephys Binary** - OpenEphysBinaryRecordingExtractor
* **Phy/Kilosort** - PhyRecordingExtractor/KilosortRecordingExtractor
* **Plexon** - PlexonRecordingExtractor
* **Shybrid** - SHYBRIDRecordingExtractor
* **SpikeGLX** - SpikeGLXRecordingExtractor
* **Spyking Circus** - SpykingCircusRecordingExtractor

Sorted Data Formats
-------------------

For sorted data formats, we currently support:

* **BlackRock** - BlackRockSortingExtractor
* **Combinato** - CombinatoSortingExtractor
* **Experimental Directory Structure (Exdir)** - ExdirSortingExtractor
* **HerdingSpikes2** - HS2SortingExtractor
* **JRClust** - JRCSortingExtractor
* **Kilosort/Kilosort2** - KiloSortSortingExtractor
* **Klusta** - KlustaSortingExtractor
* **MEArec** - MEArecSortingExtractor
* **Mountainsort MDA** - MdaSortingExtractor
* **Neurodata Without Borders** - NwbSortingExtractor
* **Neuroscope** - NeuroscopeSortingExtractor
* **NPZ (created by SpikeInterface)** - NpzSortingExtractor
* **Open Ephys** - OpenEphysSortingExtractor
* **Shybrid** - SHYBRIDSortingExtractor
* **Spyking Circus** - SpykingCircusSortingExtractor
* **Trideclous** - TridesclousSortingExtractor
* **YASS** - YassSortingExtractor

Installed Extractors
--------------------

To check which extractors are useable in a given python environment, one can print the installed recording extractor
list and the installed sorting extractor list. An example from a newly installed miniconda3 environment is shown below,

First, import the spikeextractors package,

.. code:: python

  import spikeinterface.extractors as se

Then you can check the installed RecordingExtractor list,

.. code:: python

  se.installed_recording_extractor_list

which outputs,

.. parsed-literal::
  [spikeinterface.extractors.MdaRecordingExtractor,
   spikeinterface.extractorsBiocamRecordingExtractor,
   spikeinterface.core..BinaryRecordingExtractor,
   ...

and the installed SortingExtractors list,

.. code:: python

  se.installed_sorting_extractor_list

which outputs,

.. parsed-literal::
  [spikeinterface.extractors.SpykingCircusSortingExtractor,
   spikeinterface.extractors.HerdingspikesSortingExtractor,
   ...

When trying to use an extractor that has not been installed in your environment, an installation message will appear indicating which python packages must be installed as a prerequisite to using the extractor,

.. code:: python

  mearec_file = 'path_to_exdir_file.h5'
  recording = se.MEArecRecordingExtractor(mearec_file)

throws the error,

.. parsed-literal::
  ----> 1 se.MEArecRecordingExtractor(mearec_file)

  AssertionError: To use the MEArecRecordingExtractor run:

  pip install MEARec


Dealing with Non-Supported File Formats
---------------------------------------

Many users store their datasets in custom file formats that are not general enough to create new extractors. To allow these users to still utilize SpikeInterface with their data,
we built two in-memory Extractors: the **NumpyRecordingExtractor** and the **NumpySortingExtractor**.

The NumpyRecordingExtractor can be instantiated with a numpy array that contains the underlying extracellular traces (channels x frames), the sampling frequency, and the probe geometry (optional).
Once instantiated, the NumpyRecordingExtractor can be used like any other RecordingExtractor.

The NumpySortingExtractor does not need any data during instantiation. However, after instantiation, it can be filled with data using its built-in functions (load_from_extractor, set_times_labels, and add_unit).
After sorted data is added to the NumpySortingExtractor, it can be used like any other SortingExtractor.

With these two objects, we hope that any user can access SpikeInterface regardless of the nature of their underlying file format. If you feel like a non-supported file format should be included in SpikeInterface as
an actual extractor, please leave an issue in the spikeextractors repository.




Read one Recording
------------------

.. code-block:: python

    import spikeinterface.extractors as se

    recording_OE = se.read_openephys("open-ephys-folder")


Read one Sorting
----------------

.. code-block:: python

    import spikeinterface.extractors as se

    sorting_KS = se.read_kilosort("kilosort-folder")


Read one Event
--------------

.. code-block:: python

    import spikeinterface.extractors as se

    events_OE = se.read_openephys_event("open-ephys-folder")

For a comprehensive list of compatible technologies, see :ref:`compatible-tech`.
