Compatible Technology
~~~~~~~~~~~~~~~~~~~~~

Supported File Formats
======================

Currently, we support many popular file formats for both raw and sorted extracellular datasets. Given the standardized, modular design of our recording and sorting extractors, adding new file formats is straightforward so we expect this list to grow in future versions.

Raw Data Formats
----------------

For raw data formats, we currently support:

* **Binary** - BinDatRecordingExtractor
* **Biocam HDF5** - BiocamRecordingExtractor
* **Experimental Directory Structure (Exdir)** - ExdirRecordingExtractor
* **Intan** - IntanRecordingExtractor
* **Klusta** - KlustaRecordingExtractor
* **MaxOne** - MaxOneRecordingExtractor
* **MCSH5** - MCSH5RecordingExtractor
* **MEArec** - MEArecRecordingExtractor
* **Mountainsort MDA** - MdaRecordingExtractor
* **Neurodata Without Borders** - NwbRecordingExtractor
* **Open Ephys** - OpenEphysRecordingExtractor
* **Phy/Kilosort** - PhyRecordingExtractor/KilosortRecordingExtractor
* **SpikeGLX** - SpikeGLXRecordingExtractor
* **Spyking Circus** - SpykingCircusRecordingExtractor

Sorted Data Formats
-------------------

For sorted data formats, we currently support:

* **Experimental Directory Structure (Exdir)** - ExdirSortingExtractor
* **HerdingSpikes2** - HS2SortingExtractor
* **Kilosort/Kilosort2** - KiloSortSortingExtractor
* **Klusta** - KlustaSortingExtractor
* **MEArec** - MEArecSortingExtractor
* **Mountainsort MDA** - MdaSortingExtractor
* **Neurodata Without Borders** - NwbSortingExtractor
* **Neuroscope** - NeuroscopeSortingExtractor
* **NPZ (created by SpikeInterface)** - NpzSortingExtractor
* **Open Ephys** - OpenEphysSortingExtractor
* **Spyking Circus** - SpykingCircusSortingExtractor
* **Trideclous** - TridesclousSortingExtractor

Installed Extractors
--------------------

To check which extractors are useable in a given python environment, one can print the installed recording extractor list and the installed sorting extractor list. An example from a newly installed miniconda3 environment is shown below,

First, import the spikeextractors package,

.. code:: python

  import spikeextractors as se

Then you can check the installed RecordingExtractor list,

.. code:: python

  se.installed_recording_extractor_list
  
which outputs,

.. parsed-literal::
  [spikeextractors.extractors.mdaextractors.mdaextractors.MdaRecordingExtractor,
   spikeextractors.extractors.biocamrecordingextractor.biocamrecordingextractor.BiocamRecordingExtractor,
   spikeextractors.extractors.bindatrecordingextractor.bindatrecordingextractor.BinDatRecordingExtractor,
   spikeextractors.extractors.spikeglxrecordingextractor.spikeglxrecordingextractor.SpikeGLXRecordingExtractor,
   spikeextractors.extractors.phyextractors.phyextractors.PhyRecordingExtractor,
   spikeextractors.extractors.maxonerecordingextractor.maxonerecordingextractor.MaxOneRecordingExtractor]
   
and the installed SortingExtractors list,

.. code:: python

  se.installed_sorting_extractor_list

which outputs,

.. parsed-literal::
  [spikeextractors.extractors.mdaextractors.mdaextractors.MdaSortingExtractor,
   spikeextractors.extractors.hs2sortingextractor.hs2sortingextractor.HS2SortingExtractor,
   spikeextractors.extractors.klustasortingextractor.klustasortingextractor.KlustaSortingExtractor,
   spikeextractors.extractors.kilosortsortingextractor.kilosortsortingextractor.KiloSortSortingExtractor,
   spikeextractors.extractors.phyextractors.phyextractors.PhySortingExtractor,
   spikeextractors.extractors.spykingcircussortingextractor.spykingcircussortingextractor.SpykingCircusSortingExtractor,
   spikeextractors.extractors.npzsortingextractor.npzsortingextractor.NpzSortingExtractor]

 
When trying to use an extractor that has not been installed in your environment, an installation message will appear indicating which python packages must be installed as a prerequisite to using the extractor,

.. code:: python

  exdir_file = 'path_to_exdir_file'
  recording = se.ExdirRecordingExtractor(exdir_file)

throws the error,

.. parsed-literal::
  ----> 1 se.ExdirRecordingExtractor(exdir_file)

  ~/spikeextractors/spikeextractors/extractors/exdirextractors/exdirextractors.py in __init__(self, exdir_file)
       22 
       23     def __init__(self, exdir_file):
  ---> 24         assert HAVE_EXDIR, "To use the ExdirExtractors run:\n\n pip install exdir\n\n"
       25         RecordingExtractor.__init__(self)
       26         self._exdir_file = exdir_file

  AssertionError: To use the ExdirExtractors run:

  pip install exdir

So to use either of the Exdir extractors, you must install the python package exdir. The python packages that are required to use of all the extractors can be installed as below,

.. parsed-literal::
  pip install exdir h5py pyintan MEArec pyopenephys tridesclous
  
Dealing with Non-Supported File Formats
=======================================

Many users may store their datasets in custom file formats that are not general enough to create new extractors. To allow these users to still utilize SpikeInterface with their data,
we built two in-memory Extractors: the **NumpyRecordingExtractor** and the **NumpySortingExtractor**.

The NumpyRecordingExtractor can be instantiated with a numpy array that contains the underlying extracellular traces (channels x frames), the sampling frequency, and the probe geometry (optional).
Once instantiated, the NumpyRecordingExtractor can be used like any other RecordingExtractor.

The NumpySortingExtractor does not need any data during instantiation. However, after instantiation, it can be filled with data using its built-in functions (load_from_extractor, set_times_labels, and add_unit).
After sorted data is added to the NumpySortingExtractor, it can be used like any other SortingExtractor.

With these two objects, we hope that any user can access SpikeInterface regardless of the nature of their underlying file format. If you feel like a non-supported file format should be included in SpikeInterface as 
an actual extractor, please leave an issue in the spikeextractors repository.

Supported Spike Sorters
=======================

Currently, we support many popular semi-automatic spike sorters.  Given the standardized, modular design of our sorters, adding new ones is straightforward so we expect this list to grow in future versions.


* **HerdingSpikes2** - HerdingspikesSorter
* **IronClust** - IronClustSorter
* **Kilosort** - KilosortSorter
* **Kilosort2** - Kilosort2Sorter
* **Klusta** - KlustaSorter
* **Mountainsort4** - Mountainsort4Sorter
* **SpyKING Circus** - SpykingcircusSorter
* **Tridesclous** - TridesclousSorter
* **Wave clus** - WaveClusSorter


Installed Sorters
------------------

To check which sorters are useable in a given python environment, one can print the installed sorters list. An example is shown in a pre-defined miniconda3 environment.

First, import the spikesorters package,

.. code:: python

  import spikesorters as sorters

Then you can check the installed Sorter list,

.. code:: python

  sorters.installed_sorter_list
  
which outputs,

.. parsed-literal::
  [spikesorters.klusta.klusta.KlustaSorter,
   spikesorters.tridesclous.tridesclous.TridesclousSorter,
   spikesorters.mountainsort4.mountainsort4.Mountainsort4Sorter,
   spikesorters.spyking_circus.spyking_circus.SpykingcircusSorter,
   spikesorters.herdingspikes.herdingspikes.HerdingspikesSorter]


When trying to use an sorter that has not been installed in your environment, an installation message will appear indicating how to install the given sorter,

.. code:: python

  recording = sorters.run_ironclust(recording)

throws the error,

.. parsed-literal::
  AssertionError: This sorter ironclust is not installed.
        Please install it with:  

  To use IronClust run:

        >>> git clone https://github.com/jamesjun/ironclust
    and provide the installation path by setting the IRONCLUST_PATH
    environment variables or using IronClustSorter.set_ironclust_path().
