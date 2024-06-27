Extractors module
=================

Overview
--------

The :py:mod:`~spikeinterface.extractors` module allows you to load :py:class:`~spikeinterface.core.BaseRecording`,
:py:class:`~spikeinterface.core.BaseSorting`, and :py:class:`~spikeinterface.core.BaseEvent` objects from
a large variety of acquisition systems and spike sorting outputs.

Most of the :code:`Recording` classes are implemented by wrapping the
`NEO rawio implementation <https://github.com/NeuralEnsemble/python-neo/tree/master/neo/rawio>`_.

Most of the :code:`Sorting` classes are instead directly implemented in SpikeInterface.

Although SpikeInterface is object-oriented (class-based), each object can also be loaded with a convenient
:code:`read_XXXXX()` function.

.. code-block:: python

    import spikeinterface.extractors as se


Read one Recording
------------------

Every format can be read with a simple function:

.. code-block:: python

    recording_oe = read_openephys(folder_path="open-ephys-folder")

    recording_spikeglx = read_spikeglx(folder_path="spikeglx-folder")

    recording_blackrock = read_blackrock(folder_path="blackrock-folder")

    recording_mearec = read_mearec(file_path="mearec_file.h5")


Importantly, some formats directly handle the probe information:

.. code-block:: python

    recording_spikeglx = read_spikeglx(folder_path="spikeglx-folder")
    print(recording_spikeglx.get_probe())

    recording_mearec = read_mearec(file_path="mearec_file.h5")
    print(recording_mearec.get_probe())




Read one Sorting
----------------

.. code-block:: python

    sorting_KS = read_kilosort(folder_path="kilosort-folder")


Read one Event
--------------

.. code-block:: python

    events_OE = read_openephys_event(folder_path="open-ephys-folder")


For a comprehensive list of compatible technologies, see :ref:`compatible_formats`.


Lazy loading
------------

An important concept is that all :code:`read_XXXX()` functions are lazy.
Traces are not read from disk; instead only the relevant metadata (e.g. channel_ids, sampling frequency, etc.) is.

The actual reading will be done on demand using the :py:meth:`~spikeinterface.core.BaseRecording.get_traces` method:

.. code-block:: python

    # opening a 40GB SpikeGLX dataset is fast
    recording_spikeglx = read_spikeglx(folder_path="spikeglx-folder")

    # this really does load the full 40GB into memory : not recommended!!!!!
    traces = recording_spikeglx.get_traces(start_frame=None, end_frame=None, return_scaled=False)



.. _compatible_formats:

Supported File Formats
----------------------

Currently, we support many popular file formats for both raw and sorted extracellular datasets.
Given the standardized, modular design of our recording and sorting extractors,
adding new file formats is straightforward so we expect this list to grow in future versions.

Most formats are supported on top of `NEO <https://github.com/NeuralEnsemble/python-neo>`_

Dependencies
------------

The :code:`neo` package is a hard dependency of SpikeInterface. So all formats handled by Neo directly will also be handled
in SpikeInterface.

However, some formats are handled directly by SpikeInterface and need extra installation.

You can install all extractors' dependencies with:

.. code-block:: python

    pip install spikeinterface[extractor]


Raw Data Formats
----------------

For raw recording formats, we currently support:

* **AlphaOmega** :py:func:`~spikeinterface.extractors.read_alphaomega()`
* **Axona** :py:func:`~spikeinterface.extractors.read_axona()`
* **BlackRock** :py:func:`~spikeinterface.extractors.read_blackrock()`
* **Binary** :py:func:`~spikeinterface.core.read_binary()`
* **Biocam HDF5** :py:func:`~spikeinterface.extractors.read_biocam()`
* **CED** :py:func:`~spikeinterface.extractors.read_ced()`
* **EDF** :py:func:`~spikeinterface.extractors.read_edf()`
* **IBL streaming** :py:func:`~spikeinterface.extractors.read_ibl_recording()`
* **Intan** :py:func:`~spikeinterface.extractors.read_intan()`
* **MaxWell** :py:func:`~spikeinterface.extractors.read_maxwell()`
* **MCS H5** :py:func:`~spikeinterface.extractors.read_mcsh5()`
* **MCS RAW** :py:func:`~spikeinterface.extractors.read_mcsraw()`
* **MEArec** :py:func:`~spikeinterface.extractors.read_mearec()`
* **Mountainsort MDA** :py:func:`~spikeinterface.extractors.read_mda_recording()`
* **Neuralynx** :py:func:`~spikeinterface.extractors.read_neuralynx()`
* **Neurodata Without Borders** :py:func:`~spikeinterface.extractors.read_nwb_recording()`
* **Neuroscope** :py:func:`~spikeinterface.coextractorsre.read_neuroscope_recording()`
* **Neuroexplorer** :py:func:`~spikeinterface.extractors.read_neuroexplorer()`
* **NIX** :py:func:`~spikeinterface.extractors.read_nix()`
* **Open Ephys Legacy** :py:func:`~spikeinterface.extractors.read_openephys()`
* **Open Ephys Binary** :py:func:`~spikeinterface.extractors.read_openephys()`
* **Plexon** :py:func:`~spikeinterface.extractors.read_plexon()`
* **Plexon 2** :py:func:`~spikeinterface.extractors.read_plexon2()`
* **Shybrid** :py:func:`~spikeinterface.extractors.read_shybrid_recording()`
* **SpikeGLX** :py:func:`~spikeinterface.extractors.read_spikeglx()`
* **SpikeGLX IBL compressed** :py:func:`~spikeinterface.extractors.read_cbin_ibl()`
* **SpikeGLX IBL stream** :py:func:`~spikeinterface.extractors.read_streaming_ibl()`
* **Spike 2** :py:func:`~spikeinterface.extractors.read_spike2()`
* **TDT** :py:func:`~spikeinterface.extractors.read_tdt()`
* **Zarr** :py:func:`~spikeinterface.core.read_zarr()`


Sorted Data Formats
-------------------

For sorted data formats, we currently support:

* **BlackRock** :py:func:`~spikeinterface.extractors.read_blackrock_sorting()`
* **Combinato** :py:func:`~spikeinterface.extractors.read_combinato()`
* **Cell explorer** :py:func:`~spikeinterface.extractors.read_cellexplorer()`
* **HerdingSpikes2** :py:func:`~spikeinterface.extractors.read_herdingspikes()`
* **HDsort** :py:func:`~spikeinterface.extractors.read_hdsort()`
* **Kilosort1/2/2.5/3** :py:func:`~spikeinterface.extractors.read_kilosort()`
* **Klusta** :py:func:`~spikeinterface.extractors.read_klusta()`
* **MClust** :py:func:`~spikeinterface.extractors.read_mclust()`
* **MEArec** :py:func:`~spikeinterface.extractors.read_mearec()`
* **Mountainsort MDA** :py:func:`~spikeinterface.extractors.read_mda_sorting()`
* **Neurodata Without Borders** :py:func:`~spikeinterface.extractors.read_nwb_sorting()`
* **Neuroscope** :py:func:`~spikeinterface.extractors.read_neuroscope_sorting()`
* **Neuralynx spikes** :py:func:`~spikeinterface.extractors.read_neuralynx_sorting()`
* **NPZ (created by SpikeInterface)** :py:func:`~spikeinterface.core.read_npz_sorting()`
* **Plexon spikes** :py:func:`~spikeinterface.extractors.read_plexon_sorting()`
* **Plexon 2 spikes** :py:func:`~spikeinterface.extractors.read_plexon2_sorting()`
* **Shybrid**  :py:func:`~spikeinterface.extractors.read_shybrid_sorting()`
* **Spyking Circus** :py:func:`~spikeinterface.extractors.read_spykingcircus()`
* **Trideclous** :py:func:`~spikeinterface.extractors.read_tridesclous()`
* **Wave Clus** :py:func:`~spikeinterface.extractors.read_waveclus()`
* **YASS** :py:func:`~spikeinterface.extractors.read_yass()`


Dealing with Non-Supported File Formats
---------------------------------------

With recording and sorting objects, we hope that any user can access SpikeInterface regardless of the nature of their
underlying file format. If you feel like a non-supported file format should be included in SpikeInterface as an
actual extractor, please open an issue.
