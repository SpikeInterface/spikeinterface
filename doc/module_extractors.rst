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
