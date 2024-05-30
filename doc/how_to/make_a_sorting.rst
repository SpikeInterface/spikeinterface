How to Make your own Sorting
============================

Why make a :code:`Sorting`?

The :code:`Sorting` object is one of the core objects within the SpikeInterface library
along with :code:`Recording` and :code:`SortingAnalyzer`. Although SpikeInterface has
many tools for reading sorting formats you may have some data in a nonstandard format
(e.g. old csv file). If this is this the case you would need to make your own :code:`Sorting`.


Making a :code:`Sorting`
------------------------

For most formats the :code:`Sorting` is automatically generated. For example one could do

.. code-block:: python

    from spikeinterface.extractors import read_kilosort, read_phy

    # For kilosort/phy files we can use either reader
    ks_sorting = read_kilosort('path/to/folder')
    phy_sorting = read_phy('path/to/folder')

This :code:`Sorting` contains important information about your spike trains including
the spike times (i.e. when the neurons were actually firing) the unit labels (i.e.
who the spikes belong to. Also called cluster ids by some sorters), the unit ids (the unique
set of unit labels) and the sampling_frequency. To make your own :code:`Sorting` object you can
use :code:`NumpySorting`. There are 3 options:

With lists
----------

In this case we need a list or array of spike times, unit labels, sampling_frequency and optional
unit_ids if you want specific labels to be used.

.. code-block:: python

    from spikeinterface.core import NumpySorting

    my_sorting = NumpySorting.from_times_labels(times_list = [1,2,3,4],
                                                labels_list = [0,1,0,1],
                                                sampling_frequency = 30_000.0
                                                )


With a unit dict
----------------

We can also use a dictionary where each unit is a key and its spike times are values.
This is entered as either a list of dicts with each dict being a segment or as a single
dict for monosegment.

.. code-block:: python

    from spikeinterface.core import NumpySorting

    my_sorting = NumpySorting.from_unit_dict(units_dict_list={'0': [1,3],
                                                              '1': [2,4]
                                                              },
                                            sampling_frequency=30_000.0
                                           )


From Neo
--------

Finally since SpikeInterface is tightly integrated with the Neo project you can create
a sorting from :code:`Neo.SpikeTrain` objects.

.. code-block:: python

    from spikeinterface.core import NumpySorting

    my_sorting = NumpySorting.from_neo_spiketrain_list(neo_spiketrain, sampling_frequency=30_000.0)
