Make your own Sorting
=====================

Why make a :code:`Sorting`?

The :code:`Sorting` object is one of the core objects within the SpikeInterface library
along with :code:`Recording` and :code:`SortingAnalyzer`. Although SpikeInterface has
many tools for reading sorting formats you may have some data in a nonstandard format
(e.g. old csv file). If this is the case you would need to make your own :code:`Sorting`.

At a fundamental level the :code:`Sorting` is a series of spike times and a series of
labels for each spike along with some associated metadata. Thus by providing this
information you can easily make a :code:`Sorting` object to be used for various analyses
(e.g. correlograms or for generating a :code:`SortingAnalyzer`)


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
use :code:`NumpySorting`. It is important to note that in SpikeiInterface spike trains are handled internally in samples/frames rather than in seconds and we use the sampling frequency to ...
are typically stored in samples/frames rather than in seconds. So you should input the times
in samples/frames. The sampling_frequency allows for easily switching between samples and seconds.

There are 3 options (along with making a NumpySorting from another sorting which will not be covered here):

With lists of spike trains and spike labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this case we need a list or array (or lists of lists for multisegment) of spike times,
unit labels, sampling_frequency and optional unit_ids if you want specific labels to be
used (in this case we only create the :code:`Sorting` based on the requested unit_ids).

.. code-block:: python

    from spikeinterface.core import NumpySorting

    # in this case we are making a monosegment sorting
    my_sorting = NumpySorting.from_times_labels(times_list = [1,2,3,4],
                                                labels_list = [0,1,0,1],
                                                sampling_frequency = 30_000.0
                                                )


With a unit dictionary
^^^^^^^^^^^^^^^^^^^^^^

We can also use a dictionary where each unit is a key and its spike times are values.
This is entered as either a list of dicts with each dict being a segment or as a single
dict for monosegment. We still need to separately specify the sampling_frequency

.. code-block:: python

    from spikeinterface.core import NumpySorting

    my_sorting = NumpySorting.from_unit_dict(units_dict_list={'0': [1,3],
                                                              '1': [2,4]
                                                              },
                                             sampling_frequency=30_000.0
                                           )


With Neo SpikeTrains
^^^^^^^^^^^^^^^^^^^^

Finally since SpikeInterface is tightly integrated with the Neo project you can create
a sorting from :code:`Neo.SpikeTrain` objects. See Neo documentation for more information on
using :code:`Neo.SpikeTrain`'s.

.. code-block:: python

    from spikeinterface.core import NumpySorting

    # neo_spiketrain is a Neo spiketrain object
    my_sorting = NumpySorting.from_neo_spiketrain_list(neo_spiketrain,
                                                       sampling_frequency=30_000.0)
