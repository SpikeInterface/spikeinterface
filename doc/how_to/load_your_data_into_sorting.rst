Load Your Own Data into a Sorting
=================================

Why make a :code:`Sorting`?

SpikeInterface contains pre-build readers for the output of many common sorters.
However, what if you have sorting output that is not in a standard format (e.g.
old csv file)? If this is the case you can make your own Sorting object to load
your data into SpikeInterface. This means you can still easily apply various
downstream analyses to your results (e.g. building correlograms or for generating
a :code:`SortingAnalyzer``).

The Sorting object is a core object within SpikeInterface that acts as a convenient
way to interface with sorting results, no matter which sorter was used to generate
them. **At a fundamental level it is a series of spike times and a series of labels
for each unit and a sampling frequency for transforming frames to time.** Below, we will show you have
to take your existing data and load it as a SpikeInterface :code:`Sorting` object.


Reading a standard spike sorting format into a :code:`Sorting`
--------------------------------------------------------------

For most spike sorting output formats the :code:`Sorting` is automatically generated. For example one could do

.. code-block:: python

    from spikeinterface.extractors import read_phy

    # For kilosort/phy output files we can use the read_phy
    # most formats will have a read_xx that can used.
    phy_sorting = read_phy('path/to/folder')

And voilà you now have your :code:`Sorting` object generated and can use it for further analysis. For all the
current formats see :ref:`compatible_formats`.



Loading your own data into a :code:`Sorting`
--------------------------------------------


This :code:`Sorting` contains important information about your spike trains including:

  * spike times: the peaks of the extracellular potentials expressed in samples/frames these can
    be converted to seconds under the hood using the sampling_frequency
  * spike labels: the neuron id for each spike, can also be called cluster ids or unit ids
    Stored as the :code:`unit_ids` in SpikeInterface
  * sampling_frequency: the rate at which the recording equipment was run at. Note this is the
    frequency and not the period. This value allows for switching between samples/frames to seconds


There are 3 options for loading your own data into a sorting object

With lists of spike trains and spike labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this case we need a list of spike times unit labels, sampling_frequency and optional unit_ids
if you want specific labels to be used (in this case we only create the :code:`Sorting` based on
the requested unit_ids).

.. code-block:: python

    import numpy as np
    from spikeinterface.core import NumpySorting

    # in this case we are making a monosegment sorting
    # we have four spikes that are spread among two neurons
    my_sorting = NumpySorting.from_times_labels(
        times_list=[
            np.array([1000,12000,15000,22000])   # Note these are samples/frames not times in seconds
            ],
        labels_list=[
            np.array(["a","b","a","b"])
            ],
        sampling_frequency=30_000.0
        )


With a unit dictionary
^^^^^^^^^^^^^^^^^^^^^^

We can also use a dictionary where each unit is a key and its spike times are values.
This is entered as either a list of dicts with each dict being a segment or as a single
dict for monosegment. We still need to separately specify the sampling_frequency

.. code-block:: python

    from spikeinterface.core import NumpySorting

    my_sorting = NumpySorting.from_unit_dict(
        units_dict_list={
            '0': [1000,15000],
            '1': [12000,22000],
            },
        sampling_frequency=30_000.0
        )


With Neo SpikeTrains
^^^^^^^^^^^^^^^^^^^^

Finally since SpikeInterface is tightly integrated with the Neo project you can create
a sorting from :code:`Neo.SpikeTrain` objects. See :doc:`Neo documentation<neo:index>` for more information on
using :code:`Neo.SpikeTrain`'s.

.. code-block:: python

    from spikeinterface.core import NumpySorting

    # neo_spiketrain is a Neo spiketrain object
    my_sorting = NumpySorting.from_neo_spiketrain_list(
        neo_spiketrain,
        sampling_frequency=30_000.0,
        )


Loading multisegment data into a :code:`Sorting`
------------------------------------------------

One of the great advantages of SpikeInterface :code:`Sorting` objects is that they can also handle
multisegment recordings and sortings (e.g. you have a baseline, stimulus, post-stimulus). The
exact same machinery can be used to generate your sorting, but in this case we do a list of arrays instead of
a single list. Let's go through one example for using :code:`from_times_labels`:

.. code-block:: python

    import numpy as np
    from spikeinterface.core import NumpySorting

    # in this case we are making three-segment sorting
    # we have four spikes that are spread among two neurons
    # in each segment
    my_sorting = NumpySorting.from_times_labels(
        times_list=[
            np.array([1000,12000,15000,22000]),
            np.array([30000,33000, 41000, 47000]),
            np.array([50000,53000,64000,70000]),
            ],
        labels_list=[
            np.array([0,1,0,1]),
            np.array([0,0,1,1]),
            np.array([1,0,1,0]),
        ],
        sampling_frequency=30_000.0
        )


Next steps
----------

Now that we've created a Sorting object you can combine it with a Recording to make a
:ref:`SortingAnalyzer<sphx_glr_tutorials_core_plot_4_sorting_analyzer.py>`
or start visualizing using plotting functions from our widgets model such as
:py:func:`~spikeinterface.widgets.plot_crosscorrelograms`.
