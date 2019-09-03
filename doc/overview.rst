Overview
========

When recording from extracellular electrodes, **spike sorting** is needed to separate the activity of individual
neurons, or units.

However, there are many different file formats for recording data and several spike sorting algorithms available. This
usually results in custom-made codes to parse the data and set up a specific spike sorting software.

SpikeInterface is therefore here to make the analysis of extracellular electrophysiology data easy and accessible, even
with very little coding experience.

SpikeInterface interfaces with several file formats and allows to run numerous spike sorting algorithms with a few lines
of code. Moreover, it enables users to run common pre- and post-processing functions, to validate the spike sorting
output with state-of-the-art quality metrics, to compare the output of different sorters, and to visualize all steps
involved in the electrophysiology pipeline.

Organization
------------

SpikeInterface consists of a collection of Python packages aimed to ease the spike sorting process.

There are 5 packages with different scopes and functions:

- `spikeextractors <https://github.com/SpikeInterface/spikeextractors/>`_
- `spiketoolkit <https://github.com/SpikeInterface/spiketoolkit/>`_
- `spikesorters <https://github.com/SpikeInterface/spikesorters/>`_
- `spikecomparison <https://github.com/SpikeInterface/spikecomparison/>`_
- `spikewidgets <https://github.com/SpikeInterface/spikewidgets/>`_

On top of these packages, the :code:`spikeinterface` package allows users to install and use all of the packages,
available as modules as shown in the figure.

.. image:: images/overview.png


Imporant links
--------------

- `spikeforest <https://spikeforest.flatironinstitute.org>`_ is a reproducible, continuously updating platform which
  benchmarks the performance of some spike sorting codes (kilosort, herdingspike, ironcliust, jrclust, klusta, moutainsort4,
  spykingcircus, tridesclous, waveclust, ...) based on spikeinterface modules.
- `spikely <https://github.com/SpikeInterface/spikely>`_ released very soon that make possible to use spikeinterface with
  a graphicall user interface done in Qt.
