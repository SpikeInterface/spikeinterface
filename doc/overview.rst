Overview
========

Extracellular recordings are an essential source of data in experimental and clinical neuroscience.
Of particular interest in these recordings is the activity of single neurons which must be inferred using a blind source separation procedure called spike sorting.
Given the importance of this processing step, much attention has been directed towards the development of tools and algorithms that can increase its performance and automation.
These developments, however, introduce new challenges in software and file format incompatibility which reduce interoperability,
make utilizing new methods difficult, and preclude reproducible analysis.
To address these limitations, we developed SpikeInterface, a Python framework designed to unify preexisting spike sorting
technologies into a single code base. Users are able to run and compare many popular spike sorting algorithms,
pre-process and post-process extracellular datasets, calculate quality metrics and more, with only a few lines of code,
regardless of the underlying data format. We give an overview of SpikeInterface and show how it can be utilized to improve
the accessibility, reliability, and reproducibility of spike sorting in preparation for the wide-spread use of large-scale
electrophysiology.

SpikeInterface consists of a collection of Python packages aimed to ease the spike sorting process.

There are 5 packages with different scopes and functions:

- `spikeextractors <https://github.com/SpikeInterface/spikeextractors/>`_
- `spiketoolkit <https://github.com/SpikeInterface/spiketoolkit/>`_
- `spikesorters <https://github.com/SpikeInterface/spikesorters/>`_
- `spikecomparison <https://github.com/SpikeInterface/spikecomparison/>`_
- `spikewidgets <https://github.com/SpikeInterface/spikewidgets/>`_

On top of these packages, the :code:`spikeinterface` metapackage allows users to install and use all of the packages,
available as modules as shown in the figure.

.. image:: images/overview.png

