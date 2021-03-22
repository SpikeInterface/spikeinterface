Overview
========

Extracellular recordings are an essential source of data in experimental and clinical neuroscience. 
Of particular interest in these recordings is the activity of single neurons which must be inferred 
using a blind source separation procedure called spike sorting. 

Given the importance of spike sorting, much attention has been directed towards the development of tools 
and algorithms that can increase its performance and automation. These developments, however, introduce new challenges 
in software and file format incompatibility which reduce interoperability, hinder benchmarking, and preclude reproducible analysis.

To address these limitations, we developed **SpikeInterface**, a Python framework designed to unify preexisting spike sorting technologies 
into a single code base and to standardize extracellular data file handling. With a few lines of code, users can run, compare, and benchmark 
most modern spike sorting algorithms; pre-process, post-process, and visualize extracellular datasets; validate, curate, and export sorted results; 
and more, regardless of the underlying data format. 

In the following documentation, we provide an overview of SpikeInterface.


Organization
------------

SpikeInterface consists of 5 main sub-packages which encapsulate all steps in a typical spike sorting pipeline:

- spikeinterface.extrtactors
- spikeinterface.toolkit
- spikeinterface.sorters
- spikeinterface.comparisons
- spikeinterface.widgets


Contrary to previous version (<0.90.0) spikeinterface is one unique package.
Before that spikeinterface was a metapacakge that depended on 5 independent package.

.. image:: images/overview.png


Related projects
-----------------

- `probeinterface <https://probeinterface.readthedocs.io>`_ Handle probe geometry.
- `spikeforest <https://spikeforest.flatironinstitute.org>`_ is a reproducible, continuously updating platform which
  benchmarks the performance of some spike sorting software (kilosort, herdingspike, ironcliust, jrclust, klusta,
  moutainsort4, spykingcircus, tridesclous, waveclus) using many ground-truth datasets. The processing engine is based
  on SpikeInterface.
- `probeinterface <https://github.com/SpikeInterface/probeinterface>`_ is a python package to define and handle neural
   probes and the wiring to recording devices.
- `spikely <https://github.com/SpikeInterface/spikely>`_ is a graphical user interface (GUI) that allows users to build
  and run SpikeInterface spike sorting pipelines on extracellular datasets.
- `spikemetrics <https://github.com/SpikeInterface/spikemetrics>`_ external python package wrapped by SpikeInterface to
  compute quality metrics related to spike sorting output.
- `spikefeatures <https://github.com/SpikeInterface/spikefeatures>`_ external python package wrapped by SpikeInterface
  to compute different features from extracellular action potentials.
- `MEArec <https://mearec.readthedocs.io>`_ is a fast customizable biophysical simulation of extracellular recording.
