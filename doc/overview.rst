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

SpikeInterface consists several sub-packages which encapsulate all steps in a typical spike sorting pipeline:

- spikeinterface.extractors
- spikeinterface.preprocessing
- spikeinterface.sorters
- spikeinterface.postprocessing
- spikeinterface.qualitymetrics
- spikeinterface.comparisons
- spikeinterface.widgets


Contrary to the previous version (<0.90.0), :code:`spikeinterface` is now one unique package.
Before that, :code:`spikeinterface` was a metapackage that depended on 5 independent packages.



Related projects
----------------

- `probeinterface <https://github.com/SpikeInterface/probeinterface>`_ is a python package to define and handle neural
   probes and the wiring to recording devices.
- `spikeforest <https://spikeforest.flatironinstitute.org>`_ is a reproducible, continuously updating platform which
  benchmarks the performance of some spike sorting software (kilosort, herdingspike, ironclust, jrclust, klusta,
  mountainsort4, spykingcircus, tridesclous, waveclus) using many ground-truth datasets. The processing engine is based
  on SpikeInterface.
- `spikely <https://github.com/SpikeInterface/spikely>`_ is a graphical user interface (GUI) that allows users to build
  and run SpikeInterface spike sorting pipelines on extracellular datasets.
- `MEArec <https://mearec.readthedocs.io>`_ is a fast customizable biophysical simulation of extracellular recording.
