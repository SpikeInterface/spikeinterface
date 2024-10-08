Generation module
=================

The :py:mod:`spikeinterface.generation` provides functions to generate recordings containing spikes,
which can be used as "ground-truth" for benchmarking spike sorting algorithms.

There are several approaches to generating such recordings.
One possibility is to generate purely synthetic recordings. Another approach is to use real
recordings and add synthetic spikes to them, to make "hybrid" recordings.
The advantage of the former is that the ground-truth is known exactly, which is useful for benchmarking.
The advantage of the latter is that the spikes are added to real noise, which can be more realistic.

For hybrid recordings, the main challenge is to generate realistic spike templates.
We therefore built an open database of templates that we have constructed from the International
Brain Laboratory - Brain Wide Map (available on
`DANDI <https://dandiarchive.org/dandiset/000409?search=IBL&page=2&sortOption=0&sortDir=-1&showDrafts=true&showEmpty=false&pos=9>`_).
You can check out this collection of over 600 templates from this `web app <https://spikeinterface.github.io/hybrid_template_library/>`_.

The :py:mod:`spikeinterface.generation` module offers tools to interact with this database to select and download templates,
manupulating (e.g. rescaling and relocating them), and construct hybrid recordings with them.
Importantly, recordings from long-shank probes, such as Neuropixels, usually experience drifts.
Such drifts can be taken into account in order to smoothly inject spikes into the recording.

The :py:mod:`spikeinterface.generation` also includes functions to generate different kinds of drift signals and drifting
recordings, as well as generating synthetic noise profiles of various types.

Some of the generation functions are defined in the :py:mod:`spikeinterface.core.generate` module, but also exposed at the
:py:mod:`spikeinterface.generation` level for convenience.
