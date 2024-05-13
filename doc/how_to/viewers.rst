Visualise Data
==============

There are several ways to plot signals (raw, preprocessed) and spikes.

1. Internally, you can use :code:`spikeinterface.widgets` submodule.
   This creates static figures with :code:`matplotlib`, interactive widgets with :code:`ipywidgets`,
   and web-based shareable views with :code:`sortingview`.
2. You can view simple :code:`recording` and :code:`sorting` objects with :code:`ephyviewer`
3. You can use  the :code:`spikeinterface-gui`
4. You can use the :code:`phy` software


spikeinterface.widgets
----------------------

The easiest way to visualize :code:`spikeinterface` objects is to use the :code:`widgets` module for plotting.
You can find an extensive description in the module documentation :ref:`modulewidgets`
and many examples in the :code:`Widgets tutorials` section of the :code:`Modules example gallery`.

spikeinterface-gui
------------------

`spikeinterface-gui <https://github.com/SpikeInterface/spikeinterface-gui>`_ is a local desktop application
which is built on top of :code:`spikeinterface`.

It is the easiest and fastest way to interactively inspect a spike sorting output.
It's easy to install and ready to use!

Authors: Samuel Garcia

ephyviewer
----------

`ephyviewer <https://github.com/NeuralEnsemble/ephyviewer>`_ is a customizable viewer that can
mix several views together: signals, spikes, events, video.

:code:`spikeinterface` objects (:code:`recording` and :code:`sorting`) can be loaded directly in :code:`ephyviewer` with a few lines of code.
See this `example <https://ephyviewer.readthedocs.io/en/latest/examples.html#viewers-for-spikeinterface-objects>`_.

Author: Jeffrey Gill and Samuel Garcia

phy
---

`phy <https://github.com/cortex-lab/phy>`_ is the de-facto standard tool for manual curation of a sorting output.
The current drawback of :code:`phy` is that the dataset (including filtered signals and **all** waveforms of spikes) has to be copied
in a separate folder and this is very time consuming process and occupies a lot of disk space.

Author : Cyrill Rossant
