Viewers
=======

There are several way to plot signals (raw, preprocessed) and spikes.

1. Internaly you can use `spikeinterface.widgets` submodule.
   This create figure with matplotlib and so can be embeded in jupyter notebook.
2. You can view simple `recording` and `sorting` objects with ephyviewer
3. You can use `spikeinterface-gui`
4. You can use `phy`


spikeinterface.widgets
----------------------

The easiest way is to use this module for plotting.
Many examples are in this  tutorial :ref:`_sphx_glr_modules_widgets`.

ephyviewer
----------

`ephyviewer <https://github.com/NeuralEnsemble/ephyviewer>`_ is a custumisable viewer that can 
mix several views togother : signals, spikes, events, video.

spikeinterface objects (`recording` and `sorting`) can be used directly in ephyviewer with few lines.

See this `example <https://ephyviewer.readthedocs.io/en/latest/examples.html#viewers-for-spikeinterface-objects>`_

Author: Samuel Garcia

spikeinterface-gui
------------------

`spikeinterface-gui <https://github.com/SpikeInterface/spikeinterface-gui>`_ is a local desktop application
which is build on top of spikeinterface.

It is the easiest, fastest way to inspect interactivelly a sorting output.

Install and launch take 2 min.

Authors: Jeffrey Gill and Samuel Garcia

phy
---

`phy <https://github.com/cortex-lab/phy>`_ is the de-facto standard tools for manual curation of a sorting output.

The drawback of phy : the dataset (include filtered signals and **all** waveforms of spikes) have to be copy in a
sperate folder and this is very time consuming and disk-space-eater.

Author : Cyril Rossant

sortingview (work-in-progress)
------------------------------

`sortingview <https://github.com/magland/sortingview>`_ is a web based engine to display output of a sorter.

It is work-in-progress and is based on the old API of spikeinterface. So it cannot be used with the current version.


Authors : Jeremy Magland and Jeff Soules
