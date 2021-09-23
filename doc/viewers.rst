Viewers
=======

There are several ways to plot signals (raw, preprocessed) and spikes.

1. Internally, you can use `spikeinterface.widgets` submodule.
   This creates figures with matplotlib and can be embedded in jupyter notebooks.
2. You can view simple `recording` and `sorting` objects with `ephyviewer`
3. You can use  the`spikeinterface-gui`
4. You can use the `phy` software


spikeinterface.widgets
----------------------

The easiest way to visualize `spikeinterface` objects is to use the `widgets` module for plotting.
You can find many examples in this  tutorial :ref:`_sphx_glr_modules_widgets`.

ephyviewer
----------

`ephyviewer <https://github.com/NeuralEnsemble/ephyviewer>`_ is a custumisable viewer that can 
mix several views togother: signals, spikes, events, video.

`spikeinterface` objects (`recording` and `sorting`) can be loaded directly in `ephyviewer` with a few lines of code.

See this `example <https://ephyviewer.readthedocs.io/en/latest/examples.html#viewers-for-spikeinterface-objects>`_.

Author: Samuel Garcia

spikeinterface-gui
------------------

`spikeinterface-gui <https://github.com/SpikeInterface/spikeinterface-gui>`_ is a local desktop application
which is built on top of spikeinterface.

It is the easiest, fastest way to inspect interactively a spike sorting output.

It's easy to install and ready to use!

Authors: Jeffrey Gill and Samuel Garcia

phy
---

`phy <https://github.com/cortex-lab/phy>`_ is the de-facto standard tool for manual curation of a sorting output.

The current drawback of `phy` is that the dataset (including filtered signals and **all** waveforms of spikes) has to be copied in a separate folder and this is very time consuming process and occupies a lot of disk space.

Author : Cyril Rossant

sortingview (work-in-progress)
-------------------------------

`sortingview <https://github.com/magland/sortingview>`_ is a web-based engine to display the output of a sorter.

It is work-in-progress and is still based on the old `spikeinterface` API  (version<0.90), so currently it cannot be readily used with the current version.


Authors : Jeremy Magland and Jeff Soules
