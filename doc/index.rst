Welcome to SpikeInterface's documentation!
==========================================



.. warning::

    This is the documentation for SpikeInterface (0.90.0), which breaks compatibility with version 0.13.0.
    
    See the documentation for the version 0.13.0 (old API) `here <https://spikeinterface.readthedocs.io/en/0.13.0/>`_.

.. image:: images/logo.png
  :scale: 100 %
  :align: center


Spikeinterface is a collection of Python modules designed to improve the accessibility, reliability, and reproducibility
of spike sorting and all its associated computations.



With SpikeInterface, users can:

- read/write many extracellular file formats.
- pre-process extracellular recordings.
- run many popular, semi-automatic spike sorters.
- post-process sorted datasets
- compare and benchmark spike sorting outputs.
- compute quality metrics to validate and curate spike sorting outputs
- visualize recordings and spike sorting outputs.
- export report and export to phy with one line

**NEWS**

- New SpikeInterface major release! Version 0.90.0 is out!

  * breaks backward compatility with 0.10/0.11/0.12/0.13 series
  * has been released on 31st July 2021
  * is not a metapackage anymore
  * doesn't depend on spikeextractors/spiketoolkit/spikesorters/spikecomparison/spikewidgets sub-packages

  Please see the release notes here: :ref:`release0.90.0`

- spikeinterface have now its own GUI see `spikeinterface-gui <https://https://github.com/SpikeInterface/spikeinterface-gui>`_

.. toctree::
    :maxdepth: 2
    :caption: Contents:
   
    overview
    installation
    getting_started/plot_getting_started.rst
    modules_doc
    modules/index.rst
    supported_formats_and_sorters
    install_sorters
    viewers
    contribute
    api
    whatisnew
    authors


For more information, please have a look at:

- The `eLife paper <https://elifesciences.org/articles/61834>`_

- 1-hour `video tutorial <https://www.youtube.com/watch?v=fvKG_-xQ4D8&t=3364s&ab_channel=NeurodataWithoutBorders>`_, recorded for the NWB User Days (Sep 2020)
   
- A collection of analysis notebook `SpikeInterface Reports <https://spikeinterface.github.io/>`_

.. Indices and tables
.. ==================
.. 
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
