Welcome to SpikeInterface's documentation!
==========================================



.. warning::

    This is the not yet released documentation of spikeinterface (0.90.0)
    
    To see reelased documention this is here https://spikeinterface.readthedocs.io/en/stable
    
    Actual  API released is 0.12.0.

.. image:: images/logo.png
  :scale: 100 %
  :align: center


Spikeinterface is a collection of Python modules designed to improve the accessibility, reliability, and reproducibility of spike sorting and all its associated computations.



With SpikeInterface, users can:

- read/write many extracellular file formats.
- pre-process extracellular recordings.
- run many popular, semi-automatic spike sorters.
- post-process sorted datasets
- compare and benchmark spike sorting outputs.
- compute quliy metrics to validate and curate spike sorting outputs
- visualize recordings and spike sorting outputs.
- export report and export to phy with one line

**NEWS**

- New SpikeInterface major release! Version 0.90.0 is will be out soon!

  * breaks backward compatility with 0.10/0.11/0.12 series
  * will be released in summer 2021
  * is not a metapackage anymore
  * doesn't depend on spikeextractors/spiketoolkit/spikesorters/spikecomparison/spikewidgets sub-packages


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   overview
   installation
   getting_started/plot_getting_started.rst
   module_core
   module_extractors
   module_sorter
   module_toolkit
   module_comparison
   module_exporters
   module_sorting_components
   modules/index.rst
   supported_formats_and_sorters
   install_sorters
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
