Welcome to SpikeInterface's documentation!
==========================================


.. image:: images/logo.png
  :scale: 100 %
  :align: center


Spikeinterface is a Python module designed to improve the accessibility, reliability, and reproducibility
of spike sorting and all its associated computations.

With SpikeInterface, users can:

- read/write many extracellular file formats.
- pre-process extracellular recordings.
- run many popular, semi-automatic spike sorters (also in Docker/Singularity containers).
- post-process sorted datasets.
- compare and benchmark spike sorting outputs.
- compute quality metrics to validate and curate spike sorting outputs.
- visualize recordings and spike sorting outputs.
- export report and export to Phy.
- offer a powerful Qt-based viewer in separate package `spikeinterface-gui <https://https://github.com/SpikeInterface/spikeinterface-gui>`_
- have some powerful sorting components to build your own sorter.

Overview of SpikeInterface modules
----------------------------------

.. image:: images/overview.png
  :align: center

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    overview
    installation
    getting_started/plot_getting_started.rst
    modules_doc
    modules/index.rst
    supported_formats_and_sorters
    containerized_sorters
    install_sorters
    viewers
    contribute
    api
    whatisnew
    authors


Other resources
---------------

To get started with SpikeInterface, you can take a look at the following additional resources:

- | `spiketutorials <https://github.com/SpikeInterface/spiketutorials>`_ is a collection of basic and advanced 
  | tutorials. It includes links to videos to dive into the SpikeInterface framework.

- | `SpikeInterface Reports <https://spikeinterface.github.io/>`_ contains several notebooks to reproduce analysis 
  | figures of SpikeInterface-based papers and to showcase the latest features of SpikeInterface.

- | The `2020 eLife paper <https://elifesciences.org/articles/61834>`_ introduces the concept and motivation and 
  | performs an in-depth comparison of multiple sorters (spolier alert: they strongly disagree between each other!). 
  | **Note**: the code-base and implementation have changed a lot since the "paper" version in published in 2020. 
  | For detailed documentation we therefore suggest more recent resources, like this documentation and :code:`spiketutorials`.


Versions and compatibility
--------------------------

We released a major version of SpikeInterface (referred to as **new API**) in July 2021 (version>=0.90) which:

  * breaks backward compatibility with 0.10/0.11/0.12/0.13 series (**old API**)
  * is not a metapackage anymore
  * doesn't depend on spikeextractors/spiketoolkit/spikesorters/spikecomparison/spikewidgets sub-packages

Please see the release notes here: :ref:`release0.90.0`.

For other version release notes check the :ref:`releasenotes`.
  
See the documentation for the version **old API** (version<=0.13) `here <https://spikeinterface.readthedocs.io/en/0.13.0/>`_.


.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
