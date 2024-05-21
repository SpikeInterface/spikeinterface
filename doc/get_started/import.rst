Importing SpikeInterface
========================

SpikeInterface allows for the generation of powerful and reproducible spike sorting pipelines.
Flexibility is built into the package starting from import to maximize the productivity of
the developer and the scientist. Thus there are three ways that SpikeInterface and its components
can be imported:


Importing by Module
-------------------

Since each spike sorting pipeline involves a series of often repeated steps, many of the developers
working on SpikeInterface recommend importing in a module by module fashion. This will allow you to
keep track of your processing steps (preprocessing, postprocessing, quality metrics, etc.). This can
be accomplished by:

.. code-block:: python

    import spikeinterface.core as si

to import the :code:`core` module followed by:

.. code-block:: python

    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre
    import spikeinterface.sorters as ss
    import spikinterface.postprocessing as spost
    import spikeinterface.qualitymetrics as sqm
    import spikeinterface.exporters as sexp
    import spikeinterface.comparsion as scmp
    import spikeinterface.curation as scur
    import spikeinterface.sortingcomponents as sc
    import spikeinterface.widgets as sw

to import any of the other modules you wish to use.

The benefit of this approach is that it is lighter than importing the whole library as a flat module and allows
you to choose which of the modules you actually want to use. It also reminds you what step of the pipeline each
submodule is meant to be used for. If you don't plan to export the results out of SpikeInterface then you
don't have to :code:`import spikeinterface.exporters`. Additionally the documentation of SpikeInterface is set-up
in a modular fashion, so if you have a problem with the submodule  :code:`spikeinterface.curation`,you will know
to go to the :code:`curation` section of this documention. The disadvantage of this approach is that you have
more aliases to keep track of.


Flat Import
-----------

A second option is to import the SpikeInterface package in :code:`full` mode.
To accomplish this one does:


.. code-block:: python

    import spikeinterface.full as si


This import statement will import all of the SpikeInterface modules as one flattened module.
We recommend this approach for advanced (or lazy) users, since it requires a deeper knowledge of the API. The advantage
being that users can access all functions using one alias without the need of memorizing all aliases.


Importing Individual Functions
------------------------------

Finally, some users may find it useful to have extremely light imports and only import the exact functions
they plan to use. This can easily be accomplished by importing functions directly into the name space.

For example:

.. code-block:: python

    from spikeinterface.preprocessing import bandpass_filter, common_reference
    from spikeinterface.core import extract_waveforms
    from spikeinterface.extractors import read_binary

As mentioned this approach only imports exactly what you plan on using so it is the most minimalist. It does require
knowledge of the API to know which module to pull a function from. It could also lead to naming clashes if pulling
functions directly from other scientific libraries. Type :code:`import this` for more information.
