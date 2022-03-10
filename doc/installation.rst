Installation
============

:code:`spikeinterface` is a Python package.

From PyPi
---------

To install the current release version, you can use:

.. code-block:: bash

   pip install spikeinterface[full]

The :code:`[full]` option installs all extra dependencies.

Note that if using Z shell (:code:`zsh` - the default shell on mac), you will need to use quotes (:code:`pip install "spikeinterface[full]"`).

To install the natively supported extractors:

.. code-block:: bash

   pip install spikeinterface[extractors]

For spike sorting in docker/singularity containers:

.. code-block:: bash

   pip install spikeinterface[docker | singularity]

Install several modules simultaneously:

.. code-block:: bash

   pip install spikeinterface[full, extractors, docker]

If you wish to only install the :code:`core` module, without optional dependencies, you can use:

.. code-block:: bash

   pip install spikeinterface


From source
-----------

As :code:`spikeinterface` is undergoing a heavy development phase, it is sometimes convenient to install from source
to get latest bug fixes and improvements.

It is also recommended in that case to also install :code:`neo` and :code:`probeinterface` from source,
as :code:`spikeinterface` strongly relies on these packages to interface with various formats and handle probes.


.. code-block:: bash

    git clone https://github.com/NeuralEnsemble/python-neo.git
    cd python-neo
    pip install -e .
    cd ..

    git clone https://github.com/SpikeInterface/probeinterface.git
    cd probeinterface
    pip install -e .
    cd ..

    git clone https://github.com/SpikeInterface/spikeinterface.git
    cd spikeinterface
    pip install -e .
    cd ..


For beginners
-------------

We provide here some installation tips for beginners in Python here:

https://github.com/SpikeInterface/spikeinterface/tree/master/installation_tips


Requirements
------------

:code:`spikeinterface.core` itself has a few dependencies:

  * numpy
  * neo>=0.9.0
  * joblib
  * probeinterface
  * tqdm

Sub-modules have more dependencies, so you should also install:

  * scipy
  * h5py
  * pandas
  * sklearn
  * matplotlib
  * networkx
  * datalad
  * MEArec

All sorters must installed independently.
