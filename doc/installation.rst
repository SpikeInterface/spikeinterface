Installation
============

:code:`spikeinterface` is a Python package.

From PyPi
---------

To install the current release version, you can use:

.. code-block:: bash

   pip install spikeinterface[full]

The :code:`[full]` option installs all the extra dependencies for all the different sub-modules. 

Note that if using Z shell (:code:`zsh` - the default shell on mac), you will need to use quotes (:code:`pip install "spikeinterface[full]"`).


To install all interactive widget backends, you can use:

.. code-block:: bash

   pip install spikeinterface[full,widgets]

Note that the :code:`[widgets]` option also installs jupyter (and relative dependencies).


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
    python setup.py install (or develop)
    cd ..

    git clone https://github.com/SpikeInterface/probeinterface.git
    cd probeinterface
    python setup.py install (or develop)
    cd ..

    git clone https://github.com/SpikeInterface/spikeinterface.git
    cd spikeinterface
    python setup.py install (or develop)
    cd ..


For beginners
-------------

We provide some installation tips for beginners in Python here:

https://github.com/SpikeInterface/spikeinterface/tree/master/installation_tips



Requirements
------------

:code:`spikeinterface.core` itself has only a few dependencies:

  * numpy
  * probeinterface
  * neo>=0.9.0
  * joblib
  * threadpoolctl
  * tqdm

Sub-modules have more dependencies, so you should also install:

  * zarr
  * scipy
  * pandas
  * xarray
  * sklearn
  * networkx
  * matplotlib


All external spike sorters can be either run inside containers (Docker or Singularity - see :ref:`containerizedsorters`) 
or must be installed independently (see :ref:`installsorters`).
