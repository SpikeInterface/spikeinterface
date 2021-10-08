Installation
============

:code:`spikeinterface` is a Python package.

From PyPi
---------

.. code-block:: bash

   pip install spikeinterface[full]

The :code:`[full]` option installs all the dependencies for all the different modules. If you wish to only
install the :code:`core` module, you can use:

.. code-block:: bash

   pip install spikeinterface[full]


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

We provide here some installation tips for beginners in Python here:

https://github.com/SpikeInterface/spikeinterface/tree/master/installation_tips



Requirements
------------

:code:`spikeinterface.core` itself has only a few dependencies:

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
