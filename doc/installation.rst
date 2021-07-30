Installation
============

:code:`spikeinterface` is a Python package.

The actual "new API" (v0.90.0) is not released on pypi yet.
It will be released in July 2021.


To use it now, you have to install :code:`spikeinterface` work-in-progress
from source. You also need :code:`neo` and :code:`probeinterface`:


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




Requirements
------------

spiekinterface.core itself has only a few dependencies:

  * numpy
  * neo>=0.9.0
  * joblib
  * probeinterface
  * tqdm

Some sub-modules have more dependencies, so you should also install:

  * scipy
  * h5py
  * pandas
  * sklearn
  * matplotlib
  * networkx
  * datalad
  * MEArec

All sorters must installed independently.
