Installation
============

:code:`spikeinterface` is a Python package.

The actual "new API" (v0.90.0) is not release on pypi yet.
It will be released in July 2021. Maybe.


To use it now, you have to install `spikeinterface` work-in-progress
from source but also neo and probeinterface:


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

spiekinterface.core irself have few dependencies:

  * numpy
  * neo>=0.9.0
  * joblib
  * probeinterface
  * tqdm

But some sub modules have more dependencies, you should install also:

  * scipy
  * h5py
  * pandas
  * sklearn
  * matplotlib
  * networkx
  * datalad
  * MEArec

All sorters must installed independantly.
