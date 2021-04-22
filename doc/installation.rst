Installation
============

:code:`spikeinterface` is a Python package.

The actual "new API" (v0.90.0) is not release on pypi yet.
It will release
 in July 2021

To use it now, you have to install `spikeinterface` from source:

.. code-block:: bash

    git clone https://github.com/SpikeInterface/spikeinterface.git
    cd spikeinterface
    python setup.py install (or develop)



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
