Installation
============

:code:`spikeinterface` is a Python package. It can be installed using pip:

.. code-block:: bash

    pip install spikeinterface

The pip installation will install a specific and fixed version of the spikeinterface packages.

To use the latest updates, install `spikeinterface` and the related packages from source:

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

But some sub modules have more dependencies, you should install also:

  * h5py
  * pandas
  * sklearn
