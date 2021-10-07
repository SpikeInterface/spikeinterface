Installation
============

:code:`spikeinterface` is a Python package.

from pypi
---------

.. code-block:: bash

   pip install spikeinterface[full]


from source
-----------

As spikeinterface is in heavy developement phase, it is sometimes convinient to install from source
to get latest bug fixes and improvements.

It is also recommanded in that case to also install :code:`neo` and :code:`probeinterface` from source.


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

We provide here some installation tips for beginners in python ecosystem here:

https://github.com/SpikeInterface/spikeinterface/tree/master/installation_tips



Requirements
------------

spikeinterface.core itself has only a few dependencies:

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
