Installation
============

:code:`spikeinterface` is a Python package. It can be installed using pip:

.. code-block:: python

    pip install spikeinterface


If you want to install from the source so that you are up-to-date with the latest development, you can install with:

.. code-block:: bash

    git clone https://github.com/SpikeInterface/spikeinterface
    cd spikeinterface
    python setup.py install (or develop)

Requirements
------------

The following Python packages are required for running the full SpikeInterface framework.
They are installed when using the pip installer for :code:`spikeinterface`.

- spikeextractors
- spiketoolkit
- spikesorters
- spikecomparison
- spikewidgets

You can also install each package from GitHub to keep up with the latest updates. In order to do so, for example for
:code:`spikeextractors`, run:

.. code-block:: bash

    pip uninstall spikeextractors
    git clone https://github.com/SpikeInterface/spikeextractors
    cd spikeextractors
    python setup.py install (or develop)