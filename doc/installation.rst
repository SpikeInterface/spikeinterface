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

The following Python packages are required for running the full SpikeInterface framework.
They are installed when using the pip installer for :code:`spikeinterface`.

- spikeextractors
- spiketoolkit
- spikesorters
- spikecomparison
- spikewidgets


If you installed :code:`spikeinterface` from source, you can install the latest releases of the spikeinterface packages:

.. code-block:: bash

    pip install --upgrade spikeextractors spiketoolkit spikesorters spikecomparison spikewidgets


You can also install each package from GitHub to keep up with the latest updates. In order to do so, for example for
:code:`spikeextractors`, run:

.. code-block:: bash

    pip uninstall spikeextractors
    git clone https://github.com/SpikeInterface/spikeextractors
    cd spikeextractors
    python setup.py install (or develop)