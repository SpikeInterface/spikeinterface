Installation
============

:code:`spikeinterface` is a Python package.

From PyPi
---------

To install the current release version, you can use:

.. code-block:: bash

   pip install spikeinterface[full]

The :code:`[full]` option installs all the extra dependencies for all the different sub-modules.

Note that if using Z shell (:code:`zsh` - the default shell on macOS), you will need to use quotes (:code:`pip install "spikeinterface[full]"`).


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
to get the latest bug fixes and improvements. We recommend constructing the package within a
`virtual environment <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_
to prevent potential conflicts with local dependencies.

.. code-block:: bash


   git clone https://github.com/SpikeInterface/spikeinterface.git
   cd spikeinterface
   pip install -e .
   cd ..

Note that this will install the package in `editable mode <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_.

It is also recommended in that case to also install :code:`neo` and :code:`probeinterface` from source,
as :code:`spikeinterface` strongly relies on these packages to interface with various formats and handle probes:


.. code-block:: bash


   pip install git+https://github.com/NeuralEnsemble/python-neo.git
   pip install git+https://github.com/SpikeInterface/probeinterface.git


It is also sometimes useful to have local copies of :code:`neo` and :code:`probeinterface` to make changes to the code. To achieve this, repeat the first set of commands,
replacing :code:`https://github.com/SpikeInterface/spikeinterface.git` with the appropriate repository in the first code block of this section.

For beginners
-------------

We provide some installation tips for beginners in Python here:

https://github.com/SpikeInterface/spikeinterface/tree/main/installation_tips



Requirements
------------

:code:`spikeinterface.core` itself has only a few dependencies:

  * numpy
  * probeinterface
  * neo
  * threadpoolctl
  * tqdm
  * zarr
  * pydantic
  * numcodecs
  * packaging

Sub-modules have more dependencies, so you should also install:

  * h5py
  * scipy
  * pandas
  * xarray
  * scikit-learn
  * networkx
  * matplotlib
  * numba
  * distinctipy
  * skops
  * huggingface_hub
  * cuda-python (for non-macOS users)


For developers we offer a :code:`[dev]` option which installs testing, documentation, and linting packages necessary
for testing and building the docs.

All external spike sorters can be either run inside containers (Docker or Singularity - see :ref:`containerizedsorters`)
or must be installed independently (see :ref:`get_started/install_sorters:Installing Spike Sorters`).
