Development
==========

How to contribute
-----------------

There are various ways to contribute to SpikeInterface as a user or developer. Some tasks you can help us with include:

* Developing a new extractor for a different file format.
* Creating a new spike sorter.
* Designing a new post-processing algorithm.
* Enhancing documentation, including docstrings, tutorials, and examples.
* Crafting tutorials for common workflows (e.g., spike sorting, post-processing, etc.).
* Writing unit tests to expand code coverage and use case scenarios.
* Reporting bugs and issues.

We use a forking workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow>_ to manage contributions. Here's a summary of the steps involved, with more details available in the provided link:

* Fork the SpikeInterface repository.
* Create a new branch (e.g., :code:git switch -c my-contribution).
* Modify the code, commit, and push changes to your fork.
* Open a pull request from the "Pull Requests" tab of your fork to :code:spikeinterface/main.
* By following this process, we can review the code and even make changes as necessary. 

While we appreciate all the contributions please be mindful of the cost of reviewing pull requests <https://rgommers.github.io/2019/06/the-cost-of-an-open-source-contribution/>_ .


How to run tests locally
-------------------------
Before submitting a pull request, we recommend running the tests locally. In the CI we use pytest to run the tests so it is a good idea to do the same.
To run the tests locally, you can use the following command:

.. code-block:: bash

    pytest

From your local repository. This will run all the tests in the repository. If you want to run a specific test, you can use the following command:

.. code-block:: bash

    pytest path/to/test.py

For example, if you want to run the tests for the :code:`spikeinterface.extractors` module, you can use the following command:

.. code-block:: bash

    pytest src/spikeinterface/extractors

If you want to run a specific test in a specific file, you can use the following command:

.. code-block:: bash

    pytest pytest src/spikeinterface/core/tests/test_baserecording.py::specific_test_in_this_module

We also mantain pytest markers to run specific tests. For example, if you want to run only the tests 
for the :code:`spikeinterface.extractors` module, you can use the following command:

.. code-block:: bash

    pytest -m "extractors"

The markers are located in the :code:`pyproject.toml` file in the root of the repository.

Note that you should install the package before running the tests. You can do this by running the following command:

.. code-block:: bash

    pip install -e .[test,extractors,full]

You can change the :code:`[test,extractors,full]` to install only the dependencies you need. The dependencies are specified in the :code:`pyproject.toml` file in the root of the repository.

The specific environment for the CI is specified in the :code:`.github/actions/build-test-environment/action.yml` and you can 
find the full tests in the :code:`.github/workflows/full_test.yml` file.

The extractor tests require datalad for some of the tests. Here are instructions for installing datalad:

Installing Datalad
------------------

First install the datalad-installer package using pip:

.. code-block:: shell

    pip install datalad-installer

The following instructions depend on the operating system you are using:

Linux
^^^^^
.. code-block:: shell

    datalad-installer --sudo ok git-annex --method datalad/packages

Mac OS
^^^^^^
.. code-block:: shell

    datalad-installer --sudo ok git-annex --method brew

Windows
^^^^^^^

.. code-block:: shell

    datalad-installer --sudo ok git-annex --method datalad/git-annex:release


The following steps are common to all operating systems:

.. code-block:: shell

    pip install datalad

(Optional) Configure Git to use git-annex for large files for efficiency:

.. code-block:: shell

    git config --global filter.annex.process "git-annex filter-process"

Stylistic conventions
---------------------

The following are some styling conventions that we follow in SpikeInterface:

#. Avoid using abreviations in variable names (e.g., use :code:`recording` instead of :code:`rec`). It is specially important to avoid single letter variables.
#. Use index as singular and indices for plural following Numpy. Avoid idx or indexes. Plus, id and ids are reserved for identifiers (i.e. channel_ids)
#. We use file_path and folder_path (instead of file_name and folder_name) for clarity.
#. Use the `numpy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#numpydoc-docstring-guide>`_ in all the docstrings.

How to build the documentation
------------------------------
We use Sphinx to build the documentation. To build the documentation locally, you can use the following command:

.. code-block:: bash

    sphinx-build -b html doc ./doc/_build/

This will build the documentation in the :code:`doc/_build/html` folder. You can open the :code:`index.html` file in your browser to see the documentation.

How to run code coverage locally
--------------------------------
To run code coverage locally, you can use the following command:

.. code-block:: bash

    pytest --cov=spikeinterface --cov-report html

This will run the tests and generate a report in the :code:`htmlcov` folder. You can open the :code:`index.html` file in your browser to see the report.

Note, however, that the running time of the command above will be slow. If you want to run the tests for a specific module, you can use the following command:

.. code-block:: bash

    pytest src/spikeinterface/core/ --cov=spikeinterface/core --cov-report html

Implement a new extractor
-------------------------

SpikeInterface already supports over 30 file formats, but the acquisition system you use might not be among the 
supported formats list (***ref***). Most of the extractord rely on the `NEO <https://github.com/NeuralEnsemble/python-neo>`_ 
package to read information from files.
Therefore, to implement a new extractor to handle the unsupported format, we recommend make a new `neo.rawio `_ class.
Once that is done, the new class can be easily wrapped into SpikeInterface as an extension of the 
:py:class:`~spikeinterface.extractors.neoextractors.neobaseextractors.NeoBaseRecordingExtractor` 
(for :py:class:`~spikeinterface.core.BaseRecording` objects) or 
:py:class:`~spikeinterface.extractors.neoextractors.neobaseextractors.NeoBaseRecordingExtractor` 
(for py:class:`~spikeinterface.core.BaseSorting` objects) or with a few lines of 
code (e.g., see reader for `SpikeGLX <https://github.com/SpikeInterface/spikeinterface/blob/0.96.1/spikeinterface/extractors/neoextractors/spikeglx.py>`_ 
or `Neuralynx <https://github.com/SpikeInterface/spikeinterface/blob/0.96.1/spikeinterface/extractors/neoextractors/neuralynx.py>`_). 

**NOTE:** implementing a `neo.rawio` Class is not required, but recommended. Several extractors (especially) for Sorting 
objects are implemented directly in SpikeInterface and inherit from the base classes.
As examples, see the `CompressedBinaryIblExtractor <https://github.com/SpikeInterface/spikeinterface/blob/0.96.1/spikeinterface/extractors/cbin_ibl.py>`_ 
for a :py:class:`~spikeinterface.core.BaseRecording` object, or the `SpykingCircusSortingExtractor <https://github.com/SpikeInterface/spikeinterface/blob/0.96.1/spikeinterface/extractors/spykingcircusextractors.py>`_ 
for a a :py:class:`~spikeinterface.core.BaseSorting` object.


Implement a spike sorter
------------------------

Implementing a new spike sorter for a specific file format is as simple as creating a new
subclass based on the predefined base class :code:`BaseSorter`.

To enable standardization among subclasses, the :code:`BaseSorter` is base class which require a new
subclass to override a few methods.

The contributed extractors are in the **spikesorters** folder. You can fork the repo and create a new folder
**myspikesorter** there. In the folder, create a new file named **myspikesorter.py**. Additional configuration files
must be placed in the same folder.

You can start by importing the base class:


.. code-block:: python

    import spikeinterface.extractors as se
    from ..basesorter import BaseSorter

In order to check if your spike sorter is installed, a :code:`try` - :code:`except` block is used. For example, if your
sorter is implemented in Python (installed with the package :code:`myspikesorter`), this block will look as follows:

.. code-block:: python

    try:
        import myspikesorter
        HAVE_MSS = True
    except ImportError:
        HAVE_MSS = False

Then, you can start creating a new class:


.. code-block:: python

    class MySpikeSorter(BaseSorter):
    """
    Brief description (optional)
    """

    sorter_name = 'myspikesorter'
    installed = HAVE_MSS

    _default_params = {
        'param1': None,
        'param2': 2,
        }

    _params_description = {
        'param1': 'Description for param1',
        'param1': 'Description for param1',
    }

    installation_mesg = """
        >>> pip install myspikesorter
        More information on MySpikesorter at:
            https://myspikesorterwebsite.com
    """

Now you can start filling out the required methods:

.. code-block:: python

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    # optional
    @classmethod
    def get_sorter_version(cls):
        return myspikesorter.__version__

    @classmethod
    def is_installed(cls):

        # Fill code to check sorter installation. It returns a boolean
        return HAVE_MSS

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):


        # Fill code to set up the recording: convert to required file, parse config files, etc.
        # The files should be placed in the 'output_folder'

    @classmethod
    def _check_params(cls, recording, output_folder, params):
        # optional
        # can be implemented in subclass for custom checks
        return params


    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return False

        #  optional
        # can be implemented in subclass to check if the filter will be applied


    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):

        # Fill code to run your spike sorter based on the files created in the _setup_recording()
        # You can run CLI commands (e.g. klusta, spykingcircus, tridesclous), pure Python code (e.g. Mountainsort5,
        # Herding Spikes), or even MATLAB code (e.g. Kilosort, Kilosort2, Ironclust)

    @classmethod
    def _get_result_from_folder(cls, output_folder):

        # If your spike sorter has a specific file format, you should implement a SortingExtractor in spikeextractors.
        # Let's assume you have done so, and the extractor is called MySpikeSorterSortingExtractor

        sorting = se.MySpikeSorterSortingExtractor(output_folder)
        return sorting

When your spike sorter class is implemented, you have to add it to the list of available spike sorters in the
`sorterlist.py`
Moreover, you have to add a launcher function like `run_XXXX()`.

.. code-block:: python

    def run_myspikesorter(*args, **kwargs):
        return run_sorter('myspikesorter', *args, **kwargs)


When you are done you need to write a test in **tests/test_myspikesorter.py**. In order to be tested, you can
install the required packages by changing the **.travis.yml**. Note that MATLAB based tests cannot be run at the moment,
but we recommend testing the implementation locally.

After this you need to add a block in doc/sorters_info.rst

Finally, make a pull request to the spikesorters repo, so we can review the code and merge it to the spikesorters!

