Implement a spike sorter
--------------------------

Implementing a new spike sorter for a specific file format is as simple as creating a new
subclass based on the predefined base classes provided in the
`spikesorters <https://github.com/SpikeInterface/spikesorters>`_ package.

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
        # can be implemented in subclass for custum checks
        return params


    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return False

        #  optional
        # can be implemented in subclass to check if the filter will be apllied


    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):

        # Fill code to run your spike sorter based on the files created in the _setup_recording()
        # You can run CLI commands (e.g. klusta, spykingcircus, tridescous), pure Python code (e.g. Mountainsort4,
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

    def run_myspikesorter(*args, **kargs):
        return run_sorter('myspikesorter', *args, **kargs)


When you are done you need to write a test in **tests/test_myspikesorter.py**. In order to be tested, you can
install the required packages by changing the **.travis.yml**. Note that MATLAB based tests cannot be run at the moment,
but we recommend testing the implementation locally.

After this you need to add a block in doc/sorters_info.rst

Finally, make a pull request to the spikesorters repo, so we can review the code and merge it to the spikesorters!