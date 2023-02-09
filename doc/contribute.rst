Contribute
==========

To contribute to SpikeInterface, a user/developer can help us integrate in a new recorded file
format, a new sorted file format, or a new spike sorting algorithm or more!

In order to make a contribution, you first need to **fork** the SpikeInterface repository.
Then you can clone from your fork, make a new branch (e.g. :code:`git checkout -b my-contribution`),
make changes to the code, commit and push to your fork.
Next, you can open a pull request (PR) from the "Pull Requests" tab of your fork to :code:`spikeinterface/master`.
This way, we'll be able to review the code and even make changes.


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
        # You can run CLI commands (e.g. klusta, spykingcircus, tridesclous), pure Python code (e.g. Mountainsort4,
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

