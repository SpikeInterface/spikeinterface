Build a SortingExtractor
------------------------------------------

Building a new :code:`SortingExtractor` for a specific file format is as simple as creating a new
subclass based on the predefined base classes provided in the
`spikeextractors <https://github.com/SpikeInterface/spikeextractors>`_ package.

To enable standardization among subclasses, the :code:`SortingExtractor` is an abstract base class which require a new
subclass to **override all methods which are decorated with @abstractmethod**. The :code:`SortingExtractor` class has two abstract methods: :code:`get_unit_ids()`, :code:`get_unit_spike_trains()`. So all you need to do is create a class that inherits from :code:`:code:`SortingExtractor`` and implements these two methods.

Along with these two methods, you can also optionally override the :code:`write_sorting()` function which enables any :code:`SortingExtractor` to be written into your format.

Any other methods, such as :code:`set_unit_spike_features()` or :code:`clear_unit_property()`, **should not** be overwritten as they are generic functions that any :code:`SortingExtractor` has access to upon initialization.

Finally, if your file format contains information about the units (e.g. location, morphology, etc.) or spikes (e.g. locations, pcs, etc.), you are suggested to add that as either unit properties or spike features upon initialization (this is optional).

The contributed extractors are in the **spikeextractors/extractors** folder. You can fork the repo and create a new folder
**myformatextractors** there. In the folder, create a new file named **myformatsortingextractor.py**.

.. code-block:: python

    from spikeextractors import SortingExtractor
    from spikeextractors.extraction_tools import check_get_unit_spike_train

     try:
        import mypackage
        HAVE_MYPACKAGE = True
    except ImportError:
        HAVE_MYPACKAGE = False

    class MyFormatSortingExtractor(SortingExtractor):
        """
        Description of your sorting extractor

        Parameters
        ----------
        file_path: str or Path
            Path to myformat file
        extra_parameter_1: (type)
            What extra_parameter_1 does
        extra_parameter_2: (type)
            What extra_parameter_2 does
        """
        extractor_name = 'MyFormatSorting'
        installed = HAVE_MYPACKAGE  # check at class level if installed or not
        is_writable = True # set to True if extractor implements `write_sorting()` function
        mode = 'file'  # 'file' if input is 'file_path', 'folder' if input 'folder_path', 'file_or_folder' if input is 'file_or_folder_path'
        installation_mesg = "To use the MyFormatSortingExtractor extractors, install mypackage: \n\n pip install mypackage\n\n"

        def __init__(self, file_path, extra_parameter_1, extra_parameter_2):
            # check if installed
            assert self.installed, self.installation_mesg

            # instantiate base SortingExtractor
            SortingExtractor.__init__(self)

            ## All file specific initialization code can go here.
            # If your format stores the sampling frequency, you can overwrite the self._sampling_frequency. This way,
            # the base method self.get_sampling_frequency() will return the correct sampling frequency

            self._sampling_frequency = my_sampling_frequency

        def get_unit_ids(self):

            #Fill code to get a unit_ids list containing all the ids (ints) of detected units in the recording

            return unit_ids

        @check_get_unit_spike_train
        def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):

            '''Code to extract spike frames from the specified unit.
            It will return spike frames from within three ranges:
                [start_frame, t_start+1, ..., end_frame-1]
                [start_frame, start_frame+1, ..., final_unit_spike_frame - 1]
                [0, 1, ..., end_frame-1]
                [0, 1, ..., final_unit_spike_frame - 1]
            if both start_frame and end_frame are given, if only start_frame is
            given, if only end_frame is given, or if neither start_frame or end_frame
            are given, respectively. Spike frames are returned in the form of an
            array_like of spike frames. In this implementation, start_frame is inclusive
            and end_frame is exclusive conforming to numpy standards.

            '''

            return spike_train

        .
        .
        .
        .
        . #Optional functions and pre-implemented functions that a new SortingExtractor doesn't need to implement
        .
        .
        .
        .

        @staticmethod
        def write_sorting(sorting, save_path):
            '''
            This is an example of a function that is not abstract so it is optional if you want to override it. It allows other
            SortingExtractors to use your new SortingExtractor to convert their sorted data into your
            sorting file format.
            '''


When you are done you can optionally write a test in the **tests/test_extractors.py** (this is easier if a
:code:`write_sorting` function is implemented).

Finally, make a pull request to the spikeextractors repo, so we can review the code and merge it to the spikeextractors!
