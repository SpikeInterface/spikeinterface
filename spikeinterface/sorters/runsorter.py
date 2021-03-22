from spikeinterface.core import BaseRecording
from .sorterlist import sorter_dict


_common_param_doc = """
    Parameters
    ----------
    sorter_name: str
        The sorter name
    recording: RecordingExtractor
        The recording extractor to be spike sorted
    output_folder: str or Path
        Path to output folder
    remove_existing_folder: bool
        Tf True and output_folder exists yet then delete.
    delete_output_folder: bool
        If True, output folder is deleted (default False)
    verbose: bool
        If True, output is verbose
    raise_error: bool
        If True, an error is raised if spike sorting fails (default). If False, the process continues and the error is
        logged in the log file.
    **sorter_params: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params(sorter_name_or_class)'

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data

    """

def run_sorter(sorter_name, recording, output_folder=None,
            remove_existing_folder=True, delete_output_folder=False,
            verbose=False, raise_error=True,  **sorter_params):
    """
    Generic function to run a sorter via function approach.

    >>> sorting = run_sorter('tridesclous', recording)
    """ + _common_param_doc
    
    if isinstance(recording, list):
        raise Exception('You you want to run several sorters/recordings use run_sorters(...)')

    SorterClass = sorter_dict[sorter_name]
    
    # only classmethod call not instance (stateless at instance level but state is in folder) 
    output_folder = SorterClass.initialize_folder(recording, output_folder, verbose, remove_existing_folder)
    SorterClass.set_params_to_folder(recording, output_folder, sorter_params, verbose)
    SorterClass.setup_recording(recording, output_folder, verbose=verbose)
    SorterClass.run_from_folder(output_folder, raise_error, verbose)
    sorting = SorterClass.get_result_from_folder(output_folder)
    
    if delete_output_folder:
        raise NotImplementedError
        # TODO : delete_output_folder
    
    return sorting


_common_run_doc =     """
    Runs {} sorter
    """ + _common_param_doc


def run_hdsort(*args, **kwargs):
    __doc__ = _common_run_doc.format('hdsort')
    return run_sorter('hdsort', *args, **kwargs)


def run_klusta(*args, **kwargs):
    __doc__ = _common_run_doc.format('klusta')
    return run_sorter('klusta', *args, **kwargs)


def run_tridesclous(*args, **kwargs):
    __doc__ = _common_run_doc.format('tridesclous')
    return run_sorter('tridesclous', *args, **kwargs)


def run_mountainsort4(*args, **kwargs):
    __doc__ = _common_run_doc.format('mountainsort4')
    return run_sorter('mountainsort4', *args, **kwargs)


def run_ironclust(*args, **kwargs):
    __doc__ = _common_run_doc.format('ironclust')
    return run_sorter('ironclust', *args, **kwargs)


def run_kilosort(*args, **kwargs):
    __doc__ = _common_run_doc.format('kilosort')
    return run_sorter('kilosort', *args, **kwargs)


def run_kilosort2(*args, **kwargs):
    __doc__ = _common_run_doc.format('kilosort2')
    return run_sorter('kilosort2', *args, **kwargs)

def run_kilosort2_5(*args, **kwargs):
    __doc__ = _common_run_doc.format('kilosort2_5')
    return run_sorter('kilosort2_5', *args, **kwargs)


def run_kilosort3(*args, **kwargs):
    __doc__ = _common_run_doc.format('kilosort3')
    return run_sorter('kilosort3', *args, **kwargs)


def run_spykingcircus(*args, **kwargs):
    __doc__ = _common_run_doc.format('spykingcircus')
    return run_sorter('spykingcircus', *args, **kwargs)


def run_herdingspikes(*args, **kwargs):
    __doc__ = _common_run_doc.format('herdingspikes')
    return run_sorter('herdingspikes', *args, **kwargs)


def run_waveclus(*args, **kwargs):
    __doc__ = _common_run_doc.format('waveclus')
    return run_sorter('waveclus', *args, **kwargs)


def run_combinato(*args, **kwargs):
    __doc__ = _common_run_doc.format('combinato')
    return run_sorter('combinato', *args, **kwargs)


def run_yass(*args, **kwargs):
    __doc__ = _common_run_doc.format('yass')
    return run_sorter('yass', *args, **kwargs)
