from .sorterlist import sorter_dict
from pathlib import Path


def run_sorter(sorter_name, recording, output_folder=None,
            remove_existing_folder=False, delete_output_folder=False,
            verbose=False, raise_error=True,  **sorter_params):
    """
    Generic function to run a sorter via function approach.

    >>> sorting = run_sorter('tridesclous', recording)

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
    SorterClass = sorter_dict[sorter_name]
    
    # only classmethod call not instance (stateless at instance level but state is in folder) 
    output_folder = SorterClass.initialize_folder(recording, output_folder, verbose, remove_existing_folder)
    SorterClass.set_params_to_folder(recording, output_folder, sorter_params, verbose)
    SorterClass.setup_recording(recording, output_folder, sorter_params, verbose=verbose)
    SorterClass.run_from_folder(output_folder, raise_error)
    sorting = SorterClass.get_result_from_folder(output_folder)
    
    # TODO : delete_output_folder
    
    
    
    return sorting




def run_hdsort(*args, **kwargs):
    """
    Runs HDsort sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('hdsort')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('hdsort', *args, **kwargs)


def run_klusta(*args, **kwargs):
    """
    Runs klusta sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error 
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('klusta')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('klusta', *args, **kwargs)


def run_tridesclous(*args, **kwargs):
    """
    Runs tridesclous sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error 
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('tridesclous')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('tridesclous', *args, **kwargs)


def run_mountainsort4(*args, **kwargs):
    """
    Runs mountainsort4 sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error 
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('mountainsort4')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('mountainsort4', *args, **kwargs)


def run_ironclust(*args, **kwargs):
    """
    Runs ironclust sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('ironclust')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('ironclust', *args, **kwargs)


def run_kilosort(*args, **kwargs):
    """
    Runs kilosort sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error 
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('kilosort')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('kilosort', *args, **kwargs)


def run_kilosort2(*args, **kwargs):
    """
    Runs kilosort2 sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error 
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('kilosort2')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('kilosort2', *args, **kwargs)

def run_kilosort2_5(*args, **kwargs):
    """
    Runs kilosort2_5 sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('kilosort2')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('kilosort2_5', *args, **kwargs)


def run_kilosort3(*args, **kwargs):
    """
    Runs kilosort3 sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('kilosort3')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('kilosort3', *args, **kwargs)


def run_spykingcircus(*args, **kwargs):
    """
    Runs spykingcircus sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error 
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('spykingcircus')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('spykingcircus', *args, **kwargs)


def run_herdingspikes(*args, **kwargs):
    """
    Runs herdingspikes sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error 
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('herdingspikes')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('herdingspikes', *args, **kwargs)


def run_waveclus(*args, **kwargs):
    """
    Runs waveclus sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error 
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('waveclus')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('waveclus', *args, **kwargs)


def run_combinato(*args, **kwargs):
    """
    Runs combinato sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('combinato')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('combinato', *args, **kwargs)


def run_yass(*args, **kwargs):
    """
    Runs YASS sorter

    Parameters
    ----------
    *args: arguments of 'run_sorter'
        recording: RecordingExtractor
            The recording extractor to be spike sorted
        output_folder: str or Path
            Path to output folder
        delete_output_folder: bool
            If True, output folder is deleted (default False)
        grouping_property: str
            Splits spike sorting by 'grouping_property' (e.g. 'groups')
        parallel: bool
            If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
        verbose: bool
            If True, output is verbose
        raise_error: bool
            If True, an error is raised if spike sorting fails (default). If False, the process continues and the error
            is logged in the log file
        n_jobs: int
            Number of jobs when parallel=True (default=-1)
        joblib_backend: str
            joblib backend when parallel=True (default='loky')
    **kwargs: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params('yass')

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """
    return run_sorter('yass', *args, **kwargs)
