"""
Utils functions to launch several sorter on several recording in parallel or not.
"""
import os
from pathlib import Path
import shutil
import numpy as np
import json

from spikeinterface.core import load_extractor, aggregate_units

from .sorterlist import sorter_dict
from .runsorter import run_sorter, _common_param_doc, run_sorter


def _run_one(arg_list):
    # the multiprocessing python module force to have one unique tuple argument
    (sorter_name, recording, output_folder, verbose, sorter_params,
        docker_image, singularity_image, with_output) = arg_list

    if isinstance(recording, dict):
        recording = load_extractor(recording)
    else:
        recording = recording

    # because this is checks in run_sorters before this call
    remove_existing_folder = False
    # result is retrieve later
    delete_output_folder = False
    # because we won't want the loop/worker to break
    raise_error = False

    run_sorter(sorter_name, recording, output_folder=output_folder,
               remove_existing_folder=remove_existing_folder,
               delete_output_folder=delete_output_folder,
               verbose=verbose, raise_error=raise_error,
               docker_image=docker_image, singularity_image=singularity_image,
               with_output=with_output, **sorter_params)




_implemented_engine = ('loop', 'joblib', 'dask')


def run_sorter_by_property(sorter_name,
                           recording,
                           grouping_property,
                           working_folder,
                           mode_if_folder_exists='raise',
                           engine='loop',
                           engine_kwargs={},
                           verbose=False,
                           docker_image=None,
                           singularity_image=None,
                           **sorter_params):
    """
    Generic function to run a sorter on a recording after splitting by a 'grouping_property' (e.g. 'group').

    Internally, the function works as follows:
        * the recording is split based on the provided 'grouping_property' (using the 'split_by' function)
        * the 'run_sorters' function is run on the split recordings
        * sorting outputs are aggregated using the 'aggregate_units' function
        * the 'grouping_property' is added as a property to the SortingExtractor

    Parameters
    ----------
    sorter_name: str
        The sorter name
    recording: BaseRecording
        The recording to be sorted
    grouping_property: object
        Property to split by before sorting
    working_folder: str
        The working directory.
    mode_if_folder_exists: {'raise', 'overwrite', 'keep'}
        The mode when the subfolder of recording/sorter already exists.
            * 'raise' : raise error if subfolder exists
            * 'overwrite' : delete and force recompute
            * 'keep' : do not compute again if f=subfolder exists and log is OK
    engine: {'loop', 'joblib', 'dask'}
        Which engine to use to run sorter.
    engine_kwargs: dict
        This contains kwargs specific to the launcher engine:
            * 'loop' : no kwargs
            * 'joblib' : {'n_jobs' : } number of processes
            * 'dask' : {'client':} the dask client for submitting task
    verbose: bool
        default True
    docker_image: None or str
        If str run the sorter inside a container (docker) using the docker package.
    **sorter_params: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params(sorter_name_or_class)'

    Returns
    -------
    sorting : UnitsAggregationSorting
        The aggregated SortingExtractor.

    Examples
    --------
    This example shows how to run spike sorting split by group using the 'joblib' backend with 4 jobs for parallel
    processing.

    >>> sorting = si.run_sorter_by_property("tridesclous", recording, grouping_property="group",
                                            working_folder="sort_by_group", engine="joblib",
                                            engine_kwargs={"n_jobs": 4})

    """

    assert grouping_property in recording.get_property_keys(), f"The 'grouping_property' {grouping_property} is not " \
                                                               f"a recording property!"
    recording_dict = recording.split_by(grouping_property)

    sorting_output = run_sorters([sorter_name], recording_dict, working_folder,
                                 mode_if_folder_exists=mode_if_folder_exists,
                                 engine=engine,
                                 engine_kwargs=engine_kwargs,
                                 verbose=verbose,
                                 with_output=True,
                                 docker_images={sorter_name: docker_image},
                                 singularity_images={sorter_name: singularity_image},
                                 sorter_params={sorter_name: sorter_params})

    grouping_property_values = np.array([])
    sorting_list = []
    for (output_name, sorting) in sorting_output.items():
        prop_name, sorter_name = output_name
        sorting_list.append(sorting)
        grouping_property_values = np.concatenate(
            (grouping_property_values, [prop_name] * len(sorting.get_unit_ids())))

    aggregate_sorting = aggregate_units(sorting_list)
    aggregate_sorting.set_property(
        key=grouping_property, values=grouping_property_values)

    return aggregate_sorting


def run_sorters(sorter_list,
                recording_dict_or_list,
                working_folder,
                sorter_params={},
                mode_if_folder_exists='raise',
                engine='loop',
                engine_kwargs={},
                verbose=False,
                with_output=True,
                docker_images={},
                singularity_images={},
                ):
    """Run several sorter on several recordings.

    Parameters
    ----------
    sorter_list: list of str
        List of sorter names.
    recording_dict_or_list: dict or list
        If a dict of recording, each key should be the name of the recording.
        If a list, the names should be recording_0, recording_1, etc.
    working_folder: str
        The working directory.
    sorter_params: dict of dict with sorter_name as key
        This allow to overwrite default params for sorter.
    mode_if_folder_exists: {'raise', 'overwrite', 'keep'}
        The mode when the subfolder of recording/sorter already exists.
            * 'raise' : raise error if subfolder exists
            * 'overwrite' : delete and force recompute
            * 'keep' : do not compute again if f=subfolder exists and log is OK
    engine: {'loop', 'joblib', 'dask'}
        Which engine to use to run sorter.
    engine_kwargs: dict
        This contains kwargs specific to the launcher engine:
            * 'loop' : no kwargs
            * 'joblib' : {'n_jobs' : } number of processes
            * 'dask' : {'client':} the dask client for submitting task
    verbose: bool
        Controls sorter verboseness.
    with_output: bool
        Whether to return the output.
    docker_images: dict
        A dictionary {sorter_name : docker_image} to specify if some sorters
        should use docker images.
    singularity_images: dict
        A dictionary {sorter_name : singularity_image} to specify if some sorters
        should use singularity images

    Returns
    -------
    results : dict
        The output is nested dict[(rec_name, sorter_name)] of SortingExtractor.
    """
    working_folder = Path(working_folder)

    mode_if_folder_exists in ('raise', 'keep', 'overwrite')

    if mode_if_folder_exists == 'raise' and working_folder.is_dir():
        raise Exception('working_folder already exists, please remove it')

    assert engine in _implemented_engine, f'engine must be in {_implemented_engine}'

    if isinstance(sorter_list, str):
        sorter_list = [sorter_list]

    for sorter_name in sorter_list:
        assert sorter_name in sorter_dict, f'{sorter_name} is not in sorter list'

    if isinstance(recording_dict_or_list, list):
        # in case of list
        recording_dict = {'recording_{}'.format(
            i): rec for i, rec in enumerate(recording_dict_or_list)}
    elif isinstance(recording_dict_or_list, dict):
        recording_dict = recording_dict_or_list
    else:
        raise ValueError('bad recording dict')

    need_dump = engine != 'loop'
    task_args_list = []
    for rec_name, recording in recording_dict.items():
        for sorter_name in sorter_list:

            output_folder = working_folder / str(rec_name) / sorter_name

            if output_folder.is_dir():
                # sorter folder exists
                if mode_if_folder_exists == 'raise':
                    raise Exception(f'output folder already exists for {rec_name} {sorter_name}')
                elif mode_if_folder_exists == 'overwrite':
                    shutil.rmtree(str(output_folder))
                elif mode_if_folder_exists == 'keep':
                    if is_log_ok(output_folder):
                        continue
                    else:
                        shutil.rmtree(str(output_folder))

            params = sorter_params.get(sorter_name, {})
            docker_image = docker_images.get(sorter_name, None)
            singularity_image = singularity_images.get(sorter_name, None)
            _check_container_images(
                docker_image, singularity_image, sorter_name)

            if need_dump:
                if not recording.is_dumpable:
                    raise Exception(
                        'recording not dumpable call recording.save() before')
                recording_arg = recording.to_dict()
            else:
                recording_arg = recording

            task_args = (sorter_name, recording_arg, output_folder,
                         verbose, params, docker_image, singularity_image, with_output)
            task_args_list.append(task_args)

    if engine == 'loop':
        # simple loop in main process
        for task_args in task_args_list:
            _run_one(task_args)

    elif engine == 'joblib':
        from joblib import Parallel, delayed
        n_jobs = engine_kwargs.get('n_jobs', -1)
        backend = engine_kwargs.get('backend', 'loky')
        Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_run_one)(task_args) for task_args in task_args_list)

    elif engine == 'dask':
        client = engine_kwargs.get('client', None)
        assert client is not None, 'For dask engine you have to provide : client = dask.distributed.Client(...)'

        tasks = []
        for task_args in task_args_list:
            task = client.submit(_run_one, task_args)
            tasks.append(task)

        for task in tasks:
            task.result()

    if with_output:
        if engine == 'dask':
            print('Warning!! With engine="dask" you cannot have directly output results\n'
                  'Use : run_sorters(..., with_output=False)\n'
                  'And then: results = collect_sorting_outputs(output_folders)')
            return

        results = collect_sorting_outputs(working_folder)
        return results


def is_log_ok(output_folder):
    # log is OK when run_time is not None
    if (output_folder / 'spikeinterface_log.json').is_file():
        with open(output_folder / 'spikeinterface_log.json', mode='r', encoding='utf8') as logfile:
            log = json.load(logfile)
            run_time = log.get('run_time', None)
            ok = run_time is not None
            return ok
    return False


def iter_output_folders(output_folders):
    output_folders = Path(output_folders)
    for rec_name in os.listdir(output_folders):
        if not os.path.isdir(output_folders / rec_name):
            continue
        for sorter_name in os.listdir(output_folders / rec_name):
            output_folder = output_folders / rec_name / sorter_name
            if not os.path.isdir(output_folder):
                continue
            if not is_log_ok(output_folder):
                continue
            yield rec_name, sorter_name, output_folder


def iter_sorting_output(output_folders):
    """Iterator over output_folder to retrieve all triplets of (rec_name, sorter_name, sorting)."""
    for rec_name, sorter_name, output_folder in iter_output_folders(output_folders):
        SorterClass = sorter_dict[sorter_name]
        sorting = SorterClass.get_result_from_folder(output_folder)
        yield rec_name, sorter_name, sorting


def collect_sorting_outputs(output_folders):
    """Collect results in a output_folders.

    The output is a  dict with double key access results[(rec_name, sorter_name)] of SortingExtractor.
    """
    results = {}
    for rec_name, sorter_name, sorting in iter_sorting_output(output_folders):
        results[(rec_name, sorter_name)] = sorting
    return results

def _check_container_images(docker_image, singularity_image, sorter_name):
    if docker_image is not None:
        assert singularity_image is None, (f"Provide either a docker or a singularity image "
                                           f"for sorter {sorter_name}")
    if singularity_image is not None:
        assert docker_image is None, (f"Provide either a docker or a singularity image "
                                      f"for sorter {sorter_name}")