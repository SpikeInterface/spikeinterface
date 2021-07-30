"""
Utils functions to launch several sorter on several recording in parralell or not.
"""
import os
from pathlib import Path
import multiprocessing
import shutil
import json
import traceback
import json

from spikeinterface.core import load_extractor

from .sorterlist import sorter_dict


def _run_one(arg_list):
    # the multiprocessing python module force to have one unique tuple argument
    sorter_name, recording, output_folder, verbose, sorter_params = arg_list
    if isinstance(recording, dict):
        recording = load_extractor(recording)
    else:
        recording = recording

    SorterClass = sorter_dict[sorter_name]

    # because this is checks in run_sorters before this call
    remove_existing_folder = False
    # result is retrieve later
    delete_output_folder = False
    # because we won't want the loop/worker to break
    raise_error = False

    # only classmethod call not instance (stateless at instance level but state is in folder)
    output_folder = SorterClass.initialize_folder(recording, output_folder, verbose, remove_existing_folder)
    SorterClass.set_params_to_folder(recording, output_folder, sorter_params, verbose)
    SorterClass.setup_recording(recording, output_folder, verbose=verbose)
    SorterClass.run_from_folder(output_folder, raise_error, verbose)


_implemented_engine = ('loop', 'joblib', 'dask')


def run_sorters(sorter_list,
                recording_dict_or_list,
                working_folder,
                sorter_params={},
                mode_if_folder_exists='raise',
                engine='loop',
                engine_kwargs={},
                verbose=False,
                with_output=True,
                ):
    """
    This run several sorter on several recording.
    Simple implementation are nested loops or with multiprocessing.

    sorter_list: list of str (sorter names)
    recording_dict_or_list: a dict (or a list) of recording
    working_folder : str

    engine = None ( = 'loop') or 'multiprocessing'
    processes = only if 'multiprocessing' if None then processes=os.cpu_count()
    verbose=True/False to control sorter verbosity

    Note: engine='multiprocessing' use the python multiprocessing module.
    This do not allow to have subprocess in subprocess.
    So sorter that already use internally multiprocessing, this will fail.

    Parameters
    ----------

    sorter_list: list of str
        List of sorter name.

    recording_dict_or_list: dict or list
        A dict of recording. The key will be the name of the recording.
        In a list is given then the name will be recording_0, recording_1, ...

    working_folder: str
        The working directory.

    sorter_params: dict of dict with sorter_name as key
        This allow to overwrite default params for sorter.

    mode_if_folder_exists: 'raise_if_exists' or 'overwrite' or 'keep'
        The mode when the subfolder of recording/sorter already exists.
            * 'raise' : raise error if subfolder exists
            * 'overwrite' : delete and force recompute
            * 'keep' : do not compute again if f=subfolder exists and log is OK

    engine: str
        'loop', 'joblib', or 'dask'

    engine_kwargs: dict
        This contains kwargs specific to the launcher engine:
            * 'loop' : no kwargs
            * 'joblib' : {'n_jobs' : } number of processes
            * 'dask' : {'client':} the dask client for submiting task
            
    verbose: bool
        default True

    with_output: bool
        return the output.

    run_sorter_kwargs: dict
        This contains kwargs specific to run_sorter function:\
            * 'raise_error' :  bool
            * 'parallel' : bool
            * 'n_jobs' : int
            * 'joblib_backend' : 'loky' / 'multiprocessing' / 'threading'

    Returns
    ----------

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
        recording_dict = {'recording_{}'.format(i): rec for i, rec in enumerate(recording_dict_or_list)}
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
                    raise (Exception('output folder already exists for {} {}'.format(rec_name, sorter_name)))
                elif mode_if_folder_exists == 'overwrite':
                    shutil.rmtree(str(output_folder))
                elif mode_if_folder_exists == 'keep':
                    if is_log_ok(output_folder):
                        continue
                    else:
                        shutil.rmtree(str(output_folder))

            params = sorter_params.get(sorter_name, {})
            if need_dump:
                if not recording.is_dumpable:
                    raise Exception('recording not dumpable call recording.save() before')
                recording_arg = recording.to_dict()
            else:
                recording_arg = recording
            task_args = (sorter_name, recording_arg, output_folder, verbose, params)
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
            print('Warning!! With engine="dask" you cannot have directly output results\n' \
                  'Use : run_sorters(..., with_output=False)\n' \
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
    """
    Iterator over output_folder to retrieve all triplets
    (rec_name, sorter_name, sorting)
    """
    for rec_name, sorter_name, output_folder in iter_output_folders(output_folders):
        SorterClass = sorter_dict[sorter_name]
        sorting = SorterClass.get_result_from_folder(output_folder)
        yield rec_name, sorter_name, sorting


def collect_sorting_outputs(output_folders):
    """
    Collect results in a output_folders.

    The output is a  dict with double key access results[(rec_name, sorter_name)] of SortingExtractor.
    """
    results = {}
    for rec_name, sorter_name, sorting in iter_sorting_output(output_folders):
        results[(rec_name, sorter_name)] = sorting
    return results
