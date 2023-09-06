"""
Utils functions to launch several sorter on several recording in parallel or not.
"""
from pathlib import Path
import shutil
import numpy as np
import json
import tempfile
import os
import stat
import subprocess
import sys
import warnings

from spikeinterface.core import load_extractor, aggregate_units
from spikeinterface.core.core_tools import check_json

from .sorterlist import sorter_dict
from .runsorter import run_sorter
from .basesorter import is_log_ok

_implemented_engine = ("loop", "joblib", "dask", "slurm")

def run_sorter_jobs(job_list, engine="loop", engine_kwargs={}, return_output=False):
    """
    Run several :py:func:`run_sorter()` sequencially or in parralel given a list of job.

    For **engine="loop"** this is equivalent to:

    ..code::

        for job in job_list:
            run_sorter(**job)
    
    For some engines, this function is blocking until the results ("loop", "joblib", "multiprocessing", "dask").
    For some other engine ("slurm") the function return almost immediatly (akak non blocking) and the results
    must be retrieve by hand when finished with :py:func:`read_sorter_folder()`.

    Parameters
    ----------
    job_list: list of dict
        A list a dict that are propagated to run_sorter(...)
    engine: str "loop", "joblib", "dask", "slurm"
        The engine to run the list.
        * "loop": a simple loop. This engine is 
    engine_kwargs: dict

    return_output: bool, dfault False
        Return a sorting or None.

    Returns
    -------
    sortings: None or list of sorting
        With engine="loop" or "joblib" you can optional get directly the list of sorting result if return_output=True.
    """

    assert engine in _implemented_engine, f"engine must be in {_implemented_engine}"

    if return_output:
        assert engine in ("loop", "joblib", "multiprocessing")
        out = []
    else:
        out = None

    if engine == "loop":
        # simple loop in main process
        for kwargs in job_list:
            sorting = run_sorter(**kwargs)
            if return_output:
                out.append(sorting)

    elif engine == "joblib":
        from joblib import Parallel, delayed

        n_jobs = engine_kwargs.get("n_jobs", -1)
        backend = engine_kwargs.get("backend", "loky")
        sortings = Parallel(n_jobs=n_jobs, backend=backend)(delayed(run_sorter)(**kwargs) for kwargs in job_list)
        if return_output:
            out.extend(sortings)

    elif engine == "multiprocessing":
        raise NotImplementedError()

    elif engine == "dask":
        client = engine_kwargs.get("client", None)
        assert client is not None, "For dask engine you have to provide : client = dask.distributed.Client(...)"

        tasks = []
        for kwargs in job_list:
            task = client.submit(run_sorter, **kwargs)
            tasks.append(task)

        for task in tasks:
            task.result()

    elif engine == "slurm":
        # generate python script for slurm
        tmp_script_folder = engine_kwargs.get("tmp_script_folder", None)
        if tmp_script_folder is None:
            tmp_script_folder = tempfile.mkdtemp(prefix="spikeinterface_slurm_")
        tmp_script_folder = Path(tmp_script_folder)
        cpus_per_task = engine_kwargs.get("cpus_per_task", 1)
        mem = engine_kwargs.get("mem", "1G")

        tmp_script_folder.mkdir(exist_ok=True, parents=True)

        # for i, task_args in enumerate(task_args_list):
        for i, kwargs in enumerate(job_list):
            script_name = tmp_script_folder / f"si_script_{i}.py"
            with open(script_name, "w") as f:
                kwargs_txt = ""
                for k, v in kwargs.items():
                    print(k, v)
                    kwargs_txt += "    "
                    if k == "recording":
                        # put None temporally
                        kwargs_txt += "recording=None"
                    else:
                        if isinstance(v, str):
                            kwargs_txt += f'{k}="{v}"'
                        elif isinstance(v, Path):
                            kwargs_txt += f'{k}="{str(v.absolute())}"'
                        else:
                            kwargs_txt += f"{k}={v}"
                    kwargs_txt += ",\n"

                # recording_dict = task_args[1]
                recording_dict = kwargs["recording"].to_dict()
                slurm_script = _slurm_script.format(
                    python=sys.executable, recording_dict=recording_dict, kwargs_txt=kwargs_txt
                )
                print(slurm_script)
                f.write(slurm_script)
                os.fchmod(f.fileno(), mode=stat.S_IRWXU)

            # subprocess.Popen(["sbatch", str(script_name.absolute()), f"-cpus-per-task={cpus_per_task}", f"-mem={mem}"])

    return out

_slurm_script = """#! {python}
from numpy import array
from spikeinterface.sorters import run_sorter

rec_dict = {recording_dict}

kwargs = dict(
{kwargs_txt}
)
kwargs['recording'] = load_extactor(rec_dict)

run_sorter(**kwargs)
"""




def run_sorter_by_property(
    sorter_name,
    recording,
    grouping_property,
    working_folder,
    mode_if_folder_exists=None,
    engine="loop",
    engine_kwargs={},
    verbose=False,
    docker_image=None,
    singularity_image=None,
    **sorter_params,
):
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
    mode_if_folder_exists: None
        Must be None. This is deprecated.
        If not None then a warning is raise.
        Will be removed in next release.
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
    if mode_if_folder_exists is not None:
        warnings.warn(
        "run_sorter_by_property(): mode_if_folder_exists is not used anymore",
        DeprecationWarning,
        stacklevel=2,
    )        

    working_folder = Path(working_folder).absolute()

    assert grouping_property in recording.get_property_keys(), (
        f"The 'grouping_property' {grouping_property} is not " f"a recording property!"
    )
    recording_dict = recording.split_by(grouping_property)
    
    job_list = []
    for k, rec in recording_dict.items():
        job = dict(
            sorter_name=sorter_name,
            recording=rec,
            output_folder=working_folder / str(k),
            verbose=verbose,
            docker_image=docker_image,
            singularity_image=singularity_image,
            **sorter_params
        )
        job_list.append(job)
    
    sorting_list = run_sorter_jobs(job_list, engine=engine, engine_kwargs=engine_kwargs, return_output=True)

    unit_groups = []
    for sorting, group in zip(sorting_list, recording_dict.keys()):
        num_units = sorting.get_unit_ids().size
        unit_groups.extend([group] * num_units)
    unit_groups = np.array(unit_groups)

    aggregate_sorting = aggregate_units(sorting_list)
    aggregate_sorting.set_property(key=grouping_property, values=unit_groups)
    aggregate_sorting.register_recording(recording)

    return aggregate_sorting



# This is deprecated and will be removed
def run_sorters(
    sorter_list,
    recording_dict_or_list,
    working_folder,
    sorter_params={},
    mode_if_folder_exists="raise",
    engine="loop",
    engine_kwargs={},
    verbose=False,
    with_output=True,
    docker_images={},
    singularity_images={},
):
    """
    This function is deprecated and will be removed.
    Please use run_sorter_jobs() instead.
    
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

    warnings.warn(
        "run_sorters()is deprecated please use run_sorter_jobs() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    working_folder = Path(working_folder)

    mode_if_folder_exists in ("raise", "keep", "overwrite")

    if mode_if_folder_exists == "raise" and working_folder.is_dir():
        raise Exception("working_folder already exists, please remove it")

    assert engine in _implemented_engine, f"engine must be in {_implemented_engine}"

    if isinstance(sorter_list, str):
        sorter_list = [sorter_list]

    for sorter_name in sorter_list:
        assert sorter_name in sorter_dict, f"{sorter_name} is not in sorter list"

    if isinstance(recording_dict_or_list, list):
        # in case of list
        recording_dict = {"recording_{}".format(i): rec for i, rec in enumerate(recording_dict_or_list)}
    elif isinstance(recording_dict_or_list, dict):
        recording_dict = recording_dict_or_list
    else:
        raise ValueError("bad recording dict")

    dtype_rec_name = np.dtype(type(list(recording_dict.keys())[0]))
    assert dtype_rec_name.kind in ("i", "u", "S", "U"), "Dict keys can only be integers or strings!"

    job_list = []
    for rec_name, recording in recording_dict.items():
        for sorter_name in sorter_list:
            output_folder = working_folder / str(rec_name) / sorter_name

            if output_folder.is_dir():
                # sorter folder exists
                if mode_if_folder_exists == "raise":
                    raise Exception(f"output folder already exists for {rec_name} {sorter_name}")
                elif mode_if_folder_exists == "overwrite":
                    shutil.rmtree(str(output_folder))
                elif mode_if_folder_exists == "keep":
                    
                    if is_log_ok(output_folder):
                        continue
                    else:
                        shutil.rmtree(str(output_folder))

            params = sorter_params.get(sorter_name, {})
            docker_image = docker_images.get(sorter_name, None)
            singularity_image = singularity_images.get(sorter_name, None)

            job = dict(
                sorter_name=sorter_name,
                recording=recording,
                output_folder=output_folder,
                verbose=verbose,
                docker_image=docker_image,
                singularity_image=singularity_image,
                **params
            )
            job_list.append(job)
        
    sorting_list = run_sorter_jobs(job_list, engine=engine, engine_kwargs=engine_kwargs, return_output=with_output)

    if with_output:
        keys = [(rec_name, sorter_name) for rec_name in recording_dict for sorter_name in sorter_list ]
        results = dict(zip(keys, sorting_list))
        return results

