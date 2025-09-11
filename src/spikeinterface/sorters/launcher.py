"""
Utils functions to launch several sorter on several recording in parallel or not.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import warnings
import numpy as np
from pathlib import Path
from spikeinterface.core import aggregate_units
from .runsorter import run_sorter

_default_engine_kwargs = dict(
    loop=dict(),
    joblib=dict(n_jobs=-1, backend="loky"),
    processpoolexecutor=dict(max_workers=2, mp_context=None),
    dask=dict(client=None),
    slurm={"tmp_script_folder": None, "sbatch_args": {"cpus-per-task": 1, "mem": "1G"}},
)

_implemented_engine = list(_default_engine_kwargs.keys())


def run_sorter_jobs(job_list, engine="loop", engine_kwargs=None, return_output=False):
    """
    Run several :py:func:`run_sorter()` sequentially or in parallel given a list of jobs.

    For **engine="loop"** this is equivalent to:

    .. code-block:: python

        for job in job_list:
            run_sorter(**job)

    The following engines block the I/O:
        * "loop"
        * "joblib"
        * "multiprocessing"
        * "dask"

    The following engines are *asynchronous*:
        * "slurm"

    Where *blocking* means that this function is blocking until the results are returned.
    This is in opposition to *asynchronous*, where the function returns `None` almost immediately (aka non-blocking),
    but the results must be retrieved by hand when jobs are finished. No mechanism is provided here to know
    when jobs are finished.
    In this *asynchronous* case, the :py:func:`~spikeinterface.sorters.read_sorter_folder()` helps to retrieve individual results.

    Parameters
    ----------
    job_list : list of dict
        A list a dict that are propagated to run_sorter(...)
    engine : str "loop", "joblib", "dask", "slurm"
        The engine to run the list.
    engine_kwargs : dict
        Parameters to be passed to the underlying engine.
            * loop : None
            * joblib :
                - n_jobs : int
                    The maximum number of concurrently running jobs (default=-1, tries to use all CPUs)
                - backend : str
                    Specify the parallelization backend implementation (default="loky")
            * multiprocessing :
                - max_workers : int
                    maximum number of processes (default=2)
                - mp_context : str
                    multiprocessing context (default=None)
            * dask :
                - client : dask.distributed.Client
                    Dask client to connect to (required)
            * slurm :
                - tmp_script_folder : str,Path
                    the folder in which the job scripts are created (default=None, create a random temporary directory)
                - sbatch_args: dict
                    dictionary of arguments to be passed to the sbatch command. They will be automatically prefixed with --.
                    Arguments must be in the format slurm specify, see the [documentation for `sbatch`](https://slurm.schedmd.com/sbatch.html)
                    for a list of possible arguments (default={"cpus-per-task": 1, "mem": "1G"})

    return_output : bool, default: False
        Return a sortings or None.
        This also overwrites kwargs in run_sorter(with_sorting=True/False)

    Returns
    -------
    sortings : None or list of sorting
        With engine="loop" or "joblib" you can optional get directly the list of sorting result if return_output=True.
    """

    assert engine in _implemented_engine, f"engine must be in {_implemented_engine}"

    if engine_kwargs is None:
        engine_kwargs = dict()
    engine_kwargs_ = dict()
    engine_kwargs_.update(_default_engine_kwargs[engine])
    engine_kwargs_.update(engine_kwargs)
    engine_kwargs = engine_kwargs_

    if return_output:
        assert engine in (
            "loop",
            "joblib",
            "processpoolexecutor",
        ), "Only 'loop', 'joblib', and 'processpoolexecutor' support return_output=True."
        out = []
        for kwargs in job_list:
            kwargs["with_output"] = True
    else:
        out = None
        for kwargs in job_list:
            kwargs["with_output"] = False

    if engine == "loop":
        # simple loop in main process
        for kwargs in job_list:
            sorting = run_sorter(**kwargs)
            if return_output:
                out.append(sorting)

    elif engine == "joblib":
        from joblib import Parallel, delayed

        n_jobs = engine_kwargs["n_jobs"]
        backend = engine_kwargs["backend"]
        sortings = Parallel(n_jobs=n_jobs, backend=backend)(delayed(run_sorter)(**kwargs) for kwargs in job_list)
        if return_output:
            out.extend(sortings)

    elif engine == "processpoolexecutor":
        from concurrent.futures import ProcessPoolExecutor

        max_workers = engine_kwargs["max_workers"]
        mp_context = engine_kwargs["mp_context"]

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            futures = []
            for kwargs in job_list:
                res = executor.submit(run_sorter, **kwargs)
                futures.append(res)
            for futur in futures:
                sorting = futur.result()
                if return_output:
                    out.append(sorting)

    elif engine == "dask":
        client = engine_kwargs["client"]
        assert client is not None, "For dask engine you have to provide : client = dask.distributed.Client(...)"

        tasks = []
        for kwargs in job_list:
            task = client.submit(run_sorter, **kwargs)
            tasks.append(task)

        for task in tasks:
            task.result()

    elif engine == "slurm":
        if "cpus_per_task" in engine_kwargs:
            raise ValueError(
                "keyword argument cpus_per_task is no longer supported for slurm engine, "
                "please use cpus-per-task instead."
            )
        # generate python script for slurm
        tmp_script_folder = engine_kwargs["tmp_script_folder"]
        if tmp_script_folder is None:
            tmp_script_folder = tempfile.mkdtemp(prefix="spikeinterface_slurm_")
        tmp_script_folder = Path(tmp_script_folder)
        tmp_script_folder.mkdir(exist_ok=True, parents=True)

        for i, kwargs in enumerate(job_list):
            script_name = tmp_script_folder / f"si_script_{i}.py"
            with open(script_name, "w") as f:
                kwargs_txt = ""
                for k, v in kwargs.items():
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
                f.write(slurm_script)
                os.fchmod(f.fileno(), mode=stat.S_IRWXU)

            progr = ["sbatch"]
            for k, v in engine_kwargs["sbatch_args"].items():
                progr.append(f"--{k}")
                progr.append(f"{v}")
            progr.append(str(script_name.absolute()))
            print(f"subprocess called with command {' '.join(progr)}")
            p = subprocess.run(progr, capture_output=True, text=True)
            print(p.stdout)
            if len(p.stderr) > 0:
                warnings.warn(p.stderr)

    return out


_slurm_script = """#! {python}
from numpy import array
from spikeinterface import load
from spikeinterface.sorters import run_sorter

rec_dict = {recording_dict}

kwargs = dict(
{kwargs_txt}
)
kwargs['recording'] = load(rec_dict)

run_sorter(**kwargs)
"""


def run_sorter_by_property(
    sorter_name,
    recording,
    grouping_property,
    folder,
    engine="loop",
    engine_kwargs=None,
    verbose=False,
    docker_image=None,
    singularity_image=None,
    **sorter_params,
):
    """
    Generic function to run a sorter on a recording after splitting by a "grouping_property" (e.g. "group").

    Internally, the function works as follows:
        * the recording is split based on the provided "grouping_property" (using the "split_by" function)
        * the "run_sorters" function is run on the split recordings
        * sorting outputs are aggregated using the "aggregate_units" function
        * the "grouping_property" is added as a property to the SortingExtractor

    Parameters
    ----------
    sorter_name : str
        The sorter name
    recording : BaseRecording
        The recording to be sorted
    grouping_property : object
        Property to split by before sorting
    folder : str | Path
        The working directory.
    engine : "loop" | "joblib" | "dask" | "slurm", default: "loop"
        Which engine to use to run sorter.
    engine_kwargs : dict
        This contains kwargs specific to the launcher engine.
        See the documentation for :py:func:`~spikeinterface.sorters.launcher.run_sorter_jobs()` for more details.
    verbose : bool, default: False
        Controls sorter verboseness
    docker_image : None or str, default: None
        If str run the sorter inside a container (docker) using the docker package
    singularity_image : None or str, default: None
        If str run the sorter inside a container (singularity) using the docker package
    **sorter_params : keyword args
        Spike sorter specific arguments (they can be retrieved with `get_default_sorter_params(sorter_name_or_class)`)

    Returns
    -------
    sorting : UnitsAggregationSorting
        The aggregated SortingExtractor.

    Examples
    --------
    This example shows how to run spike sorting split by group using the "joblib" backend with 4 jobs for parallel
    processing.

    >>> sorting = si.run_sorter_by_property("tridesclous", recording, grouping_property="group",
                                            folder="sort_by_group", engine="joblib",
                                            engine_kwargs={"n_jobs": 4})

    """

    working_folder = Path(folder).absolute()

    assert grouping_property in recording.get_property_keys(), (
        f"The 'grouping_property' {grouping_property} is not " f"a recording property!"
    )
    recording_dict = recording.split_by(grouping_property)

    job_list = []
    for k, rec in recording_dict.items():
        job = dict(
            sorter_name=sorter_name,
            recording=rec,
            folder=working_folder / str(k),
            verbose=verbose,
            docker_image=docker_image,
            singularity_image=singularity_image,
            **sorter_params,
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
