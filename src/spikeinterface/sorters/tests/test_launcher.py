import os
import sys
import shutil
import time

import pytest
from pathlib import Path

from spikeinterface.core import load_extractor

# from spikeinterface.extractors import toy_example
from spikeinterface import generate_ground_truth_recording
from spikeinterface.sorters import run_sorter_jobs, run_sorters, run_sorter_by_property


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sorters"
else:
    cache_folder = Path("cache_folder") / "sorters"

base_output = cache_folder / "sorter_output"

# no need to have many
num_recordings = 2
sorters = ["tridesclous2"]


def setup_module():
    base_seed = 42
    for i in range(num_recordings):
        rec, _ = generate_ground_truth_recording(num_channels=8, durations=[10.0], seed=base_seed + i)
        rec_folder = cache_folder / f"toy_rec_{i}"
        if rec_folder.is_dir():
            shutil.rmtree(rec_folder)

        if i % 2 == 0:
            rec.set_channel_groups(["0"] * 4 + ["1"] * 4)
        else:
            rec.set_channel_groups([0] * 4 + [1] * 4)

        rec.save(folder=rec_folder)


def get_job_list():
    jobs = []
    for i in range(num_recordings):
        for sorter_name in sorters:
            recording = load_extractor(cache_folder / f"toy_rec_{i}")
            kwargs = dict(
                sorter_name=sorter_name,
                recording=recording,
                output_folder=base_output / f"{sorter_name}_rec{i}",
                verbose=True,
                raise_error=False,
            )
            jobs.append(kwargs)

    return jobs


@pytest.fixture(scope="module")
def job_list():
    return get_job_list()


def test_run_sorter_jobs_loop(job_list):
    if base_output.is_dir():
        shutil.rmtree(base_output)
    sortings = run_sorter_jobs(job_list, engine="loop", return_output=True)
    print(sortings)


def test_run_sorter_jobs_joblib(job_list):
    if base_output.is_dir():
        shutil.rmtree(base_output)
    sortings = run_sorter_jobs(
        job_list, engine="joblib", engine_kwargs=dict(n_jobs=2, backend="loky"), return_output=True
    )
    print(sortings)


def test_run_sorter_jobs_processpoolexecutor(job_list):
    if base_output.is_dir():
        shutil.rmtree(base_output)
    sortings = run_sorter_jobs(
        job_list, engine="processpoolexecutor", engine_kwargs=dict(max_workers=2), return_output=True
    )
    print(sortings)


@pytest.mark.skipif(True, reason="This is tested locally")
def test_run_sorter_jobs_dask(job_list):
    if base_output.is_dir():
        shutil.rmtree(base_output)

    # create a dask Client for a slurm queue
    from dask.distributed import Client

    test_mode = "local"
    # test_mode = "client_slurm"

    if test_mode == "local":
        client = Client()
    elif test_mode == "client_slurm":
        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(
            processes=1,
            cores=1,
            memory="12GB",
            python=sys.executable,
            walltime="12:00:00",
        )
        cluster.scale(2)
        client = Client(cluster)

    # dask
    t0 = time.perf_counter()
    run_sorter_jobs(job_list, engine="dask", engine_kwargs=dict(client=client))
    t1 = time.perf_counter()
    print(t1 - t0)


@pytest.mark.skip("Slurm launcher need a machine with slurm")
def test_run_sorter_jobs_slurm(job_list):
    if base_output.is_dir():
        shutil.rmtree(base_output)

    working_folder = cache_folder / "test_run_sorters_slurm"
    if working_folder.is_dir():
        shutil.rmtree(working_folder)

    tmp_script_folder = working_folder / "slurm_scripts"

    run_sorter_jobs(
        job_list,
        engine="slurm",
        engine_kwargs=dict(
            tmp_script_folder=tmp_script_folder,
            cpus_per_task=32,
            mem="32G",
        ),
    )


def test_run_sorter_by_property():
    working_folder1 = cache_folder / "test_run_sorter_by_property_1"
    if working_folder1.is_dir():
        shutil.rmtree(working_folder1)

    working_folder2 = cache_folder / "test_run_sorter_by_property_2"
    if working_folder2.is_dir():
        shutil.rmtree(working_folder2)

    rec0 = load_extractor(cache_folder / "toy_rec_0")
    rec0_by = rec0.split_by("group")
    group_names0 = list(rec0_by.keys())

    sorter_name = "tridesclous2"
    sorting0 = run_sorter_by_property(sorter_name, rec0, "group", working_folder1, engine="loop", verbose=False)
    assert "group" in sorting0.get_property_keys()
    assert all([g in group_names0 for g in sorting0.get_property("group")])

    rec1 = load_extractor(cache_folder / "toy_rec_1")
    rec1_by = rec1.split_by("group")
    group_names1 = list(rec1_by.keys())

    sorter_name = "tridesclous2"
    sorting1 = run_sorter_by_property(sorter_name, rec1, "group", working_folder2, engine="loop", verbose=False)
    assert "group" in sorting1.get_property_keys()
    assert all([g in group_names1 for g in sorting1.get_property("group")])


# run_sorters is deprecated
# This will test will be removed in next release
def test_run_sorters_with_list():
    working_folder = cache_folder / "test_run_sorters_list"
    if working_folder.is_dir():
        shutil.rmtree(working_folder)

    # make serializable
    rec0 = load_extractor(cache_folder / "toy_rec_0")
    rec1 = load_extractor(cache_folder / "toy_rec_1")

    recording_list = [rec0, rec1]
    sorter_list = ["tridesclous2"]

    run_sorters(sorter_list, recording_list, working_folder, engine="loop", verbose=False, with_output=False)


# run_sorters is deprecated
# This will test will be removed in next release
def test_run_sorters_with_dict():
    working_folder = cache_folder / "test_run_sorters_dict"
    if working_folder.is_dir():
        shutil.rmtree(working_folder)

    rec0 = load_extractor(cache_folder / "toy_rec_0")
    rec1 = load_extractor(cache_folder / "toy_rec_1")

    recording_dict = {"toy_tetrode": rec0, "toy_octotrode": rec1}

    sorter_list = ["tridesclous2"]

    sorter_params = {"tridesclous2": dict()}

    # simple loop
    t0 = time.perf_counter()
    results = run_sorters(
        sorter_list,
        recording_dict,
        working_folder,
        engine="loop",
        sorter_params=sorter_params,
        with_output=True,
        mode_if_folder_exists="raise",
    )

    t1 = time.perf_counter()
    print(t1 - t0)
    print(results)

    shutil.rmtree(working_folder / "toy_tetrode" / "tridesclous2")
    run_sorters(
        sorter_list,
        recording_dict,
        working_folder / "by_dict",
        engine="loop",
        sorter_params=sorter_params,
        with_output=False,
        mode_if_folder_exists="keep",
    )


if __name__ == "__main__":
    setup_module()
    job_list = get_job_list()

    test_run_sorter_jobs_loop(job_list)
    # test_run_sorter_jobs_joblib(job_list)
    # test_run_sorter_jobs_processpoolexecutor(job_list)
    # test_run_sorter_jobs_multiprocessing(job_list)
    # test_run_sorter_jobs_dask(job_list)
    # test_run_sorter_jobs_slurm(job_list)

    # test_run_sorter_by_property()

    # this deprecated
    # test_run_sorters_with_list()
    # test_run_sorters_with_dict()
