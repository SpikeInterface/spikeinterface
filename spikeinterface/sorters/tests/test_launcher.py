import os
import shutil
import time

import pytest
from pathlib import Path

from spikeinterface.core import set_global_tmp_folder
from spikeinterface.extractors import toy_example
from spikeinterface.sorters import run_sorters, run_sorter_by_property, collect_sorting_outputs


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sorters"
else:
    cache_folder = Path("cache_folder") / "sorters"


def test_run_sorters_with_list():
    working_folder = cache_folder / 'test_run_sorters_list'
    if working_folder.is_dir():
        shutil.rmtree(working_folder)

    rec0, _ = toy_example(num_channels=4, duration=30, seed=0, num_segments=1)
    rec1, _ = toy_example(num_channels=8, duration=30, seed=0, num_segments=1)

    # make dumpable
    set_global_tmp_folder(cache_folder)
    rec0 = rec0.save(name='rec0')
    rec1 = rec1.save(name='rec1')

    recording_list = [rec0, rec1]
    sorter_list = ['tridesclous']

    run_sorters(sorter_list, recording_list, working_folder,
                engine='loop', verbose=False, with_output=False)


def test_run_sorter_by_property():
    working_folder1 = cache_folder / 'test_run_sorter_by_property1'
    if working_folder1.is_dir():
        shutil.rmtree(working_folder1)
    working_folder2 = cache_folder / 'test_run_sorter_by_property2'
    if working_folder2.is_dir():
        shutil.rmtree(working_folder2)

    rec0, _ = toy_example(num_channels=8, duration=30, seed=0, num_segments=1)
    rec0.set_channel_groups(["0"] * 4 + ["1"] * 4)
    rec0_by = rec0.split_by("group")
    group_names0 = list(rec0_by.keys())

    # make dumpable
    set_global_tmp_folder(cache_folder)
    rec0 = rec0.save(name='rec000')
    sorter_name = 'tridesclous'

    sorting0 = run_sorter_by_property(sorter_name, rec0, "group", working_folder1,
                                      engine='loop', verbose=False)
    assert "group" in sorting0.get_property_keys()
    assert all([g in group_names0 for g in sorting0.get_property("group")])
    
    rec1, _ = toy_example(num_channels=8, duration=30, seed=0, num_segments=1)
    rec1.set_channel_groups([0] * 4 + [1] * 4)
    rec1_by = rec1.split_by("group")
    group_names1 = list(rec1_by.keys())

    # make dumpable
    set_global_tmp_folder(cache_folder)
    rec1 = rec1.save(name='rec001')
    sorter_name = 'tridesclous'

    sorting1 = run_sorter_by_property(sorter_name, rec1, "group", working_folder2,
                                      engine='loop', verbose=False)
    assert "group" in sorting1.get_property_keys()
    assert all([g in group_names1 for g in sorting1.get_property("group")])


def test_run_sorters_with_dict():
    working_folder = cache_folder / 'test_run_sorters_dict'
    if working_folder.is_dir():
        shutil.rmtree(working_folder)

    rec0, _ = toy_example(num_channels=4, duration=30, seed=0, num_segments=1)
    rec1, _ = toy_example(num_channels=8, duration=30, seed=0, num_segments=1)

    # make dumpable
    set_global_tmp_folder(cache_folder)
    rec0 = rec0.save(name='rec00')
    rec1 = rec1.save(name='rec01')

    recording_dict = {'toy_tetrode': rec0, 'toy_octotrode': rec1}

    sorter_list = ['tridesclous', 'spykingcircus']

    sorter_params = {
        'tridesclous': dict(detect_threshold=5.6),
        'spykingcircus': dict(detect_threshold=5.6),
    }

    # simple loop
    t0 = time.perf_counter()
    results = run_sorters(sorter_list, recording_dict, working_folder,
                          engine='loop', sorter_params=sorter_params,
                          with_output=True,
                          mode_if_folder_exists='raise')

    t1 = time.perf_counter()
    print(t1 - t0)
    print(results)

    shutil.rmtree(working_folder / 'toy_tetrode' / 'tridesclous')
    run_sorters(sorter_list, recording_dict, working_folder,
                engine='loop', sorter_params=sorter_params,
                with_output=False,
                mode_if_folder_exists='keep')


@pytest.mark.skipif(True, reason='This is tested locally')
def test_run_sorters_joblib():
    working_folder = cache_folder / 'test_run_sorters_joblib'
    if working_folder.is_dir():
        shutil.rmtree(working_folder)

    recording_dict = {}
    for i in range(8):
        rec, _ = toy_example(num_channels=4, duration=30,
                             seed=0, num_segments=1)
        # make dumpable
        rec = rec.save(folder=cache_folder / f'rec_{i}')
        recording_dict[f'rec_{i}'] = rec

    sorter_list = ['tridesclous', ]

    # joblib
    t0 = time.perf_counter()
    run_sorters(sorter_list, recording_dict, working_folder,
                engine='joblib', engine_kwargs={'n_jobs': 4},
                with_output=False,
                mode_if_folder_exists='keep')
    t1 = time.perf_counter()
    print(t1 - t0)


@pytest.mark.skipif(True, reason='This is tested locally')
def test_run_sorters_dask():
    working_folder = cache_folder / 'test_run_sorters_dask'
    if working_folder.is_dir():
        shutil.rmtree(working_folder)

    # create recording
    recording_dict = {}
    for i in range(8):
        rec, _ = toy_example(num_channels=4, duration=30,
                             seed=0, num_segments=1)
        # make dumpable
        rec = rec.save(name=f'rec_{i}')
        recording_dict[f'rec_{i}'] = rec

    sorter_list = ['tridesclous', ]

    # create a dask Client for a slurm queue
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    python = '/home/samuel.garcia/.virtualenvs/py36/bin/python3.6'
    cluster = SLURMCluster(processes=1, cores=1, memory="12GB",
                           python=python, walltime='12:00:00', )
    cluster.scale(5)
    client = Client(cluster)

    # dask
    t0 = time.perf_counter()
    run_sorters(sorter_list, recording_dict, working_folder,
                engine='dask', engine_kwargs={'client': client},
                with_output=False,
                mode_if_folder_exists='keep')
    t1 = time.perf_counter()
    print(t1 - t0)


def test_collect_sorting_outputs():
    working_folder = cache_folder / 'test_run_sorters_dict'
    results = collect_sorting_outputs(working_folder)
    print(results)


def test_sorter_installation():
    # This import is to get error on github when import fails
    import tridesclous
    import circus


if __name__ == '__main__':
    #pass
    # test_run_sorters_with_list()

    test_run_sorter_by_property()

    # test_run_sorters_with_dict()

    # test_run_sorters_joblib()

    # test_run_sorters_dask()

    # test_collect_sorting_outputs()
