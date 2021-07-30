import os
import shutil
import time

import pytest

from spikeinterface.core import set_global_tmp_folder
from spikeinterface.extractors import toy_example
from spikeinterface.sorters import run_sorters, collect_sorting_outputs


def test_run_sorters_with_list():
    # This import is to get error on github whenn import fails
    import tridesclous

    cache_folder = './local_cache'
    working_folder = 'test_run_sorters_list'

    if os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)
    if os.path.exists(working_folder):
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


def test_run_sorters_with_dict():
    # This import is to get error on github whenn import fails
    import tridesclous
    import circus

    cache_folder = './local_cache'
    working_folder = 'test_run_sorters_dict'

    if os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)

    rec0, _ = toy_example(num_channels=4, duration=30, seed=0, num_segments=1)
    rec1, _ = toy_example(num_channels=8, duration=30, seed=0, num_segments=1)

    # make dumpable
    set_global_tmp_folder(cache_folder)
    rec0 = rec0.save(name='rec0')
    rec1 = rec1.save(name='rec1')

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

    shutil.rmtree(working_folder + '/toy_tetrode/tridesclous')
    run_sorters(sorter_list, recording_dict, working_folder,
                engine='loop', sorter_params=sorter_params,
                with_output=False,
                mode_if_folder_exists='keep')


@pytest.mark.skipif(True, reason='This bug with pytest/travis but not run directly')
def test_run_sorters_joblib():
    cache_folder = './local_cache'
    working_folder = 'test_run_sorters_joblib'
    if os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)

    set_global_tmp_folder(cache_folder)

    recording_dict = {}
    for i in range(8):
        rec, _ = toy_example(num_channels=4, duration=30, seed=0, num_segments=1)
        # make dumpable
        rec = rec.save(name=f'rec_{i}')
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


@pytest.mark.skipif(True, reason='This bug with pytest/travis but not run directly')
def test_run_sorters_dask():
    cache_folder = './local_cache'
    working_folder = 'test_run_sorters_dask'
    if os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)

    # create recording
    recording_dict = {}
    for i in range(8):
        rec, _ = toy_example(num_channels=4, duration=30, seed=0, num_segments=1)
        # make dumpable
        rec = rec.save(name=f'rec_{i}')
        recording_dict[f'rec_{i}'] = rec

    sorter_list = ['tridesclous', ]

    # create a dask Client for a slurm queue
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    python = '/home/samuel.garcia/.virtualenvs/py36/bin/python3.6'
    cluster = SLURMCluster(processes=1, cores=1, memory="12GB", python=python, walltime='12:00:00', )
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
    working_folder = 'test_run_sorters_dict'
    results = collect_sorting_outputs(working_folder)
    print(results)


if __name__ == '__main__':
    # test_run_sorters_with_list()

    test_run_sorters_with_dict()

    # test_run_sorters_joblib()

    # test_run_sorters_dask()

    # test_collect_sorting_outputs()
