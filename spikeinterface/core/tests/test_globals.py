import pytest
from pathlib import Path

from spikeinterface import (set_global_dataset_folder, get_global_dataset_folder,
                            set_global_tmp_folder, get_global_tmp_folder,
                            set_global_job_kwargs, get_global_job_kwargs, reset_global_job_kwargs)
from spikeinterface.core.job_tools import fix_job_kwargs

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_global_dataset_folder():
    dataset_folder = get_global_dataset_folder()
    assert dataset_folder.is_dir()
    new_dataset_folder = cache_folder / "dataset_folder"
    set_global_dataset_folder(new_dataset_folder)
    assert new_dataset_folder == get_global_dataset_folder()
    assert new_dataset_folder.is_dir()


def test_global_tmp_folder():
    tmp_folder = get_global_tmp_folder()
    assert tmp_folder.is_dir()
    new_tmp_folder = cache_folder / "tmp_folder"
    set_global_tmp_folder(new_tmp_folder)
    assert new_tmp_folder == get_global_tmp_folder()
    assert new_tmp_folder.is_dir()


def test_global_job_kwargs():
    job_kwargs = dict(n_jobs=4, chunk_duration="1s", progress_bar=True)
    global_job_kwargs = get_global_job_kwargs()
    assert global_job_kwargs == dict(n_jobs=1, chunk_duration="1s", progress_bar=True)
    set_global_job_kwargs(**job_kwargs)
    assert get_global_job_kwargs() == job_kwargs
    # test updating only one field
    partial_job_kwargs = dict(n_jobs=2)
    set_global_job_kwargs(**partial_job_kwargs)
    global_job_kwargs = get_global_job_kwargs()
    assert global_job_kwargs == dict(n_jobs=2, chunk_duration="1s", progress_bar=True)
    # test that fix_job_kwargs grabs global kwargs
    new_job_kwargs = dict(n_jobs=10)
    job_kwargs_split = fix_job_kwargs(new_job_kwargs)
    assert job_kwargs_split['n_jobs'] == new_job_kwargs['n_jobs']
    assert job_kwargs_split['chunk_duration'] == job_kwargs['chunk_duration']
    assert job_kwargs_split['progress_bar'] == job_kwargs['progress_bar']
    reset_global_job_kwargs()


if __name__ == '__main__':
    test_global_dataset_folder()
    test_global_tmp_folder()
    test_global_job_kwargs()
