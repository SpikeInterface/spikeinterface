import pytest
import warnings
from pathlib import Path
from os import cpu_count

from spikeinterface import (
    set_global_dataset_folder,
    get_global_dataset_folder,
    set_global_tmp_folder,
    get_global_tmp_folder,
    set_global_job_kwargs,
    get_global_job_kwargs,
    reset_global_job_kwargs,
)
from spikeinterface.core.job_tools import fix_job_kwargs


def test_global_dataset_folder(create_cache_folder):
    cache_folder = create_cache_folder
    dataset_folder = get_global_dataset_folder()
    assert dataset_folder.is_dir()
    new_dataset_folder = cache_folder / "dataset_folder"
    set_global_dataset_folder(new_dataset_folder)
    assert new_dataset_folder == get_global_dataset_folder()
    assert new_dataset_folder.is_dir()


def test_global_tmp_folder(create_cache_folder):
    cache_folder = create_cache_folder
    tmp_folder = get_global_tmp_folder()
    assert tmp_folder.is_dir()
    new_tmp_folder = cache_folder / "tmp_folder"
    set_global_tmp_folder(new_tmp_folder)
    assert new_tmp_folder == get_global_tmp_folder()
    assert new_tmp_folder.is_dir()


def test_global_job_kwargs():
    job_kwargs = dict(n_jobs=4, chunk_duration="1s", progress_bar=True, mp_context=None, max_threads_per_process=1)
    global_job_kwargs = get_global_job_kwargs()

    # test warning when not setting n_jobs and calling fix_job_kwargs
    with pytest.warns(UserWarning):
        job_kwargs_split = fix_job_kwargs({})

    assert global_job_kwargs == dict(
        n_jobs=1, chunk_duration="1s", progress_bar=True, mp_context=None, max_threads_per_process=1
    )
    set_global_job_kwargs(**job_kwargs)
    assert get_global_job_kwargs() == job_kwargs

    # after setting global job kwargs, fix_job_kwargs should not raise a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        job_kwargs_split = fix_job_kwargs({})

    # test updating only one field
    partial_job_kwargs = dict(n_jobs=2)
    set_global_job_kwargs(**partial_job_kwargs)
    global_job_kwargs = get_global_job_kwargs()
    assert global_job_kwargs == dict(
        n_jobs=2, chunk_duration="1s", progress_bar=True, mp_context=None, max_threads_per_process=1
    )
    # test that fix_job_kwargs grabs global kwargs
    new_job_kwargs = dict(n_jobs=cpu_count())
    job_kwargs_split = fix_job_kwargs(new_job_kwargs)
    assert job_kwargs_split["n_jobs"] == new_job_kwargs["n_jobs"]
    assert job_kwargs_split["chunk_duration"] == job_kwargs["chunk_duration"]
    assert job_kwargs_split["progress_bar"] == job_kwargs["progress_bar"]
    # test that None values do not change existing global kwargs
    none_job_kwargs = dict(n_jobs=None, progress_bar=None, chunk_duration=None)
    job_kwargs_split = fix_job_kwargs(none_job_kwargs)
    assert job_kwargs_split["chunk_duration"] == job_kwargs["chunk_duration"]
    assert job_kwargs_split["progress_bar"] == job_kwargs["progress_bar"]
    # test that n_jobs are clipped if using more than virtual cores
    excessive_n_jobs = dict(n_jobs=cpu_count() + 2)
    job_kwargs_split = fix_job_kwargs(excessive_n_jobs)
    assert job_kwargs_split["n_jobs"] == cpu_count()
    reset_global_job_kwargs()


if __name__ == "__main__":
    test_global_dataset_folder()
    test_global_tmp_folder()
    test_global_job_kwargs()
