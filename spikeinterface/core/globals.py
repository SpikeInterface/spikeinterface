"""
'global_tmp_folder' is a variable that is generated or can be set manually.

It is useful when we do extractor.save(name='name').
"""

import tempfile
from pathlib import Path
from copy import deepcopy
from .job_tools import job_keys, _shared_job_kwargs_doc

########################################

global temp_folder
global temp_folder_set
base = Path(tempfile.gettempdir()) / 'spikeinterface_cache'
temp_folder_set = False


def get_global_tmp_folder():
    """
    Get the global path temporary folder.
    """
    global temp_folder
    global temp_folder_set
    if not temp_folder_set:
        base.mkdir(exist_ok=True)
        temp_folder = Path(tempfile.mkdtemp(dir=base))
    temp_folder.mkdir(exist_ok=True, parents=True)
    return temp_folder


def set_global_tmp_folder(folder):
    """
    Set the global path temporary folder.
    """
    global temp_folder
    temp_folder = Path(folder)
    global temp_folder_set
    temp_folder_set = True


def is_set_global_tmp_folder():
    """
    Check is the global path temporary folder have been manually set.
    """
    global temp_folder_set
    return temp_folder_set


def reset_global_tmp_folder():
    """
    Generate a new global path temporary folder.
    """
    global temp_folder
    temp_folder = Path(tempfile.mkdtemp(dir=base))
    global temp_folder_set
    temp_folder_set = False


########################################

global dataset_folder
dataset_folder = Path.home() / 'spikeinterface_datasets'
global dataset_folder_set
dataset_folder_set = False


def get_global_dataset_folder():
    """
    Get the global dataset folder.
    """
    global dataset_folder
    global dataset_folder_set
    dataset_folder.mkdir(exist_ok=True, parents=True)
    return dataset_folder


def set_global_dataset_folder(folder):
    """
    Set the global dataset folder.
    """
    global dataset_folder
    dataset_folder = Path(folder)
    global dataset_folder_set
    dataset_folder_set = True


def is_set_global_dataset_folder():
    """
    Check is the global path dataset folder have been manually set.
    """
    global dataset_folder_set
    return dataset_folder_set


########################################
global global_job_kwargs
global_job_kwargs = dict(n_jobs=1, chunk_duration="1s", progress_bar=True)
global global_job_kwargs_set
global_job_kwargs_set = False


def get_global_job_kwargs():
    """
    Get the global job kwargs.
    """
    global global_job_kwargs
    return deepcopy(global_job_kwargs)


def set_global_job_kwargs(**job_kwargs):
    """
    Set the global job kwargs.
    
    Parameters
    ----------
    
    {}
    """
    global global_job_kwargs
    for k in job_kwargs:
        assert k in job_keys, (f"{k} is not a valid job keyword argument. "
                               f"Available keyword arguments are: {list(job_keys)}")
    current_global_kwargs = get_global_job_kwargs()
    current_global_kwargs.update(job_kwargs)
    global_job_kwargs = current_global_kwargs
    global global_job_kwargs_set
    global_job_kwargs_set = True


def reset_global_job_kwargs():
    """
    Reset the global job kwargs.
    """
    global global_job_kwargs
    global_job_kwargs = dict()


def is_set_global_job_kwargs_set():
    global global_job_kwargs_set
    return global_job_kwargs_set


set_global_job_kwargs.__doc__ = set_global_job_kwargs.__doc__.format(_shared_job_kwargs_doc)