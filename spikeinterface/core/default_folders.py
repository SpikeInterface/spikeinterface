"""
'global_tmp_folder' is a variable that is generated or can be set manually.

It is useful when we do extractor.save(name='name').
"""

import tempfile
from pathlib import Path

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
    #  print('New global_tmp_folder: ', temp_folder)
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
    if not dataset_folder_set:
        dataset_folder.mkdir(exist_ok=True)
    return dataset_folder


def set_global_dataset_folder(folder):
    """
    Set the global dataset folder.
    """
    global dataset_folder
    dataset_folder = Path(folder)
    global temp_folder_set
    dataset_folder_set = True


def is_set_global_dataset_folder():
    """
    Check is the global path dataset folder have been manually set.
    """
    global dataset_folder_set
    return dataset_folder_set
