"""
'global_tmp_folder' is a variable that is generated or can be set manually.

It is useful when we do extractor.save(name='name').
"""

import tempfile
from pathlib import Path

global temp_folder
base = Path(tempfile.gettempdir()) / 'spikeinterface_cache'
base.mkdir(exist_ok=True)
temp_folder = Path(tempfile.mkdtemp(dir=base))

global temp_folder_set
temp_folder_set = False


def get_global_tmp_folder():
    """
    Get the global path temprary folder.
    """
    global temp_folder
    return temp_folder


def set_global_tmp_folder(folder):
    """
    Set the global path temprary folder.
    """
    global temp_folder
    temp_folder = Path(folder)
    global temp_folder_set
    temp_folder_set = True


def is_set_global_tmp_folder():
    """
    Check is the global path temprary folder have been manually set.
    """
    global temp_folder_set
    return temp_folder_set


def reset_global_tmp_folder():
    """
    Generate a new global path temprary folder.
    """
    global temp_folder
    temp_folder = Path(tempfile.mkdtemp(dir=base))
    print('New global_tmp_folder: ', temp_folder)
    global temp_folder_set
    temp_folder_set = False

