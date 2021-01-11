"""
Utils function cor
"""
import tempfile
from pathlib import Path

global temp_folder
temp_folder = Path(tempfile.mkdtemp())

global temp_folder_set
temp_folder_set = False


def get_global_temp_folder():
    global temp_folder
    return temp_folder

def set_global_temp_folder(folder):
    global temp_folder
    temp_folder = Path(folder)
    global temp_folder_set
    temp_folder_set = True

def is_set_global_temp_folder():
    global temp_folder_set
    return temp_folder_set