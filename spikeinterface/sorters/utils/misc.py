from subprocess import check_output
from typing import List, Union

import numpy as np

class SpikeSortingError(RuntimeError):
    """Raised whenever spike sorting fails"""


def get_git_commit(git_folder, shorten=True):
    """
    Get commit to generate sorters version.
    """
    if git_folder is None:
        return None
    try:
        commit = check_output(['git', 'rev-parse', 'HEAD'], cwd=git_folder).decode('utf8').strip()
        if shorten:
            commit = commit[:12]
    except:
        commit = None
    return commit


def has_nvidia():
    """
    Checks if the machine has nvidia capability.
    """
    try:
        check_output('nvidia-smi')
        return True
    except Exception:  # this command not being found can raise quite a few different errors depending on the configuration
        return False
