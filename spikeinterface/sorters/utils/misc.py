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


def resolve_sif_file(container_image):
    """
    Resolves container_image string for singularity image file

    Converts image names from docker notation to .sif files
    """
    if '.sif' in container_image:
        return container_image

    # Resolve repo name
    if '/' in container_image:
        container_image = container_image.split('/')[-1]

    # Resolve tag
    if ':' in container_image:
        container_image = container_image.replace(':', '_')
    else:
        container_image += '_latest'

    container_image += '.sif'
    return container_image
