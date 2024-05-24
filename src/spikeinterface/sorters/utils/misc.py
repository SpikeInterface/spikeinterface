from __future__ import annotations

from pathlib import Path
from subprocess import check_output, CalledProcessError
from typing import List, Union

import numpy as np


class SpikeSortingError(RuntimeError):
    """Raised whenever spike sorting fails"""


def get_bash_path():
    """Return path to existing bash install."""
    try:
        return check_output(["which bash"], shell=True).decode().strip("\n")
    except CalledProcessError as e:
        raise Exception("Bash is not installed or accessible on your system.")


def get_matlab_shell_name():
    """Return name of shell program used by MATLAB.

    As per MATLAB docs:
    'On UNIX, MATLAB uses a shell program to execute the given command. It
    determines which shell program to use by checking environment variables on
    your system. MATLAB first checks the MATLAB_SHELL variable, and if either
    empty or not defined, then checks SHELL. If SHELL is also empty or not
    defined, MATLAB uses /bin/sh'
    """
    try:
        # Either of "", "bash", "zsh", "fish",...
        # CalledProcessError if not defined
        matlab_shell_name = check_output(["which $MATLAB_SHELL"], shell=True).decode().strip("\n").split("/")[-1]
        return matlab_shell_name
    except CalledProcessError as e:
        pass
    try:
        # Either of "", "bash", "zsh", "fish",...
        # CalledProcessError if not defined
        df_shell_name = check_output(["which $SHELL"], shell=True).decode().strip("\n").split("/")[-1]
        return df_shell_name
    except CalledProcessError as e:
        pass
    return "sh"


def get_git_commit(git_folder, shorten=True):
    """
    Get commit to generate sorters version.
    """
    if git_folder is None:
        return None
    try:
        commit = check_output(["git", "rev-parse", "HEAD"], cwd=git_folder).decode("utf8").strip()
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
        from cuda import cuda
    except ModuleNotFoundError as err:
        raise Exception(
            "This sorter requires cuda, but the package 'cuda-python' is not installed. You can install it with:\npip install cuda-python"
        ) from err

    try:
        (cu_result_init,) = cuda.cuInit(0)
        cu_result, cu_string = cuda.cuGetErrorString(cu_result_init)
        cu_result_device_count, device_count = cuda.cuDeviceGetCount()
        return device_count > 0
    except RuntimeError:  #  Failed to dlopen libcuda.so
        return False
