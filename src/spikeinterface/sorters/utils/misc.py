from __future__ import annotations

import subprocess  # TODO: decide best format for this
from subprocess import check_output, CalledProcessError
import importlib.util


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
    cuda_spec = importlib.util.find_spec("cuda")
    if cuda_spec is not None:
        from cuda import cuda
    else:
        raise Exception(
            "This sorter requires cuda, but the package 'cuda-python' is not installed. You can install it with:\npip install cuda-python"
        )

    try:
        (cu_result_init,) = cuda.cuInit(0)
        cu_result, cu_string = cuda.cuGetErrorString(cu_result_init)
        cu_result_device_count, device_count = cuda.cuDeviceGetCount()
        return device_count > 0
    except RuntimeError:  #  Failed to dlopen libcuda.so
        return False


def _run_subprocess_silently(command):
    """
    Run a subprocess command without outputting to stderr or stdout.
    """
    output = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output


def has_docker():
    return _run_subprocess_silently("docker --version").returncode == 0


def has_singularity():
    return (
        _run_subprocess_silently("singularity --version").returncode == 0
        or _run_subprocess_silently("apptainer --version").returncode == 0
    )


def has_docker_nvidia_installed():
    """
    On Linux, nvidia has a set of container dependencies
    that are required for running GPU in docker. This is a little
    complex and is described in more detail in the links below.
    To summarise breifly, at least one of the `get_nvidia_docker_dependecies()`
    is almost certainly required to run docker with GPU.

    https://github.com/NVIDIA/nvidia-docker/issues/1268
    https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/

    Returns
    -------
    Whether at least one of the dependencies listed in
    `get_nvidia_docker_dependecies()` is installed.
    """
    all_dependencies = get_nvidia_docker_dependencies()
    has_dep = []
    for dep in all_dependencies:
        has_dep.append(_run_subprocess_silently(f"{dep} --version").returncode == 0)
    return any(has_dep)


def get_nvidia_docker_dependencies():
    """
    See `has_docker_nvidia_installed()`
    """
    return [
        "nvidia-docker",
        "nvidia-docker2",
        "nvidia-container-toolkit",
    ]


def has_docker_python():
    docker_spec = importlib.util.find_spec("docker")
    if docker_spec is not None:
        return True
    else:
        return False


def has_spython():
    spython_spec = importlib.util.find_spec("spython")
    if spython_spec is not None:
        return True
    else:
        return False
