from __future__ import annotations

import shutil
import os
from pathlib import Path
import json
import pickle
import platform
from warnings import warn
from typing import Optional, Union

from spikeinterface import DEV_MODE
import spikeinterface


from .. import __version__ as si_version


from ..core import BaseRecording, NumpySorting, load_extractor
from ..core.core_tools import check_json, is_editable_mode
from .sorterlist import sorter_dict
from .utils import (
    SpikeSortingError,
    has_nvidia,
    has_docker,
    has_docker_python,
    has_singularity,
    has_spython,
    has_docker_nvidia_installed,
    get_nvidia_docker_dependecies,
)
from .container_tools import (
    find_recording_folders,
    path_to_unix,
    windows_extractor_dict_to_unix,
    ContainerClient,
    install_package_in_container,
)


REGISTRY = "spikeinterface"

SORTER_DOCKER_MAP = dict(
    combinato="combinato",
    herdingspikes="herdingspikes",
    kilosort4="kilosort4",
    klusta="klusta",
    mountainsort4="mountainsort4",
    mountainsort5="mountainsort5",
    pykilosort="pykilosort",
    spykingcircus="spyking-circus",
    spykingcircus2="spyking-circus2",
    tridesclous="tridesclous",
    yass="yass",
    # Matlab compiled sorters:
    hdsort="hdsort-compiled",
    ironclust="ironclust-compiled",
    kilosort="kilosort-compiled",
    kilosort2="kilosort2-compiled",
    kilosort2_5="kilosort2_5-compiled",
    kilosort3="kilosort3-compiled",
    waveclus="waveclus-compiled",
    waveclus_snippets="waveclus-compiled",
)

SORTER_DOCKER_MAP = {k: f"{REGISTRY}/{v}-base" for k, v in SORTER_DOCKER_MAP.items()}


_common_param_doc = """
    Parameters
    ----------
    sorter_name : str
        The sorter name
    recording : RecordingExtractor
        The recording extractor to be spike sorted
    folder : str or Path
        Path to output folder
    remove_existing_folder : bool
        If True and folder exists then delete.
    delete_output_folder : bool, default: False
        If True, output folder is deleted
    verbose : bool, default: False
        If True, output is verbose
    raise_error : bool, default: True
        If True, an error is raised if spike sorting fails
        If False, the process continues and the error is logged in the log file.
    docker_image : bool or str, default: False
        If True, pull the default docker container for the sorter and run the sorter in that container using docker.
        Use a str to specify a non-default container. If that container is not local it will be pulled from docker hub.
        If False, the sorter is run locally
    singularity_image : bool or str, default: False
        If True, pull the default docker container for the sorter and run the sorter in that container using
        singularity. Use a str to specify a non-default container. If that container is not local it will be pulled
        from Docker Hub. If False, the sorter is run locally
    with_output : bool, default: True
        If True, the output Sorting is returned as a Sorting
    delete_container_files : bool, default: True
        If True, the container temporary files are deleted after the sorting is done
    extra_requirements : list, default: None
        List of extra requirements to install in the container
    installation_mode : "auto" | "pypi" | "github" | "folder" | "dev" | "no-install", default: "auto"
        How spikeinterface is installed in the container:
          * "auto" : if host installation is a pip release then use "github" with tag
                    if host installation is DEV_MODE=True then use "dev"
          * "pypi" : use pypi with pip install spikeinterface
          * "github" : use github with `pip install git+https`
          * "folder" : mount a folder in container and install from this one.
                      So the version in the container is a different spikeinterface version from host, useful for
                      cross checks
          * "dev" : same as "folder", but the folder is the spikeinterface.__file__ to ensure same version as host
          * "no-install" : do not install spikeinterface in the container because it is already installed
    spikeinterface_version : str, default: None
        The spikeinterface version to install in the container. If None, the current version is used
    spikeinterface_folder_source : Path or None, default: None
        In case of installation_mode="folder", the spikeinterface folder source to use to install in the container
    output_folder : None, default: None
        Do not use. Deprecated output function to be removed in 0.103.
    **sorter_params : keyword args
        Spike sorter specific arguments (they can be retrieved with `get_default_sorter_params(sorter_name_or_class)`)

    Returns
    -------
    BaseSorting | None
        The spike sorted data (it `with_output` is True) or None (if `with_output` is False)
    """


def run_sorter(
    sorter_name: str,
    recording: BaseRecording,
    folder: Optional[str] = None,
    remove_existing_folder: bool = False,
    delete_output_folder: bool = False,
    verbose: bool = False,
    raise_error: bool = True,
    docker_image: Optional[Union[bool, str]] = False,
    singularity_image: Optional[Union[bool, str]] = False,
    delete_container_files: bool = True,
    with_output: bool = True,
    output_folder: None = None,
    **sorter_params,
):
    """
    Generic function to run a sorter via function approach.

    {}

    Examples
    --------
    >>> sorting = run_sorter("tridesclous", recording)
    """

    if output_folder is not None and folder is None:
        deprecation_msg = (
            "`output_folder` is deprecated and will be removed in version 0.103.0 Please use folder instead"
        )
        folder = output_folder
        warn(deprecation_msg, category=DeprecationWarning, stacklevel=2)

    common_kwargs = dict(
        sorter_name=sorter_name,
        recording=recording,
        folder=folder,
        remove_existing_folder=remove_existing_folder,
        delete_output_folder=delete_output_folder,
        verbose=verbose,
        raise_error=raise_error,
        with_output=with_output,
        **sorter_params,
    )

    if docker_image or singularity_image:
        common_kwargs.update(dict(delete_container_files=delete_container_files))
        if docker_image:
            mode = "docker"
            assert not singularity_image
            if isinstance(docker_image, bool):
                container_image = None
            else:
                container_image = docker_image

            if not has_docker():
                raise RuntimeError(
                    "Docker is not installed. Install docker on this machine to run sorting with docker."
                )

            if not has_docker_python():
                raise RuntimeError("The python `docker` package must be installed. Install with `pip install docker`")

        else:
            mode = "singularity"
            assert not docker_image
            if isinstance(singularity_image, bool):
                container_image = None
            else:
                container_image = singularity_image

            if not has_singularity():
                raise RuntimeError(
                    "Singularity is not installed. Install singularity "
                    "on this machine to run sorting with singularity."
                )

            if not has_spython():
                raise RuntimeError(
                    "The python `spython` package must be installed to "
                    "run singularity. Install with `pip install spython`"
                )

        return run_sorter_container(
            container_image=container_image,
            mode=mode,
            **common_kwargs,
        )

    return run_sorter_local(**common_kwargs)


run_sorter.__doc__ = run_sorter.__doc__.format(_common_param_doc)


def run_sorter_local(
    sorter_name,
    recording,
    folder=None,
    remove_existing_folder=True,
    delete_output_folder=False,
    verbose=False,
    raise_error=True,
    with_output=True,
    output_folder=None,
    **sorter_params,
):
    """
    Runs a sorter locally.

    Parameters
    ----------
    sorter_name : str
        The sorter name
    recording : RecordingExtractor
        The recording extractor to be spike sorted
    folder : str or Path
        Path to output folder. If None, a folder is created in the current directory
    remove_existing_folder : bool, default: True
        If True and output_folder exists yet then delete
    delete_output_folder : bool, default: False
        If True, output folder is deleted
    verbose : bool, default: False
        If True, output is verbose
    raise_error : bool, default: True
        If True, an error is raised if spike sorting fails.
        If False, the process continues and the error is logged in the log file
    with_output : bool, default: True
        If True, the output Sorting is returned as a Sorting
    output_folder : None, default: None
        Do not use. Deprecated output function to be removed in 0.103.
    **sorter_params : keyword args
    """
    if isinstance(recording, list):
        raise Exception("If you want to run several sorters/recordings use run_sorter_jobs(...)")

    if output_folder is not None and folder is None:
        deprecation_msg = (
            "`output_folder` is deprecated and will be removed in version 0.103.0 Please use folder instead"
        )
        folder = output_folder
        warn(deprecation_msg, category=DeprecationWarning, stacklevel=2)

    SorterClass = sorter_dict[sorter_name]

    # only classmethod call not instance (stateless at instance level but state is in folder)
    folder = SorterClass.initialize_folder(recording, folder, verbose, remove_existing_folder)
    SorterClass.set_params_to_folder(recording, folder, sorter_params, verbose)
    SorterClass.setup_recording(recording, folder, verbose=verbose)
    SorterClass.run_from_folder(folder, raise_error, verbose)
    if with_output:
        sorting = SorterClass.get_result_from_folder(folder, register_recording=True, sorting_info=True)
    else:
        sorting = None
    sorter_output_folder = folder / "sorter_output"
    if delete_output_folder:
        if with_output and sorting is not None:
            # if we delete the folder the sorting can have a data reference to deleted file/folder: we need a copy
            sorting_info = sorting.sorting_info
            sorting = NumpySorting.from_sorting(sorting, with_metadata=True, copy_spike_vector=True)
            sorting.set_sorting_info(
                recording_dict=sorting_info["recording"],
                params_dict=sorting_info["params"],
                log_dict=sorting_info["log"],
            )
        shutil.rmtree(sorter_output_folder)

    return sorting


def run_sorter_container(
    sorter_name: str,
    recording: BaseRecording,
    mode: str,
    container_image: Optional[str] = None,
    folder: Optional[str] = None,
    remove_existing_folder: bool = True,
    delete_output_folder: bool = False,
    verbose: bool = False,
    raise_error: bool = True,
    with_output: bool = True,
    delete_container_files: bool = True,
    extra_requirements=None,
    installation_mode="auto",
    spikeinterface_version=None,
    spikeinterface_folder_source=None,
    output_folder: None = None,
    **sorter_params,
):
    """

    Parameters
    ----------
    sorter_name : str
        The sorter name
    recording : BaseRecording
        The recording extractor to be spike sorted
    mode : str
        The container mode : "docker" or "singularity"
    container_image : str, default: None
        The container image name and tag. If None, the default container image is used
    output_folder : str, default: None
        Path to output folder
    remove_existing_folder : bool, default: True
        If True and output_folder exists yet then delete
    delete_output_folder : bool, default: False
        If True, output folder is deleted
    verbose : bool, default: False
        If True, output is verbose
    raise_error : bool, default: True
        If True, an error is raised if spike sorting fails
    with_output : bool, default: True
        If True, the output Sorting is returned as a Sorting
    delete_container_files : bool, default: True
        If True, the container temporary files are deleted after the sorting is done
    extra_requirements : list, default: None
        List of extra requirements to install in the container
    installation_mode : "auto" | "pypi" | "github" | "folder" | "dev" | "no-install", default: "auto"
        How spikeinterface is installed in the container:
          * "auto" : if host installation is a pip release then use "github" with tag
                    if host installation is DEV_MODE=True then use "dev"
          * "pypi" : use pypi with pip install spikeinterface
          * "github" : use github with `pip install git+https`
          * "folder" : mount a folder in container and install from this one.
                      So the version in the container is a different spikeinterface version from host, useful for
                      cross checks
          * "dev" : same as "folder", but the folder is the spikeinterface.__file__ to ensure same version as host
          * "no-install" : do not install spikeinterface in the container because it is already installed
    spikeinterface_version : str, default: None
        The spikeinterface version to install in the container. If None, the current version is used
    spikeinterface_folder_source : Path or None, default: None
        In case of installation_mode="folder", the spikeinterface folder source to use to install in the container
    **sorter_params : keyword args for the sorter

    """

    assert installation_mode in ("auto", "pypi", "github", "folder", "dev", "no-install")

    if output_folder is not None and folder is None:
        deprecation_msg = (
            "`output_folder` is deprecated and will be removed in version 0.103.0 Please use folder instead"
        )
        folder = output_folder
        warn(deprecation_msg, category=DeprecationWarning, stacklevel=2)
    spikeinterface_version = spikeinterface_version or si_version

    if extra_requirements is None:
        extra_requirements = []

    # common code for docker and singularity
    if folder is None:
        folder = sorter_name + "_output"

    if container_image is None:
        if sorter_name in SORTER_DOCKER_MAP:
            container_image = SORTER_DOCKER_MAP[sorter_name]
        else:
            raise ValueError(f"sorter {sorter_name} not in SORTER_DOCKER_MAP. Please specify a container_image.")

    SorterClass = sorter_dict[sorter_name]
    folder = Path(folder).absolute().resolve()
    parent_folder = folder.parent.absolute().resolve()
    parent_folder.mkdir(parents=True, exist_ok=True)

    # find input folder of recording for folder bind
    rec_dict = recording.to_dict(recursive=True)
    recording_input_folders = find_recording_folders(rec_dict)

    if platform.system() == "Windows":
        rec_dict = windows_extractor_dict_to_unix(rec_dict)

    # create 3 files for communication with container
    # recording dict inside
    if recording.check_serializability("json"):
        (parent_folder / "in_container_recording.json").write_text(
            json.dumps(check_json(rec_dict), indent=4), encoding="utf8"
        )
    elif recording.check_serializability("pickle"):
        (parent_folder / "in_container_recording.pickle").write_bytes(pickle.dumps(rec_dict))
    else:
        raise RuntimeError("To use run_sorter with a container the recording must be serializable")

    # need to share specific parameters
    (parent_folder / "in_container_params.json").write_text(
        json.dumps(check_json(sorter_params), indent=4), encoding="utf8"
    )

    in_container_sorting_folder = folder / "in_container_sorting"

    # if in Windows, skip C:
    parent_folder_unix = path_to_unix(parent_folder)
    output_folder_unix = path_to_unix(folder)
    recording_input_folders_unix = [path_to_unix(rf) for rf in recording_input_folders]
    in_container_sorting_folder_unix = path_to_unix(in_container_sorting_folder)

    # the py script
    py_script = f"""
import json
from pathlib import Path
from spikeinterface import load_extractor
from spikeinterface.sorters import run_sorter_local

if __name__ == '__main__':
    # this __name__ protection help in some case with multiprocessing (for instance HS2)
    # load recording in container
    json_rec = Path('{parent_folder_unix}/in_container_recording.json')
    pickle_rec = Path('{parent_folder_unix}/in_container_recording.pickle')
    if json_rec.exists():
        recording = load_extractor(json_rec)
    else:
        recording = load_extractor(pickle_rec)

    # load params in container
    with open('{parent_folder_unix}/in_container_params.json', encoding='utf8', mode='r') as f:
        sorter_params = json.load(f)

    # run in container
    output_folder = '{output_folder_unix}'
    sorting = run_sorter_local(
        '{sorter_name}', recording, output_folder=output_folder,
        remove_existing_folder={remove_existing_folder}, delete_output_folder=False,
        verbose={verbose}, raise_error={raise_error}, with_output=True, **sorter_params
    )
    sorting.save(folder='{in_container_sorting_folder_unix}')
"""
    (parent_folder / "in_container_sorter_script.py").write_text(py_script, encoding="utf8")

    volumes = {}
    for recording_folder, recording_folder_unix in zip(recording_input_folders, recording_input_folders_unix):
        # handle duplicates
        if str(recording_folder) not in volumes:
            volumes[str(recording_folder)] = {"bind": str(recording_folder_unix), "mode": "ro"}
    volumes[str(parent_folder)] = {"bind": str(parent_folder_unix), "mode": "rw"}

    host_folder_source = None
    if installation_mode == "auto":
        if DEV_MODE:
            if is_editable_mode():
                installation_mode = "dev"
            else:
                installation_mode = "github"
        else:
            installation_mode = "github"
        if verbose:
            print(f"installation_mode='auto' switching to installation_mode: '{installation_mode}'")

    if installation_mode == "folder":
        assert (
            spikeinterface_folder_source is not None
        ), "for installation_mode='folder', spikeinterface_folder_source must be provided"
        host_folder_source = Path(spikeinterface_folder_source)

    if installation_mode == "dev":
        host_folder_source = Path(spikeinterface.__file__).parents[2]

    if host_folder_source is not None:
        host_folder_source = host_folder_source.resolve()
        # this bind is read only  and will be copy later
        container_folder_source_ro = "/spikeinterface"
        volumes[str(host_folder_source)] = {"bind": container_folder_source_ro, "mode": "ro"}

    extra_kwargs = {}

    use_gpu = SorterClass.use_gpu(sorter_params)
    gpu_capability = SorterClass.gpu_capability
    if use_gpu:
        if gpu_capability == "nvidia-required":
            assert has_nvidia(), "The container requires a NVIDIA GPU capability, but it is not available"
            extra_kwargs["container_requires_gpu"] = True

            if platform.system() == "Linux" and not has_docker_nvidia_installed():
                warn(
                    f"nvidia-required but none of \n{get_nvidia_docker_dependecies()}\n were found. "
                    f"This may result in an error being raised during sorting. Try "
                    "installing `nvidia-container-toolkit`, including setting the "
                    "configuration steps, if running into errors."
                )

        elif gpu_capability == "nvidia-optional":
            if has_nvidia():
                extra_kwargs["container_requires_gpu"] = True
            else:
                if verbose:
                    print(
                        f"{SorterClass.sorter_name} supports GPU, but no GPU is available.\n"
                        f"Running the sorter without GPU"
                    )
        else:
            # TODO: make opencl machanism
            raise NotImplementedError("Only nvidia support is available")

    # Creating python user base folder
    py_user_base_unix = None
    if mode == "singularity":
        py_user_base_folder = parent_folder / "in_container_python_base"
        py_user_base_folder.mkdir(parents=True, exist_ok=True)
        py_user_base_unix = path_to_unix(py_user_base_folder)

    container_client = ContainerClient(mode, container_image, volumes, py_user_base_unix, extra_kwargs)
    if verbose:
        print("Starting container")
    container_client.start()

    if installation_mode == "no-install":
        need_si_install = False
    else:
        cmd_1 = ["python", "-c", "import spikeinterface; print(spikeinterface.__version__)"]
        cmd_2 = ["python", "-c", "from spikeinterface.sorters import run_sorter_local"]
        res_output = ""
        for cmd in [cmd_1, cmd_2]:
            res_output += str(container_client.run_command(cmd))
        need_si_install = "ModuleNotFoundError" in res_output

    if need_si_install:
        # update pip in container
        cmd = f"pip install --user --upgrade pip"
        res_output = container_client.run_command(cmd)

        if installation_mode == "pypi":
            install_package_in_container(
                container_client,
                "spikeinterface",
                installation_mode="pypi",
                extra="[full]",
                version=spikeinterface_version,
                verbose=verbose,
            )

        elif installation_mode == "github":
            if DEV_MODE:
                install_package_in_container(
                    container_client,
                    "spikeinterface",
                    installation_mode="github",
                    github_url="https://github.com/SpikeInterface/spikeinterface",
                    extra="[full]",
                    tag="main",
                    verbose=verbose,
                )
            else:
                install_package_in_container(
                    container_client,
                    "spikeinterface",
                    installation_mode="github",
                    github_url="https://github.com/SpikeInterface/spikeinterface",
                    extra="[full]",
                    version=spikeinterface_version,
                    verbose=verbose,
                )
        elif host_folder_source is not None:
            # this is "dev" + "folder"
            install_package_in_container(
                container_client,
                "spikeinterface",
                installation_mode="folder",
                extra="[full]",
                container_folder_source=container_folder_source_ro,
                verbose=verbose,
            )

        if installation_mode == "dev":
            # also install neo from github
            # cmd = "pip install --user --upgrade --no-input https://github.com/NeuralEnsemble/python-neo/archive/master.zip"
            # res_output = container_client.run_command(cmd)
            install_package_in_container(
                container_client,
                "neo",
                installation_mode="github",
                github_url="https://github.com/NeuralEnsemble/python-neo",
                tag="master",
            )

    if hasattr(recording, "extra_requirements"):
        extra_requirements.extend(recording.extra_requirements)

    # install additional required dependencies
    if extra_requirements:
        # if verbose:
        #     print(f"Installing extra requirements: {extra_requirements}")
        # cmd = f"pip install --user --upgrade --no-input {' '.join(extra_requirements)}"
        res_output = container_client.run_command(cmd)
        for package_name in extra_requirements:
            install_package_in_container(container_client, package_name, installation_mode="pypi", verbose=verbose)

    # run sorter on folder
    if verbose:
        print(f"Running {sorter_name} sorter inside {container_image}")

    # this do not work with singularity:
    # cmd = 'python "{}"'.format(parent_folder/'in_container_sorter_script.py')
    # this approach is better
    in_container_script_path_unix = (Path(parent_folder_unix) / "in_container_sorter_script.py").as_posix()
    cmd = ["python", f"{in_container_script_path_unix}"]
    res_output = container_client.run_command(cmd)
    run_sorter_output = res_output

    # chown folder to user uid
    if platform.system() != "Windows":
        uid = os.getuid()
        # this do not work with singularity:
        # cmd = f'chown {uid}:{uid} -R "{output_folder}"'
        # this approach is better
        cmd = ["chown", f"{uid}:{uid}", "-R", f"{folder}"]
        res_output = container_client.run_command(cmd)
    else:
        # not needed for Windows
        pass

    if verbose:
        print("Stopping container")
    container_client.stop()

    # clean useless files
    if delete_container_files:
        if (parent_folder / "in_container_recording.json").exists():
            os.remove(parent_folder / "in_container_recording.json")
        if (parent_folder / "in_container_recording.pickle").exists():
            os.remove(parent_folder / "in_container_recording.pickle")
        os.remove(parent_folder / "in_container_params.json")
        os.remove(parent_folder / "in_container_sorter_script.py")
        if mode == "singularity":
            shutil.rmtree(py_user_base_folder, ignore_errors=True)

    # check error
    folder = Path(folder)
    log_file = folder / "spikeinterface_log.json"
    if not log_file.is_file():
        run_error = True
    else:
        with log_file.open("r", encoding="utf8") as f:
            log = json.load(f)
        run_error = bool(log["error"])

    sorting = None
    if run_error:
        if raise_error:
            raise SpikeSortingError(f"Spike sorting in {mode} failed with the following error:\n{run_sorter_output}")
    else:
        if with_output:
            try:
                sorting = SorterClass.get_result_from_folder(folder)
            except Exception as e:
                try:
                    sorting = load_extractor(in_container_sorting_folder)
                except FileNotFoundError:
                    SpikeSortingError(f"Spike sorting in {mode} failed with the following error:\n{run_sorter_output}")

    sorter_output_folder = folder / "sorter_output"
    if delete_output_folder:
        shutil.rmtree(sorter_output_folder)

    return sorting


def read_sorter_folder(folder, register_recording=True, sorting_info=True, raise_error=True):
    """
    Load a sorting object from a spike sorting output folder.
    The 'folder' must contain a valid 'spikeinterface_log.json' file


    Parameters
    ----------
    folder : Pth or str
        The sorter folder
    register_recording : bool, default: True
        Attach recording (when json or pickle) to the sorting
    sorting_info : bool, default: True
        Attach sorting info to the sorting.
    """
    folder = Path(folder)
    log_file = folder / "spikeinterface_log.json"

    if not log_file.is_file():
        raise Exception(f"This folder {folder} does not have spikeinterface_log.json")

    with log_file.open("r", encoding="utf8") as f:
        log = json.load(f)

    run_error = bool(log["error"])
    if run_error:
        if raise_error:
            raise SpikeSortingError(f"Spike sorting failed for {folder}")
        else:
            return

    sorter_name = log["sorter_name"]
    SorterClass = sorter_dict[sorter_name]
    sorting = SorterClass.get_result_from_folder(
        folder, register_recording=register_recording, sorting_info=sorting_info
    )
    return sorting
