import shutil
import os
from pathlib import Path
import json
import platform
from warnings import warn
from typing import Optional, Union

from ..core import BaseRecording
from .. import __version__ as si_version
from spikeinterface.core.npzsortingextractor import NpzSortingExtractor
from spikeinterface.core.core_tools import check_json, recursive_path_modifier
from .sorterlist import sorter_dict
from .utils import SpikeSortingError, has_nvidia

try:
    HAS_DOCKER = True
    import docker
except ModuleNotFoundError:
    HAS_DOCKER = False

REGISTRY = "spikeinterface"

SORTER_DOCKER_MAP = dict(
    combinato="combinato",
    herdingspikes="herdingspikes",
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
    sorter_name: str
        The sorter name
    recording: RecordingExtractor
        The recording extractor to be spike sorted
    output_folder: str or Path
        Path to output folder
    remove_existing_folder: bool
        If True and output_folder exists yet then delete.
    delete_output_folder: bool, default: False
        If True, output folder is deleted
    verbose: bool, default: False
        If True, output is verbose
    raise_error: bool, default: True
        If True, an error is raised if spike sorting fails
        If False, the process continues and the error is logged in the log file.
    docker_image: bool or str, default: False
        If True, pull the default docker container for the sorter and run the sorter in that container using docker.
        Use a str to specify a non-default container. If that container is not local it will be pulled from docker hub.
        If False, the sorter is run locally
    singularity_image: bool or str, default: False
        If True, pull the default docker container for the sorter and run the sorter in that container using
        singularity. Use a str to specify a non-default container. If that container is not local it will be pulled
        from Docker Hub. If False, the sorter is run locally
    delete_container_files: bool, default: True
        If True, the container temporary files are deleted after the sorting is done
    with_output: bool, default: True
        If True, the output Sorting is returned as a Sorting
    **sorter_params: keyword args
        Spike sorter specific arguments (they can be retrieved with `get_default_sorter_params(sorter_name_or_class)`)

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """


def run_sorter(
    sorter_name: str,
    recording: BaseRecording,
    output_folder: Optional[str] = None,
    remove_existing_folder: bool = False,
    delete_output_folder: bool = False,
    verbose: bool = False,
    raise_error: bool = True,
    docker_image: Optional[Union[bool, str]] = False,
    singularity_image: Optional[Union[bool, str]] = False,
    delete_container_files: bool = True,
    with_output: bool = True,
    **sorter_params,
):
    """
    Generic function to run a sorter via function approach.

    {}

    Examples
    --------
    >>> sorting = run_sorter("tridesclous", recording)
    """

    common_kwargs = dict(
        sorter_name=sorter_name,
        recording=recording,
        output_folder=output_folder,
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
        else:
            mode = "singularity"
            assert not docker_image
            if isinstance(singularity_image, bool):
                container_image = None
            else:
                container_image = singularity_image
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
    output_folder=None,
    remove_existing_folder=True,
    delete_output_folder=False,
    verbose=False,
    raise_error=True,
    with_output=True,
    **sorter_params,
):
    if isinstance(recording, list):
        raise Exception("You you want to run several sorters/recordings use run_sorters(...)")

    SorterClass = sorter_dict[sorter_name]

    # only classmethod call not instance (stateless at instance level but state is in folder)
    output_folder = SorterClass.initialize_folder(recording, output_folder, verbose, remove_existing_folder)
    SorterClass.set_params_to_folder(recording, output_folder, sorter_params, verbose)
    SorterClass.setup_recording(recording, output_folder, verbose=verbose)
    SorterClass.run_from_folder(output_folder, raise_error, verbose)
    if with_output:
        sorting = SorterClass.get_result_from_folder(output_folder)
    else:
        sorting = None
    sorter_output_folder = output_folder / "sorter_output"
    if delete_output_folder:
        shutil.rmtree(sorter_output_folder)

    return sorting


def find_recording_folders(d):
    folders_to_mount = []

    def append_parent_folder(p):
        p = Path(p)
        folders_to_mount.append(p.resolve().absolute().parent)
        return p

    _ = recursive_path_modifier(d, append_parent_folder, target="path", copy=True)

    try:  # this will fail if on different drives (Windows)
        base_folders_to_mount = [Path(os.path.commonpath(folders_to_mount))]
    except ValueError:
        base_folders_to_mount = folders_to_mount

    # let's not mount root if dries are /home/..., /mnt1/...
    if len(base_folders_to_mount) == 1:
        if len(str(base_folders_to_mount[0])) == 1:
            base_folders_to_mount = folders_to_mount

    return base_folders_to_mount


def path_to_unix(path):
    path = Path(path)
    if platform.system() == "Windows":
        path = Path(str(path)[str(path).find(":") + 1 :])
    return path.as_posix()


def windows_extractor_dict_to_unix(d):
    d = recursive_path_modifier(d, path_to_unix, target="path", copy=True)
    return d


class ContainerClient:
    """
    Small abstraction class to run commands in:
      * docker with "docker" python package
      * singularity with  "spython" python package
    """

    def __init__(self, mode, container_image, volumes, py_user_base, extra_kwargs):
        """
        Parameters
        ----------
        mode: "docker" | "singularity"
            The container mode
        container_image: str
            container image name and tag
        volumes: dict
            dict of volumes to bind
        py_user_base: str
            Python user base folder to set as PYTHONUSERBASE env var in Singularity mode
            Prevents from overwriting user's packages when running pip install
        extra_kwargs: dict
            Extra kwargs to start container
        """
        assert mode in ("docker", "singularity")
        self.mode = mode
        self.py_user_base = py_user_base
        container_requires_gpu = extra_kwargs.get("container_requires_gpu", None)

        if mode == "docker":
            if not HAS_DOCKER:
                raise ModuleNotFoundError("No module named 'docker'")
            client = docker.from_env()
            if container_requires_gpu is not None:
                extra_kwargs.pop("container_requires_gpu")
                extra_kwargs["device_requests"] = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

            if self._get_docker_image(container_image) is None:
                print(f"Docker: pulling image {container_image}")
                client.images.pull(container_image)

            self.docker_container = client.containers.create(container_image, tty=True, volumes=volumes, **extra_kwargs)

        elif mode == "singularity":
            assert self.py_user_base, "py_user_base folder must be set in singularity mode"
            from spython.main import Client

            # load local image file if it exists, otherwise search dockerhub
            sif_file = Client._get_filename(container_image)
            singularity_image = None
            if Path(container_image).exists():
                singularity_image = container_image
            elif Path(sif_file).exists():
                singularity_image = sif_file
            else:
                if HAS_DOCKER:
                    docker_image = self._get_docker_image(container_image)
                    if docker_image and len(docker_image.tags) > 0:
                        tag = docker_image.tags[0]
                        print(f"Building singularity image from local docker image: {tag}")
                        singularity_image = Client.build(f"docker-daemon://{tag}", sif_file, sudo=False)
                if not singularity_image:
                    print(f"Singularity: pulling image {container_image}")
                    singularity_image = Client.pull(f"docker://{container_image}")

            if not Path(singularity_image).exists():
                raise FileNotFoundError(f"Unable to locate container image {container_image}")

            # bin options
            singularity_bind = ",".join([f'{volume_src}:{volume["bind"]}' for volume_src, volume in volumes.items()])
            options = ["--bind", singularity_bind]

            # gpu options
            if container_requires_gpu:
                # only nvidia at the moment
                options += ["--nv"]

            self.client_instance = Client.instance(singularity_image, start=False, options=options)

    @staticmethod
    def _get_docker_image(container_image):
        docker_client = docker.from_env(timeout=300)
        try:
            docker_image = docker_client.images.get(container_image)
        except docker.errors.ImageNotFound:
            docker_image = None
        return docker_image

    def start(self):
        if self.mode == "docker":
            self.docker_container.start()
        elif self.mode == "singularity":
            self.client_instance.start()

    def stop(self):
        if self.mode == "docker":
            self.docker_container.stop()
            self.docker_container.remove(force=True)
        elif self.mode == "singularity":
            self.client_instance.stop()

    def run_command(self, command):
        if self.mode == "docker":
            res = self.docker_container.exec_run(command)
            return res.output.decode(encoding="utf-8", errors="ignore")
        elif self.mode == "singularity":
            from spython.main import Client

            options = ["--cleanenv", "--env", f"PYTHONUSERBASE={self.py_user_base}"]
            res = Client.execute(self.client_instance, command, options=options)
            if isinstance(res, dict):
                res = res["message"]
            return res


def run_sorter_container(
    sorter_name: str,
    recording: BaseRecording,
    mode: str,
    container_image: Optional[str] = None,
    output_folder: Optional[str] = None,
    remove_existing_folder: bool = True,
    delete_output_folder: bool = False,
    verbose: bool = False,
    raise_error: bool = True,
    with_output: bool = True,
    delete_container_files: bool = True,
    extra_requirements=None,
    **sorter_params,
):
    """

    Parameters
    ----------
    sorter_name: str
        The sorter name
    recording: BaseRecording
        The recording extractor to be spike sorted
    mode: str
        The container mode: "docker" or "singularity"
    container_image: str, default: None
        The container image name and tag. If None, the default container image is used
    output_folder: str, default: None
        Path to output folder
    remove_existing_folder: bool, default: True
        If True and output_folder exists yet then delete
    delete_output_folder: bool, default: False
        If True, output folder is deleted
    verbose: bool, default: False
        If True, output is verbose
    raise_error: bool, default: True
        If True, an error is raised if spike sorting fails
    with_output: bool, default: True
        If True, the output Sorting is returned as a Sorting
    delete_container_files: bool, default: True
        If True, the container temporary files are deleted after the sorting is done
    extra_requirements: list, default: None
        List of extra requirements to install in the container
    **sorter_params: keyword args for the sorter

    """

    if extra_requirements is None:
        extra_requirements = []

    # common code for docker and singularity
    if output_folder is None:
        output_folder = sorter_name + "_output"

    if container_image is None:
        if sorter_name in SORTER_DOCKER_MAP:
            container_image = SORTER_DOCKER_MAP[sorter_name]
        else:
            raise ValueError(f"sorter {sorter_name} not in SORTER_DOCKER_MAP. Please specify a container_image.")

    SorterClass = sorter_dict[sorter_name]
    output_folder = Path(output_folder).absolute().resolve()
    parent_folder = output_folder.parent.absolute().resolve()
    parent_folder.mkdir(parents=True, exist_ok=True)

    # find input folder of recording for folder bind
    rec_dict = recording.to_dict(recursive=True)
    recording_input_folders = find_recording_folders(rec_dict)

    if platform.system() == "Windows":
        rec_dict = windows_extractor_dict_to_unix(rec_dict)

    # create 3 files for communication with container
    # recording dict inside
    (parent_folder / "in_container_recording.json").write_text(
        json.dumps(check_json(rec_dict), indent=4), encoding="utf8"
    )
    # need to share specific parameters
    (parent_folder / "in_container_params.json").write_text(
        json.dumps(check_json(sorter_params), indent=4), encoding="utf8"
    )

    npz_sorting_path = output_folder / "in_container_sorting"

    # if in Windows, skip C:
    parent_folder_unix = path_to_unix(parent_folder)
    output_folder_unix = path_to_unix(output_folder)
    recording_input_folders_unix = [path_to_unix(rf) for rf in recording_input_folders]
    npz_sorting_path_unix = path_to_unix(npz_sorting_path)

    # the py script
    py_script = f"""
import json
from spikeinterface import load_extractor
from spikeinterface.sorters import run_sorter_local

if __name__ == '__main__':
    # this __name__ protection help in some case with multiprocessing (for instance HS2)
    # load recording in container
    recording = load_extractor('{parent_folder_unix}/in_container_recording.json')

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
    sorting.save_to_folder(folder='{npz_sorting_path_unix}')
"""
    (parent_folder / "in_container_sorter_script.py").write_text(py_script, encoding="utf8")

    volumes = {}
    for recording_folder, recording_folder_unix in zip(recording_input_folders, recording_input_folders_unix):
        # handle duplicates
        if str(recording_folder) not in volumes:
            volumes[str(recording_folder)] = {"bind": str(recording_folder_unix), "mode": "ro"}
    volumes[str(parent_folder)] = {"bind": str(parent_folder_unix), "mode": "rw"}

    si_dev_path = os.getenv("SPIKEINTERFACE_DEV_PATH", None)

    install_si_from_source = False
    if "dev" in si_version and si_dev_path is not None:
        install_si_from_source = True
        # Making sure to get rid of last / or \
        si_dev_path = str(Path(si_dev_path).absolute().resolve())
        si_dev_path_unix = path_to_unix(si_dev_path)
        volumes[si_dev_path] = {"bind": si_dev_path_unix, "mode": "ro"}

    extra_kwargs = {}

    use_gpu = SorterClass.use_gpu(sorter_params)
    gpu_capability = SorterClass.gpu_capability
    if use_gpu:
        if gpu_capability == "nvidia-required":
            assert has_nvidia(), "The container requires a NVIDIA GPU capability, but it is not available"
            extra_kwargs["container_requires_gpu"] = True
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
        si_source_folder = f"{py_user_base_unix}/sources"
    else:
        si_source_folder = "/sources"
    container_client = ContainerClient(mode, container_image, volumes, py_user_base_unix, extra_kwargs)
    if verbose:
        print("Starting container")
    container_client.start()

    if verbose and install_si_from_source:
        print("******")
        print("Container started with the following paths")
        print(si_dev_path_unix, si_source_folder)

    # check if container contains spikeinterface already
    cmd_1 = ["python", "-c", "import spikeinterface; print(spikeinterface.__version__)"]
    cmd_2 = ["python", "-c", "from spikeinterface.sorters import run_sorter_local"]
    res_output = ""
    for cmd in [cmd_1, cmd_2]:
        res_output += str(container_client.run_command(cmd))
    need_si_install = "ModuleNotFoundError" in res_output

    if need_si_install:
        if "dev" in si_version:
            if verbose:
                print(f"Installing spikeinterface from sources in {container_image}")

            # TODO later check output
            if install_si_from_source:
                si_source = "local machine"
                # install in local copy of host SI folder in sources/spikeinterface to avoid permission errors
                cmd = f"mkdir {si_source_folder}"
                res_output = container_client.run_command(cmd)
                cmd = f"cp -r {si_dev_path_unix} {si_source_folder}"
                res_output = container_client.run_command(cmd)
                cmd = f"pip install --user {si_source_folder}/spikeinterface[full]"
            else:
                si_source = "remote repository"
                cmd = "pip install --user --upgrade --no-input git+https://github.com/SpikeInterface/spikeinterface.git#egg=spikeinterface[full]"
            if verbose:
                print(f"Installing dev spikeinterface from {si_source}")
            res_output = container_client.run_command(cmd)
            cmd = "pip install --user --upgrade --no-input https://github.com/NeuralEnsemble/python-neo/archive/master.zip"
            res_output = container_client.run_command(cmd)
        else:
            if verbose:
                print(f"Installing spikeinterface=={si_version} in {container_image}")
            cmd = f"pip install --user --upgrade --no-input spikeinterface[full]=={si_version}"
            res_output = container_client.run_command(cmd)
    else:
        # TODO version checking
        if verbose:
            print(f"spikeinterface is already installed in {container_image}")

    if hasattr(recording, "extra_requirements"):
        extra_requirements.extend(recording.extra_requirements)

    # install additional required dependencies
    if extra_requirements:
        if verbose:
            print(f"Installing extra requirements: {extra_requirements}")
        cmd = f"pip install --user --upgrade --no-input {' '.join(extra_requirements)}"
        res_output = container_client.run_command(cmd)

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
        cmd = ["chown", f"{uid}:{uid}", "-R", f"{output_folder}"]
        res_output = container_client.run_command(cmd)
    else:
        # not needed for Windows
        pass

    if verbose:
        print("Stopping container")
    container_client.stop()

    # clean useless files
    if delete_container_files:
        os.remove(parent_folder / "in_container_recording.json")
        os.remove(parent_folder / "in_container_params.json")
        os.remove(parent_folder / "in_container_sorter_script.py")
        if mode == "singularity":
            shutil.rmtree(py_user_base_folder)

    # check error
    output_folder = Path(output_folder)
    log_file = output_folder / "spikeinterface_log.json"
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
                sorting = SorterClass.get_result_from_folder(output_folder)
            except Exception as e:
                if verbose:
                    print(
                        "Failed to get result with sorter specific extractor.\n"
                        f"Error Message: {e}\n"
                        "Getting result from in-container saved NpzSortingExtractor"
                    )
                try:
                    sorting = NpzSortingExtractor.load_from_folder(npz_sorting_path)
                except FileNotFoundError:
                    SpikeSortingError(f"Spike sorting in {mode} failed with the following error:\n{run_sorter_output}")

    sorter_output_folder = output_folder / "sorter_output"
    if delete_output_folder:
        shutil.rmtree(sorter_output_folder)

    return sorting


_common_run_doc = (
    """
    Runs {} sorter
    """
    + _common_param_doc
)


def read_sorter_folder(output_folder, register_recording=True, sorting_info=True, raise_error=True):
    """
    Load a sorting object from a spike sorting output folder.
    The 'output_folder' must contain a valid 'spikeinterface_log.json' file


    Parameters
    ----------
    output_folder: Pth or str
        The sorter folder
    register_recording: bool, default: True
        Attach recording (when json or pickle) to the sorting
    sorting_info: bool, default: True
        Attach sorting info to the sorting.
    """
    output_folder = Path(output_folder)
    log_file = output_folder / "spikeinterface_log.json"

    if not log_file.is_file():
        raise Exception(f"This folder {output_folder} does not have spikeinterface_log.json")

    with log_file.open("r", encoding="utf8") as f:
        log = json.load(f)

    run_error = bool(log["error"])
    if run_error:
        if raise_error:
            raise SpikeSortingError(f"Spike sorting failed for {output_folder}")
        else:
            return

    sorter_name = log["sorter_name"]
    SorterClass = sorter_dict[sorter_name]
    sorting = SorterClass.get_result_from_folder(
        output_folder, register_recording=register_recording, sorting_info=sorting_info
    )
    return sorting


def run_hdsort(*args, **kwargs):
    warn(
        "run_hdsort is deprecated. Use run_sorter(sorter_name='hdsort') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("hdsort", *args, **kwargs)


run_hdsort.__doc__ = _common_run_doc.format("hdsort")


def run_klusta(*args, **kwargs):
    warn(
        "run_klusta is deprecated. Use run_sorter(sorter_name='klusta') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("klusta", *args, **kwargs)


run_klusta.__doc__ = _common_run_doc.format("klusta")


def run_tridesclous(*args, **kwargs):
    warn(
        "run_tridesclous is deprecated. Use run_sorter(sorter_name='tridesclous') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("tridesclous", *args, **kwargs)


run_tridesclous.__doc__ = _common_run_doc.format("tridesclous")


def run_mountainsort4(*args, **kwargs):
    warn(
        "run_mountainsort4 is deprecated. Use run_sorter(sorter_name='mountainsort4') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("mountainsort4", *args, **kwargs)


run_mountainsort4.__doc__ = _common_run_doc.format("mountainsort4")


def run_mountainsort5(*args, **kwargs):
    warn(
        "run_mountainsort5 is deprecated. Use run_sorter(sorter_name='mountainsort5') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("mountainsort5", *args, **kwargs)


run_mountainsort5.__doc__ = _common_run_doc.format("mountainsort5")


def run_ironclust(*args, **kwargs):
    warn(
        "run_ironclust is deprecated. Use run_sorter(sorter_name='ironclust') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("ironclust", *args, **kwargs)


run_ironclust.__doc__ = _common_run_doc.format("ironclust")


def run_kilosort(*args, **kwargs):
    warn(
        "run_kilosort is deprecated. Use run_sorter(sorter_name='kilosort') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("kilosort", *args, **kwargs)


run_kilosort.__doc__ = _common_run_doc.format("kilosort")


def run_kilosort2(*args, **kwargs):
    warn(
        "run_kilosort2 is deprecated. Use run_sorter(sorter_name='kilosort2') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("kilosort2", *args, **kwargs)


run_kilosort2.__doc__ = _common_run_doc.format("kilosort2")


def run_kilosort2_5(*args, **kwargs):
    warn(
        "run_kilosort2_5 is deprecated. Use run_sorter(sorter_name='kilosort2_5') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("kilosort2_5", *args, **kwargs)


run_kilosort2_5.__doc__ = _common_run_doc.format("kilosort2_5")


def run_kilosort3(*args, **kwargs):
    warn(
        "run_kilosort3 is deprecated. Use run_sorter(sorter_name='kilosort3') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("kilosort3", *args, **kwargs)


run_kilosort3.__doc__ = _common_run_doc.format("kilosort3")


def run_spykingcircus(*args, **kwargs):
    warn(
        "run_spykingcircus is deprecated. Use run_sorter(sorter_name='spykingcircus') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("spykingcircus", *args, **kwargs)


run_spykingcircus.__doc__ = _common_run_doc.format("spykingcircus")


def run_herdingspikes(*args, **kwargs):
    warn(
        "run_herdingspikes is deprecated. Use run_sorter(sorter_name='herdingspikes') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("herdingspikes", *args, **kwargs)


run_herdingspikes.__doc__ = _common_run_doc.format("herdingspikes")


def run_waveclus(*args, **kwargs):
    warn(
        "run_waveclus is deprecated. Use run_sorter(sorter_name='waveclus') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("waveclus", *args, **kwargs)


run_waveclus.__doc__ = _common_run_doc.format("waveclus")


def run_waveclus_snippets(*args, **kwargs):
    warn(
        "run_waveclus_snippets is deprecated. Use run_sorter(sorter_name='waveclus_snippets') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("waveclus_snippets", *args, **kwargs)


def run_combinato(*args, **kwargs):
    warn(
        "run_combinato is deprecated. Use run_sorter(sorter_name='combinato') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("combinato", *args, **kwargs)


run_combinato.__doc__ = _common_run_doc.format("combinato")


def run_yass(*args, **kwargs):
    warn(
        "run_yass is deprecated. Use run_sorter(sorter_name='yass') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("yass", *args, **kwargs)


run_yass.__doc__ = _common_run_doc.format("yass")


def run_pykilosort(*args, **kwargs):
    warn(
        "run_pykilosort is deprecated. Use run_sorter(sorter_name='pykilosort') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_sorter("pykilosort", *args, **kwargs)


run_pykilosort.__doc__ = _common_run_doc.format("pykilosort")
