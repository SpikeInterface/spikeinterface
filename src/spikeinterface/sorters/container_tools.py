from __future__ import annotations

from pathlib import Path
import platform
import os
import random
import string
import warnings

# TODO move this inside functions


from spikeinterface.core.core_tools import recursive_path_modifier, _get_paths_list


def find_recording_folders(d):
    """Finds all recording folders 'paths' in a dict"""

    path_list = _get_paths_list(d=d)
    folders_to_mount = [Path(p).resolve().parent for p in path_list]

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
    """Convert a Windows path to unix format"""
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
        mode : "docker" | "singularity"
            The container mode
        container_image : str
            container image name and tag
        volumes : dict
            dict of volumes to bind
        py_user_base : str
            Python user base folder to set as PYTHONUSERBASE env var in Singularity mode
            Prevents from overwriting user's packages when running pip install
        extra_kwargs : dict
            Extra kwargs to start container
        """
        assert mode in ("docker", "singularity")
        self.mode = mode
        self.py_user_base = py_user_base
        container_requires_gpu = extra_kwargs.get("container_requires_gpu", None)

        if mode == "docker":
            import docker

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

                docker_image = Client.load("docker://" + container_image)
                if docker_image and len(docker_image.tags) > 0:
                    tag = docker_image.tags[0]
                    warnings.warn(f"Building singularity image from local docker image: {tag}")
                    singularity_image = Client.build(f"docker-daemon://{tag}", sif_file, sudo=False)
                if not singularity_image:
                    warnings.warn(f"Singularity: pulling image {container_image}")
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
        import docker

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


def install_package_in_container(
    container_client,
    package_name,
    installation_mode="pypi",
    extra=None,
    version=None,
    tag=None,
    github_url=None,
    container_folder_source=None,
    verbose=False,
):
    """
    Install a package in a container with different modes:

    * pypi: pip install package_name
    * github: pip install {github_url}/archive/{tag/version}.tar.gz#egg=package_name
    * folder: pip install folder

    Parameters
    ----------
    container_client : ContainerClient
        The container client
    package_name : str
        The package name
    installation_mode : str
        The installation mode
    extra : str
        Extra pip install arguments, e.g. [full]
    version : str
        The package version to install
    tag : str
        The github tag to install
    github_url : str
        The github url to install (needed for github mode)
    container_folder_source : str
        The container folder source (needed for folder mode)
    verbose : bool
        If True, print output of pip install command

    Returns
    -------
    res_output : str
        The output of the pip install command
    """
    assert installation_mode in ("pypi", "github", "folder")

    if "[" in package_name:
        raise ValueError("Extra pip install should not be in package_name but like this extra='[full]'")

    if extra is not None:
        assert extra[0] == "[" and extra[-1] == "]", "extra should be like this: '[full]'"

    if verbose:
        print(f"Installing {package_name} with {installation_mode} in container")

    if installation_mode == "pypi":
        cmd = f"pip install --user --upgrade --no-input --no-build-isolation {package_name}"

        if extra is not None:
            cmd += f"{extra}"

        if version is not None:
            cmd += f"=={version}"
        res_output = container_client.run_command(cmd)

    elif installation_mode == "github":
        if version is None and tag is None:
            tag_or_version = "main"
        elif tag is not None:
            tag_or_version = tag
        elif version is not None:
            tag_or_version = version

        if github_url is None:
            github_url = "https://github.com/SpikeInterface/spikeinterface"

        cmd = f"pip install --user --upgrade --no-input {github_url}/archive/{tag_or_version}.tar.gz#egg={package_name}"
        if extra is not None:
            cmd += f"{extra}"
        res_output = container_client.run_command(cmd)

    elif installation_mode == "folder":
        assert tag is None

        if container_client.mode == "singularity":
            folder_copy = f"{container_client.py_user_base}/sources/"
        else:
            folder_copy = "/sources/"

        # create a folder for source copy
        rand_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
        folder_copy += rand_str
        cmd = f"mkdir -p {folder_copy}"
        res_output = container_client.run_command(cmd)

        cmd = f"cp -r {container_folder_source} {folder_copy}/{package_name}"
        res_output = container_client.run_command(cmd)

        cmd = f"pip install --user --no-input {folder_copy}/{package_name}"
        if extra is not None:
            cmd += f"{extra}"
        res_output = container_client.run_command(cmd)

    else:
        raise ValueError(f"install_package_incontainer, wrong installation_mode={installation_mode}")

    return res_output
