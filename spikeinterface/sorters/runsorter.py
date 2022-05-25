from copy import deepcopy
import shutil
import os
from pathlib import Path
import json
import platform


from ..version import version as si_version
from spikeinterface.core.core_tools import check_json, recursive_path_modifier, is_dict_extractor
from .sorterlist import sorter_dict
from .utils import SpikeSortingError


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
    delete_output_folder: bool
        If True, output folder is deleted (default False)
    verbose: bool
        If True, output is verbose
    raise_error: bool
        If True, an error is raised if spike sorting fails (default).
        If False, the process continues and the error is logged in the log file.
    docker_image: None or str
        If str run the sorter inside a container (docker) using the docker package.
    **sorter_params: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params(sorter_name_or_class)'

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data
    """


def run_sorter(sorter_name, recording, output_folder=None,
               remove_existing_folder=True, delete_output_folder=False,
               verbose=False, raise_error=True,
               docker_image=None, singularity_image=None,
               with_output=True, **sorter_params):
    """
    Generic function to run a sorter via function approach.

    >>> sorting = run_sorter('tridesclous', recording)
    """ + _common_param_doc

    if docker_image is not None:
        return run_sorter_container(sorter_name, recording, 'docker', docker_image,
                                    output_folder=output_folder,
                                    remove_existing_folder=remove_existing_folder,
                                    delete_output_folder=delete_output_folder, verbose=verbose,
                                    raise_error=raise_error, with_output=with_output, **sorter_params)
    if singularity_image is not None:
        return run_sorter_container(sorter_name, recording, 'singularity', singularity_image,
                                    output_folder=output_folder,
                                    remove_existing_folder=remove_existing_folder,
                                    delete_output_folder=delete_output_folder, verbose=verbose,
                                    raise_error=raise_error, with_output=with_output, **sorter_params)
    return run_sorter_local(sorter_name, recording, output_folder=output_folder,
                            remove_existing_folder=remove_existing_folder,
                            delete_output_folder=delete_output_folder,
                            verbose=verbose, raise_error=raise_error, with_output=with_output,
                            **sorter_params)


def run_sorter_local(sorter_name, recording, output_folder=None,
                     remove_existing_folder=True, delete_output_folder=False,
                     verbose=False, raise_error=True, with_output=True, **sorter_params):
    if isinstance(recording, list):
        raise Exception(
            'You you want to run several sorters/recordings use run_sorters(...)')

    SorterClass = sorter_dict[sorter_name]

    # only classmethod call not instance (stateless at instance level but state is in folder)
    output_folder = SorterClass.initialize_folder(
        recording, output_folder, verbose, remove_existing_folder)
    SorterClass.set_params_to_folder(
        recording, output_folder, sorter_params, verbose)
    SorterClass.setup_recording(recording, output_folder, verbose=verbose)
    SorterClass.run_from_folder(output_folder, raise_error, verbose)
    if with_output:
        sorting = SorterClass.get_result_from_folder(output_folder)
    else:
        sorting = None

    if delete_output_folder:
        shutil.rmtree(output_folder)

    return sorting


def find_recording_folder(d):
    if "kwargs" in d.keys():
        # handle nested
        kwargs = d["kwargs"]
        nested_extractor_dict = None
        for k, v in kwargs.items():
            if isinstance(v, dict) and is_dict_extractor(v):
                nested_extractor_dict = v
        if nested_extractor_dict is None:
            folder_to_mount = find_recording_folder(kwargs)
        else:
            folder_to_mount = find_recording_folder(nested_extractor_dict)
        return folder_to_mount
    else:
        for k, v in d.items():
            if "path" in k:
                # paths can be str or list of str
                if isinstance(v, str):
                    # one path
                    abs_path = Path(v)
                    if abs_path.is_file():
                        folder_to_mount = abs_path.parent
                    elif abs_path.is_dir():
                        folder_to_mount = abs_path
                elif isinstance(v, list):
                    # list of path
                    relative_paths = []
                    folder_to_mount = None
                    for abs_path in v:
                        abs_path = Path(abs_path)
                        if folder_to_mount is None:
                            folder_to_mount = abs_path.parent
                        else:
                            assert folder_to_mount == abs_path.parent
                else:
                    raise ValueError(
                        f'{k} key for path  must be str or list[str]')
            return folder_to_mount



def path_to_unix(path):
    path = Path(path)
    path_unix = Path(str(path)[str(path).find(":") + 1:]).as_posix()
    return path_unix


def windows_extractor_dict_to_unix(d):
    d = recursive_path_modifier(d, path_to_unix, target='path', copy=True)
    return d


class ContainerClient:
    """
    Small abstraction class to run commands in:
      * docker with "docker" python package
      * singularity with  "spython" python package
    """
    def __init__(self, mode, container_image, volumes, extra_kwargs):
        assert mode in ('docker', 'singularity')
        self.mode = mode

        if mode == 'docker':
            import docker
            client = docker.from_env()
            if extra_kwargs.get('requires_gpu', False):
                extra_kwargs.pop('requires_gpu')
                extra_kwargs["device_requests"] = [
                    docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]

            # check if the image is already present locally
            repo_tags = []
            for image in client.images.list():
                repo_tags.extend(image.attrs['RepoTags'])

            if container_image not in repo_tags:
                print(f"Docker: pulling image {container_image}")
                client.images.pull(container_image)

            self.docker_container = client.containers.create(
                    container_image, tty=True, volumes=volumes, **extra_kwargs)

        elif mode == 'singularity':
            from spython.main import Client
            # load local image file if it exists, otherwise search dockerhub
            if Path(container_image).exists():
                self.singularity_image = container_image
            else:
                print(f"Singularity: pulling image {container_image}")
                self.singularity_image = Client.pull(f'docker://{container_image}')

            if not Path(self.singularity_image).exists():
                raise FileNotFoundError(f'Unable to locate container image {container_image}')
            
            # bin options
            singularity_bind = ','.join([f'{volume_src}:{volume["bind"]}' for volume_src, volume in volumes.items()])
            options=['--bind', singularity_bind]

            # gpu options
            if extra_kwargs.get('requires_gpu', False):
                # only nvidia at the moment
                options += ['--nv']

            self.client_instance = Client.instance(self.singularity_image, start=False, options=options)

    def start(self):
        if self.mode == 'docker':
            self.docker_container.start()
        elif self.mode == 'singularity':
            self.client_instance.start()

    def stop(self):
        if self.mode == 'docker':
            self.docker_container.stop()
            self.docker_container.remove(force=True)
        elif self.mode == 'singularity':
            self.client_instance.stop()

    def run_command(self, command):
        if self.mode == 'docker':
            res = self.docker_container.exec_run(command)
            return str(res.output)
        elif self.mode == 'singularity':
            from spython.main import Client
            res = Client.execute(self.client_instance, command)
            if isinstance(res, dict):
                res = res['message']
            return res


def run_sorter_container(sorter_name, recording, mode, container_image, output_folder=None,
                         remove_existing_folder=True, delete_output_folder=False,
                         verbose=False, raise_error=True, with_output=True, **sorter_params):
    # common code for docker and singularity
    if output_folder is None:
        output_folder = sorter_name + '_output'

    SorterClass = sorter_dict[sorter_name]
    output_folder = Path(output_folder).absolute().resolve()
    parent_folder = output_folder.parent.absolute().resolve()
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)

    # find input folder of recording for folder bind
    rec_dict = recording.to_dict()
    recording_input_folder = find_recording_folder(rec_dict).absolute().resolve()

    if platform.system() == 'Windows':
        rec_dict = windows_extractor_dict_to_unix(rec_dict)

    # create 3 files for communication with container
    # recording dict inside
    (parent_folder / 'in_container_recording.json').write_text(
        json.dumps(check_json(rec_dict), indent=4), encoding='utf8')
    # need to share specific parameters
    (parent_folder / 'in_container_params.json').write_text(
        json.dumps(check_json(sorter_params), indent=4), encoding='utf8')

    # the py script
    if platform.system() == 'Windows':
        # skip C:
        parent_folder_unix = path_to_unix(parent_folder)
        output_folder_unix = path_to_unix(output_folder)
        recording_input_folder_unix = path_to_unix(recording_input_folder)
    else:
        parent_folder_unix = parent_folder
        output_folder_unix = output_folder
        recording_input_folder_unix = recording_input_folder
    py_script = f"""
import json
from spikeinterface import load_extractor
from spikeinterface.sorters import run_sorter_local

# load recording in docker
recording = load_extractor('{parent_folder_unix}/in_container_recording.json')

# load params in docker
with open('{parent_folder_unix}/in_container_params.json', encoding='utf8', mode='r') as f:
    sorter_params = json.load(f)

# run in docker
output_folder = '{output_folder_unix}'
run_sorter_local('{sorter_name}', recording, output_folder=output_folder,
            remove_existing_folder={remove_existing_folder}, delete_output_folder=False,
            verbose={verbose}, raise_error={raise_error}, **sorter_params)
"""
    (parent_folder / 'in_container_sorter_script.py').write_text(py_script, encoding='utf8')

    volumes = {}
    volumes[str(recording_input_folder)] = {
        'bind': str(recording_input_folder_unix), 'mode': 'ro'}
    volumes[str(parent_folder)] = {'bind': str(parent_folder_unix), 'mode': 'rw'}
    si_dev_path = os.getenv('SPIKEINTERFACE_DEV_PATH')

    if 'dev' in si_version and si_dev_path is not None:
        install_si_from_source = True
        # Making sure to get rid of last / or \
        si_dev_path = str(Path(si_dev_path).absolute().resolve())
        if platform.system() == 'Windows':
            si_dev_path_unix = path_to_unix(si_dev_path)
        else:
            si_dev_path_unix = si_dev_path
        volumes[si_dev_path] = {'bind': si_dev_path_unix, 'mode': 'ro'}
    else:
        install_si_from_source = False
        
    extra_kwargs = {}
    if SorterClass.docker_requires_gpu:
        extra_kwargs['requires_gpu'] = True
    
    container_client = ContainerClient(mode, container_image, volumes, extra_kwargs)
    if verbose:
        print('Starting container')
    container_client.start()

    # check if container contains spikeinterface already
    cmd_1 = ['python', '-c', 'import spikeinterface; print(spikeinterface.__version__)']
    cmd_2 = ['python', '-c', 'from spikeinterface.sorters import run_sorter_local']
    res_output = ''
    for cmd in [cmd_1, cmd_2]:
        res_output += str(container_client.run_command(cmd))
    need_si_install = 'ModuleNotFoundError' in res_output

    if need_si_install:
        if 'dev' in si_version:
            if verbose:
                print(f"Installing spikeinterface from sources in {container_image}")
            # TODO later check output
            cmd = 'pip install --upgrade --no-input MEArec'
            res_output = container_client.run_command(cmd)

            if install_si_from_source:
                si_source = 'local machine'
                res_output = container_client.run_command(f'cp -rf {si_dev_path_unix} /opt')
                cmd = f'pip install /opt/spikeinterface[full]'
            else:
                si_source = 'remote repository'
                cmd = 'pip install --upgrade --no-input git+https://github.com/SpikeInterface/spikeinterface.git#egg=spikeinterface[full]'
            if verbose:
                print(f'Installing dev spikeinterface from {si_source}')
            res_output = container_client.run_command(cmd)
            cmd = 'pip install --upgrade --no-input https://github.com/NeuralEnsemble/python-neo/archive/master.zip'
            res_output = container_client.run_command(cmd)
        else:
            if verbose:
                print(
                    f"Installing spikeinterface=={si_version} in {container_image}")
            cmd = f'pip install --upgrade --no-input spikeinterface[full]=={si_version}'
            res_output = container_client.run_command(cmd)
    else:
        # TODO version checking
        if verbose:
            print(f'spikeinterface is already installed in {container_image}')

    # run sorter on folder
    if verbose:
        print(f'Running {sorter_name} sorter inside {container_image}')

    # this do not work with singularity:
    # cmd = 'python "{}"'.format(parent_folder/'in_container_sorter_script.py')
    # this approach is better
    in_container_script_path_unix = (Path(parent_folder_unix) / 'in_container_sorter_script.py').as_posix()
    cmd = ['python', f'{in_container_script_path_unix}']
    res_output = container_client.run_command(cmd)
    run_sorter_output = res_output

    # chown folder to user uid
    if platform.system() != "Windows":
        uid = os.getuid()
        # this do not work with singularity:
        # cmd = f'chown {uid}:{uid} -R "{output_folder}"'
        # this approach is better
        cmd = ['chown', f'{uid}:{uid}', '-R', f'{output_folder}']
        res_output = container_client.run_command(cmd)
    else:
        # not needed for Windows
        pass

    if verbose:
        print('Stopping container')
    container_client.stop()

    # clean useless files
    os.remove(parent_folder / 'in_container_recording.json')
    os.remove(parent_folder / 'in_container_params.json')
    os.remove(parent_folder / 'in_container_sorter_script.py')

    # check error
    output_folder = Path(output_folder)
    log_file = output_folder / 'spikeinterface_log.json'
    if not log_file.is_file():
        run_error = True
    else:
        with log_file.open('r', encoding='utf8') as f:
            log = json.load(f)
        run_error = bool(log['error'])

    sorting = None
    if run_error:
        if raise_error:
            raise SpikeSortingError(
                f"Spike sorting in {mode} failed with the following error:\n{run_sorter_output}")
    else:
        if with_output:
            sorting = SorterClass.get_result_from_folder(output_folder)

    if delete_output_folder:
        shutil.rmtree(output_folder)

    return sorting


_common_run_doc = """
    Runs {} sorter
    """ + _common_param_doc


def read_sorter_folder(output_folder, raise_error=True):
    """
    Load a sorting object from a spike sorting output folder.
    The 'output_folder' must contain a valid 'spikeinterface_log.json' file
    """
    output_folder = Path(output_folder)
    log_file = output_folder / 'spikeinterface_log.json'

    if not log_file.is_file():
        raise Exception(
            f'This folder {output_folder} does not have spikeinterface_log.json')

    with log_file.open('r', encoding='utf8') as f:
        log = json.load(f)

    run_error = bool(log['error'])
    if run_error:
        if raise_error:
            raise SpikeSortingError(
                f"Spike sorting failed for {output_folder}")
        else:
            return

    sorter_name = log['sorter_name']
    SorterClass = sorter_dict[sorter_name]
    sorting = SorterClass.get_result_from_folder(output_folder)
    return sorting


def run_hdsort(*args, **kwargs):
    return run_sorter('hdsort', *args, **kwargs)


run_hdsort.__doc__ = _common_run_doc.format('hdsort')


def run_klusta(*args, **kwargs):
    return run_sorter('klusta', *args, **kwargs)


run_klusta.__doc__ = _common_run_doc.format('klusta')


def run_tridesclous(*args, **kwargs):
    return run_sorter('tridesclous', *args, **kwargs)


run_tridesclous.__doc__ = _common_run_doc.format('tridesclous')


def run_mountainsort4(*args, **kwargs):
    return run_sorter('mountainsort4', *args, **kwargs)


run_mountainsort4.__doc__ = _common_run_doc.format('mountainsort4')


def run_ironclust(*args, **kwargs):
    return run_sorter('ironclust', *args, **kwargs)


run_ironclust.__doc__ = _common_run_doc.format('ironclust')


def run_kilosort(*args, **kwargs):
    return run_sorter('kilosort', *args, **kwargs)


run_kilosort.__doc__ = _common_run_doc.format('kilosort')


def run_kilosort2(*args, **kwargs):
    return run_sorter('kilosort2', *args, **kwargs)


run_kilosort2.__doc__ = _common_run_doc.format('kilosort2')


def run_kilosort2_5(*args, **kwargs):
    return run_sorter('kilosort2_5', *args, **kwargs)


run_kilosort2_5.__doc__ = _common_run_doc.format('kilosort2_5')


def run_kilosort3(*args, **kwargs):
    return run_sorter('kilosort3', *args, **kwargs)


run_kilosort3.__doc__ = _common_run_doc.format('kilosort3')


def run_spykingcircus(*args, **kwargs):
    return run_sorter('spykingcircus', *args, **kwargs)


run_spykingcircus.__doc__ = _common_run_doc.format('spykingcircus')


def run_herdingspikes(*args, **kwargs):
    return run_sorter('herdingspikes', *args, **kwargs)


run_herdingspikes.__doc__ = _common_run_doc.format('herdingspikes')


def run_waveclus(*args, **kwargs):
    return run_sorter('waveclus', *args, **kwargs)


run_waveclus.__doc__ = _common_run_doc.format('waveclus')


def run_combinato(*args, **kwargs):
    return run_sorter('combinato', *args, **kwargs)


run_combinato.__doc__ = _common_run_doc.format('combinato')


def run_yass(*args, **kwargs):
    return run_sorter('yass', *args, **kwargs)


run_yass.__doc__ = _common_run_doc.format('yass')


def run_pykilosort(*args, **kwargs):
    return run_sorter('pykilosort', *args, **kwargs)


run_pykilosort.__doc__ = _common_run_doc.format('pykilosort')
