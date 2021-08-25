import shutil
import os
import sys
from pathlib import Path
import json
from copy import deepcopy

from ..version import version as si_version
from spikeinterface.core.base import is_dict_extractor
from spikeinterface.core.core_tools import check_json
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
        Tf True and output_folder exists yet then delete.
    delete_output_folder: bool
        If True, output folder is deleted (default False)
    verbose: bool
        If True, output is verbose
    raise_error: bool
        If True, an error is raised if spike sorting fails (default). If False, the process continues and the error is
        logged in the log file.
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
               verbose=False, raise_error=True, docker_image=None,
               **sorter_params):
    """
    Generic function to run a sorter via function approach.

    >>> sorting = run_sorter('tridesclous', recording)
    """ + _common_param_doc

    if docker_image is None:
        sorting = run_sorter_local(sorter_name, recording, output_folder=output_folder,
                                   remove_existing_folder=remove_existing_folder,
                                   delete_output_folder=delete_output_folder,
                                   verbose=verbose, raise_error=raise_error, **sorter_params)
    else:
        sorting = run_sorter_docker(sorter_name, recording, docker_image, output_folder=output_folder,
                                    remove_existing_folder=remove_existing_folder,
                                    delete_output_folder=delete_output_folder,
                                    verbose=verbose, raise_error=raise_error, **sorter_params)
    return sorting


def run_sorter_local(sorter_name, recording, output_folder=None,
                     remove_existing_folder=True, delete_output_folder=False,
                     verbose=False, raise_error=True, **sorter_params):
    if isinstance(recording, list):
        raise Exception('You you want to run several sorters/recordings use run_sorters(...)')

    SorterClass = sorter_dict[sorter_name]

    # only classmethod call not instance (stateless at instance level but state is in folder) 
    output_folder = SorterClass.initialize_folder(recording, output_folder, verbose, remove_existing_folder)
    SorterClass.set_params_to_folder(recording, output_folder, sorter_params, verbose)
    SorterClass.setup_recording(recording, output_folder, verbose=verbose)
    SorterClass.run_from_folder(output_folder, raise_error, verbose)
    sorting = SorterClass.get_result_from_folder(output_folder)

    if delete_output_folder:
        shutil.rmtree(output_folder)

    return sorting


def modify_input_folder(d, input_folder):
    dcopy = deepcopy(d)
    if "kwargs" in dcopy.keys():
        # handle nested
        kwargs = dcopy["kwargs"]
        nested_extractor_dict = None
        nested_extractor_key = None
        for k, v in kwargs.items():
            if isinstance(v, dict) and is_dict_extractor(v):
                nested_extractor_dict = v
                nested_extractor_key = k
        if nested_extractor_dict is None:
            dcopy_kwargs, folder_to_mount = modify_input_folder(kwargs, input_folder)
            dcopy["kwargs"] = dcopy_kwargs
        else:
            dcopy_kwargs, folder_to_mount = modify_input_folder(nested_extractor_dict, input_folder)
            dcopy["kwargs"][nested_extractor_key] = dcopy_kwargs
        return dcopy, folder_to_mount
    else:
        for k in dcopy.keys():
            if "path" in k:
                # paths can be str or list of str
                if isinstance(dcopy[k], str):
                    # one path
                    abs_path = Path(dcopy[k])
                    if abs_path.is_file():
                        folder_to_mount = abs_path.parent
                    elif abs_path.is_dir():
                        folder_to_mount = abs_path
                    relative_path = str(Path(dcopy[k]).relative_to(folder_to_mount))
                    dcopy[k] = f"{input_folder}/{relative_path}"
                elif isinstance(d[k], list):
                    # list of path
                    relative_paths = []
                    folder_to_mount = None
                    for abs_path in dcopy[k]:
                        abs_path = Path(abs_path)
                        if folder_to_mount is None:
                            folder_to_mount = abs_path.parent
                        else:
                            assert folder_to_mount == abs_path.parent
                        relative_path = str(abs_path.relative_to(folder_to_mount))
                        relative_paths.append(f"{input_folder}/{relative_path}")
                    dcopy[k] = relative_paths
                else:
                    raise ValueError(f'{k} key for path  must be str or list[str]')
            return dcopy, folder_to_mount


def run_sorter_docker(sorter_name, recording, docker_image, output_folder=None,
                      remove_existing_folder=True, delete_output_folder=False,
                      verbose=False, raise_error=True, **sorter_params):
    import docker

    if output_folder is None:
        output_folder = sorter_name + '_output'

    SorterClass = sorter_dict[sorter_name]
    output_folder = Path(output_folder).absolute()
    parent_folder = output_folder.parent
    folder_name = output_folder.stem

    # find input folder of recording for folder bind
    rec_dict = recording.to_dict()
    rec_dict, recording_input_folder = modify_input_folder(rec_dict, '/recording_input_folder')

    # create 3 files for communication with docker
    # recordonc dict inside
    (parent_folder / 'in_docker_recording.json').write_text(
        json.dumps(check_json(rec_dict), indent=4), encoding='utf8')
    # need to share specific parameters
    (parent_folder / 'in_docker_params.json').write_text(
        json.dumps(check_json(sorter_params), indent=4), encoding='utf8')
    # the py script
    py_script = f"""
import json
from spikeinterface import load_extractor
from spikeinterface.sorters import run_sorter_local

# load recorsding in docker
recording = load_extractor('/sorting_output_folder/in_docker_recording.json')

# load params in docker
with open('/sorting_output_folder/in_docker_params.json', encoding='utf8', mode='r') as f:
    sorter_params = json.load(f)

# run in docker
output_folder = '/sorting_output_folder/{folder_name}'
run_sorter_local('{sorter_name}', recording, output_folder=output_folder,
            remove_existing_folder={remove_existing_folder}, delete_output_folder=False,
            verbose={verbose}, raise_error={raise_error}, **sorter_params)
"""
    (parent_folder / 'in_docker_sorter_script.py').write_text(py_script, encoding='utf8')

    # docker bind (mount)
    volumes = {
        str(parent_folder): {'bind': '/sorting_output_folder', 'mode': 'rw'},
        str(recording_input_folder): {'bind': '/recording_input_folder', 'mode': 'ro'},
    }

    client = docker.from_env()

    # check if docker contains spikeinertace already
    cmd = 'python -c "import spikeinterface; print(spikeinterface.__version__)"'
    try:
        res = client.containers.run(docker_image, cmd)
        need_si_install = False
    except docker.errors.ContainerError:
        need_si_install = True

    # create a comman shell list
    commands = []

    if need_si_install:
        if 'dev' in si_version:
            if verbose:
                print(f"Installing spikeinterface from sources in {docker_image}")
            # install from github source several stuff
            cmd = 'pip install --upgrade --force MEArec'
            commands.append(cmd)

            cmd = 'pip install -e git+https://github.com/SpikeInterface/spikeinterface.git#egg=spikeinterface[full]'
            commands.append(cmd)

            cmd = 'pip install --upgrade --force https://github.com/NeuralEnsemble/python-neo/archive/master.zip'
            commands.append(cmd)
        else:
            # install from pypi with same version as host
            if verbose:
                print(f"Installing spikeinterface=={si_version} in {docker_image}")
            cmd = f'pip install --upgrade --force spikeinterface[full]=={si_version}'
            commands.append(cmd)

    # run sorter on folder
    cmd = 'python /sorting_output_folder/in_docker_sorter_script.py'
    commands.append(cmd)

    # put file permission to user (because docker is root...)
    uid = os.getuid()
    cmd = f'chown {uid}:{uid} -R /sorting_output_folder/{folder_name}'
    commands.append(cmd)

    # ~ commands = commands[0:4]
    # ~ commands = ' ; '.join(commands)
    commands = ' && '.join(commands)
    command = f'sh -c "{commands}"'
    #  print(command)

    extra_kwargs = {}
    if SorterClass.docker_requires_gpu:
        extra_kwargs["device_requests"] = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]

    if verbose:
        print(f"Running sorter in {docker_image}")
        # flush whatever is in the stdout
        sys.stdout.flush()
    try:
        res = client.containers.run(docker_image, command=command, volumes=volumes, stdout=verbose,
                                    **extra_kwargs)
        # clean useless files
        os.remove(parent_folder / 'in_docker_recording.json')
        os.remove(parent_folder / 'in_docker_params.json')
        os.remove(parent_folder / 'in_docker_sorter_script.py')
    except Exception as e:
        # clean useless files
        os.remove(parent_folder / 'in_docker_recording.json')
        os.remove(parent_folder / 'in_docker_params.json')
        os.remove(parent_folder / 'in_docker_sorter_script.py')
        raise SpikeSortingError(f"Spike sorting in docker failed with the following error:\n{e}")

    sorting = SorterClass.get_result_from_folder(output_folder)

    if delete_output_folder:
        shutil.rmtree(output_folder)

    return sorting


_common_run_doc = """
    Runs {} sorter
    """ + _common_param_doc


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
