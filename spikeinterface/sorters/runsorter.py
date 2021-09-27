import shutil
import os
import sys
from pathlib import Path
import json
from copy import deepcopy
import platform


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
               with_output=True, **sorter_params):
    """
    Generic function to run a sorter via function approach.

    >>> sorting = run_sorter('tridesclous', recording)
    """ + _common_param_doc

    if docker_image is None:
        sorting = run_sorter_local(sorter_name, recording, output_folder=output_folder,
                                   remove_existing_folder=remove_existing_folder,
                                   delete_output_folder=delete_output_folder,
                                   verbose=verbose, raise_error=raise_error, with_output=with_output, **sorter_params)
    else:
        sorting = run_sorter_docker(sorter_name, recording, docker_image, output_folder=output_folder,
                                    remove_existing_folder=remove_existing_folder,
                                    delete_output_folder=delete_output_folder,
                                    verbose=verbose, raise_error=raise_error, 
                                    with_output=with_output, **sorter_params)
    return sorting


def run_sorter_local(sorter_name, recording, output_folder=None,
                     remove_existing_folder=True, delete_output_folder=False,
                     verbose=False, raise_error=True, with_output=True, **sorter_params):
    if isinstance(recording, list):
        raise Exception('You you want to run several sorters/recordings use run_sorters(...)')

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
                    raise ValueError(f'{k} key for path  must be str or list[str]')
            return folder_to_mount




def run_sorter_docker(sorter_name, recording, docker_image, output_folder=None,
                      remove_existing_folder=True, delete_output_folder=False,
                      verbose=False, raise_error=True, with_output=True, **sorter_params):
    import docker
    
    assert platform.system() in ('Linux', 'Darwin'), 'run_sorter() with docker is supported only on linux/macos platform '

    if output_folder is None:
        output_folder = sorter_name + '_output'

    SorterClass = sorter_dict[sorter_name]
    output_folder = Path(output_folder).absolute()
    parent_folder = output_folder.parent
    folder_name = output_folder.stem

    # find input folder of recording for folder bind
    rec_dict = recording.to_dict()
    recording_input_folder = find_recording_folder(rec_dict)
    
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
recording = load_extractor('{parent_folder}/in_docker_recording.json')

# load params in docker
with open('{parent_folder}/in_docker_params.json', encoding='utf8', mode='r') as f:
    sorter_params = json.load(f)

# run in docker
output_folder = '{output_folder}'
run_sorter_local('{sorter_name}', recording, output_folder=output_folder,
            remove_existing_folder={remove_existing_folder}, delete_output_folder=False,
            verbose={verbose}, raise_error={raise_error}, **sorter_params)
"""
    (parent_folder / 'in_docker_sorter_script.py').write_text(py_script, encoding='utf8')

    volumes = {}
    volumes[str(recording_input_folder)] = {'bind': str(recording_input_folder), 'mode': 'ro'}
    volumes[str(parent_folder)] = {'bind': str(parent_folder), 'mode': 'rw'}

    extra_kwargs = {}
    if SorterClass.docker_requires_gpu:
        extra_kwargs["device_requests"] = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
    
    client = docker.from_env()
    
    container = client.containers.create(docker_image, tty=True,  volumes=volumes, **extra_kwargs)
    if verbose:
        print('Starting container')
    container.start()
    

    # check if docker contains spikeinertace already
    cmd = 'python -c "import spikeinterface; print(spikeinterface.__version__)"'
    res = container.exec_run(cmd)
    need_si_install = b'ModuleNotFoundError' in res.output

    if need_si_install:
        if 'dev' in si_version:
            if verbose:
                print(f"Installing spikeinterface from sources in {docker_image}")
            # TODO later check output
            cmd = 'pip install --upgrade --force MEArec'
            res = container.exec_run(cmd)
            cmd = 'pip install -e git+https://github.com/SpikeInterface/spikeinterface.git#egg=spikeinterface[full]'
            res = container.exec_run(cmd)
            cmd = 'pip install --upgrade --force https://github.com/NeuralEnsemble/python-neo/archive/master.zip'
            res = container.exec_run(cmd)
        else:
            if verbose:
                print(f"Installing spikeinterface=={si_version} in {docker_image}")
            cmd = f'pip install --upgrade --force spikeinterface[full]=={si_version}'
            res = container.exec_run(cmd)
    else:
        # TODO version checking
        if verbose:
            print(f'spikeinterface is already installed in {docker_image}')


    # run sorter on folder
    if verbose:
        print(f'Running {sorter_name} sorter inside {docker_image}')
    cmd = 'python "{}"'.format(parent_folder/'in_docker_sorter_script.py')
    res = container.exec_run(cmd)
    run_sorter_output = res.output

    # chown folder to user uid
    uid = os.getuid()
    cmd = f'chown {uid}:{uid} -R "{output_folder}"'
    res = container.exec_run(cmd)
    
    if verbose:
        print('Stopping container')
    container.stop()
    
    # clean useless files
    os.remove(parent_folder / 'in_docker_recording.json')
    os.remove(parent_folder / 'in_docker_params.json')
    os.remove(parent_folder / 'in_docker_sorter_script.py')    
    
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
            raise SpikeSortingError(f"Spike sorting in docker failed with the following error:\n{run_sorter_output}")
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
        raise Exception(f'This folder {output_folder} does not have spikeinterface_log.json')

    with log_file.open('r', encoding='utf8') as f:
        log = json.load(f)
    
    run_error = bool(log['error'])
    if run_error:
        if raise_error:
            raise SpikeSortingError(f"Spike sorting failed for {output_folder}")
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
