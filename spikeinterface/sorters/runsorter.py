import shutil
import os
from pathlib import Path
import json

from spikeinterface.core import BaseRecording
from spikeinterface.core.core_tools import check_json
from .sorterlist import sorter_dict


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
    use_container: bool  default False
        Run the sorter inside a container (docker) using the hither package.
    **sorter_params: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params(sorter_name_or_class)'

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data

    """

def run_sorter(sorter_name, recording, output_folder=None,
            remove_existing_folder=True, delete_output_folder=False,
            verbose=False, raise_error=True,  docker_image=None, 
            **sorter_params):
    """
    Generic function to run a sorter via function approach.

    >>> sorting = run_sorter('tridesclous', recording)
    """ + _common_param_doc
    
    if docker_image is None:
        sorting = run_sorter_local(sorter_name, recording, output_folder=output_folder,
            remove_existing_folder=remove_existing_folder, delete_output_folder=delete_output_folder,
            verbose=verbose, raise_error=raise_error, **sorter_params)
    else:
        sorting = run_sorter_docker(sorter_name, recording, docker_image, output_folder=output_folder,
                remove_existing_folder=remove_existing_folder, delete_output_folder=delete_output_folder,
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


def modify_input_folder(rec_dict, input_folder):
    if "kwargs" in rec_dict.keys():
        dcopy_kwargs, folder_to_mount = modify_input_folder(rec_dict["kwargs"], input_folder)
        rec_dict["kwargs"] = dcopy_kwargs
        return rec_dict, folder_to_mount
    else:
        if "file_path" in rec_dict:
            file_path = Path(rec_dict["file_path"])
            folder_to_mount = file_path.parent
            file_relative = file_path.relative_to(folder_to_mount)
            rec_dict["file_path"] = f"{input_folder}/{str(file_relative)}"
            return rec_dict, folder_to_mount
        elif "folder_path" in rec_dict:
            folder_path = Path(rec_dict["folder_path"])
            folder_to_mount = folder_path.parent
            folder_relative = folder_path.relative_to(folder_to_mount)
            rec_dict["folder_path"] = f"{input_folder}/{str(folder_relative)}"
            return rec_dict, folder_to_mount
        elif "file_or_folder_path" in rec_dict:
            file_or_folder_path = Path(rec_dict["file_or_folder_path"])
            folder_to_mount = file_or_folder_path.parent
            file_or_folder_relative = file_or_folder_path.relative_to(folder_to_mount)
            rec_dict["file_or_folder_path"] = f"{input_folder}/{str(file_or_folder_relative)}"
            return rec_dict, folder_to_mount
        else:
            raise Exception


def run_sorter_docker(sorter_name, recording, docker_image, output_folder=None,
            remove_existing_folder=True, delete_output_folder=False,
            verbose=False, raise_error=True, **sorter_params):
    
    import docker
    
    rec_dict = recording.to_dict()
    
    # TODO check if None
    output_folder = Path(output_folder).absolute()
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # find input folder of recording for folder bind
    rec_dict, recording_input_folder = modify_input_folder(rec_dict,  '/recording_input_folder')
    #~ print(recording_input_folder)
    (output_folder / 'in_docker_recording.json').write_text(
            json.dumps(check_json(rec_dict), indent=4),
            encoding='utf8')
    
    # TODO sorter_params
    
    # run sorter on folder
    py_script = f"""
    
from spikeinterface import load_extractor
recording = load_extractor('/sorting_output_folder/in_docker_recording.json')

output_folder = '/sorting_output_folder'
sorter_params = dict() 
from spikeinterface.sorters import sorter_dict
SorterClass = sorter_dict['{sorter_name}']
output_folder = SorterClass.initialize_folder(recording, output_folder, {verbose}, {remove_existing_folder})
SorterClass.set_params_to_folder(recording, output_folder, sorter_params, {verbose})
SorterClass.setup_recording(recording, output_folder, verbose={verbose})
SorterClass.run_from_folder(output_folder, {raise_error}, {verbose})
"""
    cmd1 = f'python -c "{py_script}"'
    
    # put file permission to user (because docker is root...)
    uid = os.getuid()
    cmd2 = f'chown {uid}:{uid} -R /data_sorting'
    
    client = docker.from_env()

    volumes = {
        str(output_folder) : {'bind': '/sorting_output_folder', 'mode': 'rw'},
        str(recording_input_folder): {'bind': '/recording_input_folder', 'mode': 'ro'},
    }
    
    res = client.containers.run(docker_image, cmd1, volumes=volumes)
    res = client.containers.run(docker_image, cmd2, volumes=volumes)



_common_run_doc =     """
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
