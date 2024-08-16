from __future__ import annotations
from pathlib import Path, WindowsPath
from typing import Union, Generator
import os
import sys
import datetime
import json
from copy import deepcopy
import importlib
from math import prod
from collections import namedtuple

import numpy as np


def define_function_from_class(source_class, name):
    "Wrapper to change the name of a class"

    return source_class


def read_python(path):
    """Parses python scripts in a dictionary

    Parameters
    ----------
    path: str or Path
        Path to file to parse

    Returns
    -------
    metadata:
        dictionary containing parsed file

    """
    from six import exec_
    import re

    path = Path(path).absolute()
    assert path.is_file()
    with path.open("r") as f:
        contents = f.read()
    contents = re.sub(r"range\(([\d,]*)\)", r"list(range(\1))", contents)
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def write_python(path, dict):
    """Saves python dictionary to file

    Parameters
    ----------
    path: str or Path
        Path to save file
    dict: dict
        dictionary to save
    """
    with Path(path).open("w") as f:
        for k, v in dict.items():
            if isinstance(v, str) and not v.startswith("'"):
                if "path" in k and "win" in sys.platform:
                    f.write(str(k) + " = r'" + str(v) + "'\n")
                else:
                    f.write(str(k) + " = '" + str(v) + "'\n")
            else:
                f.write(str(k) + " = " + str(v) + "\n")


class SIJsonEncoder(json.JSONEncoder):
    """
    An encoder used to encode Spike interface objects to json
    """

    def default(self, obj):
        from spikeinterface.core.base import BaseExtractor
        from spikeinterface.sortingcomponents.motion.motion_utils import Motion

        # Over-write behaviors for datetime object
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()

        # This should transforms integer, floats and bool to their python counterparts
        if isinstance(obj, np.generic):
            return obj.item()

        # Standard numpy dtypes like np.dtype('int32") are transformed this way
        if isinstance(obj, np.dtype):
            return np.dtype(obj).name

        # This will transform to a string canonical representation of the dtype (e.g. np.int32 -> 'int32')
        if isinstance(obj, type) and issubclass(obj, np.generic):
            return np.dtype(obj).name

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, BaseExtractor):
            return obj.to_dict()

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, Motion):
            return obj.to_dict()

        # The base-class handles the assertion
        return super().default(obj)

    # This machinery is necessary for overriding the default behavior of the json encoder with keys
    # This is a deep issue that goes deep down to cpython: https://github.com/python/cpython/issues/63020
    # This object is called before encoding (so it pre-processes the object to not have numpy scalars)
    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(self.remove_numpy_scalars(obj), _one_shot=_one_shot)

    def remove_numpy_scalars(self, object):
        from spikeinterface.core.base import BaseExtractor

        if isinstance(object, dict):
            return self.remove_numpy_scalars_in_dict(object)
        elif isinstance(object, (list, tuple, set)):
            return self.remove_numpy_scalars_in_list(object)
        elif isinstance(object, BaseExtractor):
            return self.remove_numpy_scalars_in_dict(object.to_dict())
        else:
            return object.item() if isinstance(object, np.generic) else object

    def remove_numpy_scalars_in_list(self, list_: Union[list, tuple, set]) -> list:
        return [self.remove_numpy_scalars(obj) for obj in list_]

    def remove_numpy_scalars_in_dict(self, dictionary: dict) -> dict:
        dict_copy = dict()
        for key, value in dictionary.items():
            key = self.remove_numpy_scalars(key)
            value = self.remove_numpy_scalars(value)
            dict_copy[key] = value

        return dict_copy


def check_json(dictionary: dict) -> dict:
    """
    Function that transforms a dictionary with spikeinterface objects into a json writable dictionary

    Parameters
    ----------
    dictionary : A dictionary

    """

    json_string = json.dumps(dictionary, indent=4, cls=SIJsonEncoder)
    return json.loads(json_string)


def add_suffix(file_path, possible_suffix):
    file_path = Path(file_path)
    if isinstance(possible_suffix, str):
        possible_suffix = [possible_suffix]
    possible_suffix = [s if s.startswith(".") else "." + s for s in possible_suffix]
    if file_path.suffix not in possible_suffix:
        file_path = file_path.parent / (file_path.name + possible_suffix[0])
    return file_path


def make_shared_array(shape, dtype):
    from multiprocessing.shared_memory import SharedMemory

    dtype = np.dtype(dtype)
    shape = tuple(int(x) for x in shape)  # We need to be sure that shape comes in int instead of numpy scalars
    nbytes = prod(shape) * dtype.itemsize
    shm = SharedMemory(name=None, create=True, size=nbytes)
    arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    arr[:] = 0

    return arr, shm


def is_dict_extractor(d: dict) -> bool:
    """
    Check if a dict describes an extractor.

    Returns
    -------
    is_extractor : bool
        Whether the dict describes an extractor
    """
    if not isinstance(d, dict):
        return False
    is_extractor = ("module" in d) and ("class" in d) and ("version" in d) and ("annotations" in d)
    return is_extractor


extractor_dict_element = namedtuple(typename="extractor_dict_element", field_names=["value", "name", "access_path"])


def extractor_dict_iterator(extractor_dict: dict) -> Generator[extractor_dict_element]:
    """
    Iterator for recursive traversal of a dictionary.
    This function explores the dictionary recursively and yields the path to each value along with the value itself.

    By path here we mean the keys that lead to the value in the dictionary:
    e.g. for the dictionary {'a': {'b': 1}}, the path to the value 1 is ('a', 'b').

    See `BaseExtractor.to_dict()` for a description of `extractor_dict` structure.

    Parameters
    ----------
    extractor_dict : dict
        Input dictionary

    Yields
    ------
    extractor_dict_element
        Named tuple containing the value, the name, and the access_path to the value in the dictionary.

    """

    def _extractor_dict_iterator(dict_list_or_value, access_path=(), name=""):
        if isinstance(dict_list_or_value, dict):
            for k, v in dict_list_or_value.items():
                yield from _extractor_dict_iterator(v, access_path + (k,), name=k)
        elif isinstance(dict_list_or_value, list):
            for i, v in enumerate(dict_list_or_value):
                yield from _extractor_dict_iterator(
                    v, access_path + (i,), name=name
                )  # Propagate name of list to children
        else:
            yield extractor_dict_element(
                value=dict_list_or_value,
                name=name,
                access_path=access_path,
            )

    yield from _extractor_dict_iterator(extractor_dict)


def set_value_in_extractor_dict(extractor_dict: dict, access_path: tuple, new_value):
    """
    In place modification of a value in a nested dictionary given its access path.

    Parameters
    ----------
    extractor_dict : dict
        The dictionary to modify
    access_path : tuple
        The path to the value in the dictionary
    new_value : object
        The new value to set

    Returns
    -------
    dict
        The modified dictionary
    """

    current = extractor_dict
    for key in access_path[:-1]:
        current = current[key]
    current[access_path[-1]] = new_value


def recursive_path_modifier(d, func, target="path", copy=True) -> dict:
    """
    Generic function for recursive modification of paths in an extractor dict.
    A recording can be nested and this function explores the dictionary recursively
    to find the parent file or folder paths.

    Useful for :
      * relative/absolute path change
      * docker rebase path change

    Modification is inplace with an optional copy.

    Parameters
    ----------
    d : dict
        Extractor dictionary
    func : function
        Function to apply to the path. It must take a path as input and return a path
    target : str, default: "path"
        String to match to dictionary key
    copy : bool, default: True (at first call)
        If True the original dictionary is deep copied

    Returns
    -------
    dict
        Modified dictionary
    """
    if copy:
        dc = deepcopy(d)
    else:
        dc = d

    if "kwargs" in dc.keys():
        kwargs = dc["kwargs"]

        # change in place (copy=False)
        recursive_path_modifier(kwargs, func, copy=False)

        # find nested and also change inplace (copy=False)
        nested_extractor_dict = None
        for k, v in kwargs.items():
            if isinstance(v, dict) and is_dict_extractor(v):
                nested_extractor_dict = v
                recursive_path_modifier(nested_extractor_dict, func, copy=False)
            # deal with list of extractor objects (e.g. concatenate_recordings)
            elif isinstance(v, list):
                for vl in v:
                    if isinstance(vl, dict) and is_dict_extractor(vl):
                        nested_extractor_dict = vl
                        recursive_path_modifier(nested_extractor_dict, func, copy=False)

        return dc
    else:
        for k, v in d.items():
            if target in k:
                # paths can be str or list of str or None
                if v is None:
                    continue
                if isinstance(v, (str, Path)):
                    dc[k] = func(v)
                elif isinstance(v, list):
                    dc[k] = [func(e) for e in v]
                else:
                    raise ValueError(f"{k} key for path  must be str or list[str]")


# This is the current definition that an element in a extractor_dict is a path
# This is shared across a couple of definition so it is here for DNRY
element_is_path = lambda element: isinstance(element.value, (str, Path)) and "path" in element.name


def _get_paths_list(d: dict) -> list[str | Path]:
    path_list = [e.value for e in extractor_dict_iterator(d) if element_is_path(e)]

    # if check_if_exists: TODO: Enable this once container_tools test uses proper mocks
    #     path_list = [p for p in path_list if Path(p).exists()]

    return path_list


def _relative_to(p, relative_folder):
    # custum os.path.relpath() with more checks

    relative_folder = Path(relative_folder).resolve()
    p = Path(p).resolve()
    # the as_posix transform \\ to / on window which make better json files
    rel_to = os.path.relpath(p.as_posix(), start=relative_folder.as_posix())
    return Path(rel_to).as_posix()


def check_paths_relative(input_dict, relative_folder) -> bool:
    """
    Check if relative path is possible to be applied on a dict describing an BaseExtractor.

    For instance on windows, if some data are on a drive "D:/" and the folder is on drive "C:/" it returns False.

    Parameters
    ----------
    input_dict: dict
        A dict describing an extactor obtained by BaseExtractor.to_dict()
    relative_folder: str or Path
        The folder to be relative to.

    Returns
    -------
    relative_possible: bool
        Whether the given input can be made relative to the relative_folder
    """
    path_list = _get_paths_list(input_dict)
    relative_folder = Path(relative_folder).resolve().absolute()
    not_possible = []
    for p in path_list:
        p = Path(p)
        # check path is not an URL
        if "http" in str(p):
            not_possible.append(p)
            continue

        # If windows path check have same drive
        if isinstance(p, WindowsPath) and isinstance(relative_folder, WindowsPath):
            # check that on same drive
            # important note : for UNC path on window the "//host/shared" is the drive
            if p.resolve().absolute().drive != relative_folder.drive:
                not_possible.append(p)
                continue

        # check relative is possible
        try:
            p2 = _relative_to(p, relative_folder)
        except ValueError:
            not_possible.append(p)
            continue

    return len(not_possible) == 0


def make_paths_relative(input_dict: dict, relative_folder: str | Path) -> dict:
    """
    Recursively transform a dict describing an BaseExtractor to make every path relative to a folder.

    Parameters
    ----------
    input_dict: dict
        A dict describing an extactor obtained by BaseExtractor.to_dict()
    relative_folder: str or Path
        The folder to be relative to.

    Returns
    -------
    output_dict: dict
        A copy of the input dict with modified paths.
    """

    relative_folder = Path(relative_folder).resolve().absolute()

    path_elements_in_dict = [e for e in extractor_dict_iterator(input_dict) if element_is_path(e)]
    # Only paths that exist are made relative
    path_elements_in_dict = [e for e in path_elements_in_dict if Path(e.value).exists()]

    output_dict = deepcopy(input_dict)
    for element in path_elements_in_dict:
        new_value = _relative_to(element.value, relative_folder)
        set_value_in_extractor_dict(
            extractor_dict=output_dict,
            access_path=element.access_path,
            new_value=new_value,
        )

    return output_dict


def make_paths_absolute(input_dict, base_folder):
    """
    Recursively transform a dict describing an BaseExtractor to make every path absolute given a base_folder.

    Parameters
    ----------
    input_dict: dict
        A dict describing an extactor obtained by BaseExtractor.to_dict()
    base_folder: str or Path
        The folder to be relative to.

    Returns
    -------
    output_dict: dict
        A copy of the input dict with modified paths.
    """
    base_folder = Path(base_folder)
    # use as_posix instead of str to make the path unix like even on window
    func = lambda p: (base_folder / p).resolve().absolute().as_posix()

    path_elements_in_dict = [e for e in extractor_dict_iterator(input_dict) if element_is_path(e)]
    output_dict = deepcopy(input_dict)

    output_dict = deepcopy(input_dict)
    for element in path_elements_in_dict:
        absolute_path = (base_folder / element.value).resolve()
        if Path(absolute_path).exists():
            new_value = absolute_path.as_posix()  # Not so sure about this, Sam
            set_value_in_extractor_dict(
                extractor_dict=output_dict,
                access_path=element.access_path,
                new_value=new_value,
            )

    return output_dict


def recursive_key_finder(d, key):
    # Find all values for a key on a dictionary, even if nested
    # TODO refactor to use extractor_dict_iterator

    for k, v in d.items():
        if isinstance(v, dict):
            yield from recursive_key_finder(v, key)
        else:
            if k == key:
                yield v


def convert_seconds_to_str(seconds: float, long_notation: bool = True) -> str:
    """
    Convert seconds to a human-readable string representation.
    Parameters
    ----------
    seconds : float
        The duration in seconds.
    long_notation : bool, default: True
        Whether to display the time with additional units (such as milliseconds, minutes,
        hours, or days). If set to True, the function will display a more detailed
        representation of the duration, including other units alongside the primary
        seconds representation.
    Returns
    -------
    str
        A string representing the duration, with additional units included if
        requested by the `long_notation` parameter.
    """
    base_str = f"{seconds:,.2f}s"

    if long_notation:
        if seconds < 1.0:
            base_str += f" ({seconds * 1000:.2f} ms)"
        elif seconds < 60:
            pass  # seconds is already the primary representation
        elif seconds < 3600:
            minutes = seconds / 60
            base_str += f" ({minutes:.2f} minutes)"
        elif seconds < 86400 * 2:  # 2 days
            hours = seconds / 3600
            base_str += f" ({hours:.2f} hours)"
        else:
            days = seconds / 86400
            base_str += f" ({days:.2f} days)"

    return base_str


def convert_bytes_to_str(byte_value: int) -> str:
    """
    Convert a number of bytes to a human-readable string with an appropriate unit.

    This function converts a given number of bytes into a human-readable string
    representing the value in either bytes (B), kibibytes (KiB), mebibytes (MiB),
    gibibytes (GiB), or tebibytes (TiB). The function uses the IEC binary prefixes
    (1 KiB = 1024 B, 1 MiB = 1024 KiB, etc.) to determine the appropriate unit.

    Parameters
    ----------
    byte_value : int
        The number of bytes to convert.

    Returns
    -------
    str
        The converted value as a formatted string with two decimal places,
        followed by a space and the appropriate unit (B, KiB, MiB, GiB, or TiB).

    Examples
    --------
    >>> convert_bytes_to_str(1024)
    '1.00 KiB'
    >>> convert_bytes_to_str(1048576)
    '1.00 MiB'
    >>> convert_bytes_to_str(45056)
    '43.99 KiB'
    """
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while byte_value >= 1024 and i < len(suffixes) - 1:
        byte_value /= 1024
        i += 1
    return f"{byte_value:.2f} {suffixes[i]}"


_exponents = {
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,  # Decimal (SI) prefixes
    "Ki": 1024**1,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "Pi": 1024**5,  # Binary (IEC) prefixes
}


def convert_string_to_bytes(memory_string: str) -> int:
    """
    Convert a memory size string to the corresponding number of bytes.

    Parameters:
    mem (str): Memory size string (e.g., "1G", "512Mi", "2T").

    Returns:
    int: Number of bytes.
    """
    if memory_string[-2:] in _exponents:
        suffix = memory_string[-2:]
        mem_value = memory_string[:-2]
    else:
        suffix = memory_string[-1]
        mem_value = memory_string[:-1]

    assert suffix in _exponents, f"Unknown suffix: {suffix}"
    bytes_value = int(float(mem_value) * _exponents[suffix])
    return bytes_value


def is_editable_mode() -> bool:
    """
    Check if spikeinterface is installed in editable mode
    pip install -e .
    """
    import spikeinterface

    return (Path(spikeinterface.__file__).parents[2] / "README.md").exists()


def normal_pdf(x, mu: float = 0.0, sigma: float = 1.0):
    """
    Manual implementation of the Normal distribution pdf (probability density function).
    It is about 8 to 10 times faster than scipy.stats.norm.pdf().

    Parameters
    ----------
    x: scalar or array
        The x-axis
    mu: float, default: 0.0
        The mean of the Normal distribution.
    sigma: float, default: 1.0
        The standard deviation of the Normal distribution.

    Returns
    -------
    normal_pdf: scalar or array (same type as 'x')
        The pdf of the Normal distribution for the given x-axis.
    """

    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def retrieve_importing_provenance(a_class):
    """
    Retrieve the import provenance of a class, including its import name (that consists of the class name and the module),
    the top-level module, and the module version.

    Parameters
    ----------
    a_class : type
        The class object for which to retrieve the import provenance.

    Returns
    -------
    dict
        A dictionary containing:
        - 'class':  The module path and the name of the class concatenated (e.g., 'package.subpackage.ClassName').
        - 'module': The top-level module name where the class is defined.
        - 'version': The version of the module if available, otherwise 'unknown'.
    Get class info as a dict.
    """
    module_import_path = a_class.__module__
    class_name_no_path = a_class.__name__
    class_name = f"{module_import_path}.{class_name_no_path}"  # e.g. 'spikeinterface.core.generate.AClass'
    module = class_name.split(".")[0]

    imported_module = importlib.import_module(module)
    module_version = getattr(imported_module, "__version__", "unknown")

    info = {
        "class": class_name,
        "module": module,
        "version": module_version,
    }

    return info


def measure_memory_allocation(measure_in_process: bool = True) -> float:
    """
    A local utility to measure memory allocation at a specific point in time.
    Can measure either the process resident memory or system wide memory available

    Uses psutil package.

    Parameters
    ----------
    measure_in_process : bool, True by default
        Mesure memory allocation in the current process only, if false then measures at the system
        level.
    """
    import psutil

    if measure_in_process:
        process = psutil.Process()
        memory = process.memory_info().rss
    else:
        mem_info = psutil.virtual_memory()
        memory = mem_info.total - mem_info.available

    return memory
