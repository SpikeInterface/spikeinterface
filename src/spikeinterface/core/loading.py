import warnings
from pathlib import Path

import json

from .base import BaseExtractor
from .core_tools import is_path_remote


_error_msg = (
    "{file_path} is not a file or a folder. It should point to either a json, pickle file or a "
    "folder that is the result of extractor.save(...) or sortinganalyzer.save_as(...)"
)


def load(
    file_or_folder_or_dict,
    **kwargs,
    # load_extensions=True, backend_options=None
) -> "BaseExtractor | SortingAnalyzer | Motion | Template":
    """
    General load function to load a SpikeInterface object.

    The function can load:
        - a `Recording` or `Sorting` object from:
            * dictionary
            * json file
            * pkl file
            * binary folder (after `extractor.save(..., format='binary_folder')`)
            * zarr folder (after `extractor.save(..., format='zarr')`)
            * remote zarr folder
        - a `SortingAnalyzer` object from:
            * binary folder
            * zarr folder
            * remote zarr folder
            * WaveformExtractor folder (backward compatibility for v<0.101)
        - a `Motion` object from:
           * folder (after `Motion.save(folder)`)
        - a `Templates` object from:
           * zarr folder (after `Templates.add_templates_to_zarr_group()`)
           * dictionary (after `Templates.to_dict()`)

    Parameters
    ----------
    file_or_folder_or_dict : dictionary or folder or file (json, pickle)
        The file path, folder path, or dictionary to load the Recording, Sorting, or SortingAnalyzer from
    kwargs : keyword arguments for various objects, including
        * base_folder: str | Path | bool
            The base folder to make relative paths absolute. Only used to load Recording/Sorting objects.
            If True and file_or_folder_or_dict is a file, the parent folder of the file is used.
        * load_extensions: bool, default: True
            If True, the SortingAnalyzer extensions are loaded. Only used to load SortingAnalyzer objects.
        * storage_options: dict | None, default: None
            The storage options to use when loading the object. Only used to load Recording/Sorting objects.
        * backend_options: dict | None, default: None
            The backend options to use when loading the object. Only used to load SortingAnalyzer objects.
            The dictionary can contain the following keys:
            - storage_options: dict | None (fsspec storage options)
            - saving_options: dict | None (additional saving options for creating and saving datasets)

    Returns
    -------
    spikeinterface object: Recording or Sorting or SortingAnalyzer or Motion or Templates
        The loaded spikeinterface object
    """
    base_folder = kwargs.get("base_folder", None)
    if isinstance(file_or_folder_or_dict, dict):
        # dict can be only Recording Sorting Motion Templates
        d = file_or_folder_or_dict
        object_type = _guess_object_from_dict(d)
        if object_type is None:
            raise ValueError(_error_msg.format(file_path="provided dictionary"))
        return _load_object_from_dict(d, object_type, base_folder=base_folder)

    is_local = not is_path_remote(file_or_folder_or_dict)
    if is_local:
        file_path = Path(file_or_folder_or_dict)

    if is_local and file_path.is_file():
        # Standard case based on a file (json or pickle) after a Base.dump(json/pickle)
        if base_folder is True:
            base_folder = file_path.parent

        if str(file_path).endswith(".json"):
            import json

            with open(file_path, "r") as f:
                d = json.load(f)
        elif str(file_path).endswith(".pkl") or str(file_path).endswith(".pickle"):
            import pickle

            with open(file_path, "rb") as f:
                d = pickle.load(f)
        else:
            raise ValueError(_error_msg.format(file_path=file_path))

        object_type = _guess_object_from_dict(d)
        if object_type is None:
            raise ValueError(_error_msg.format(file_path=file_path))
        return _load_object_from_dict(d, object_type, base_folder=base_folder)

    elif is_local and file_path.is_dir():

        folder = file_path
        if folder.suffix == ".zarr":
            # Local zarr can be
            # Sorting / Recording / SortingAnalyzer / Motion
            object_type = _guess_object_from_zarr(folder)
            if object_type is None:
                raise ValueError(_error_msg.format(file_path=file_path))
            loaded_object = _load_object_from_zarr(folder, object_type, **kwargs)
            return loaded_object

        else:
            # Local folder can be
            # Sorting / Recording / SortingAnalyzer / Motion
            object_type = _guess_object_from_local_folder(folder)
            if object_type is None:
                raise ValueError(_error_msg.format(file_path=file_path))
            loaded_object = _load_object_from_folder(folder, object_type, **kwargs)
            return loaded_object
    else:
        # remote zarr can be
        # Sorting Recording SortingAnalyzer Template
        url = file_or_folder_or_dict
        object_type = _guess_object_from_zarr(url)
        loaded_object = _load_object_from_zarr(url, object_type, **kwargs)
        return loaded_object


def load_extractor(file_or_folder_or_dict, base_folder=None) -> "BaseExtractor":
    warnings.warn(
        "load_extractor() is deprecated and will be removed in the future. Please use load() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load(file_or_folder_or_dict, base_folder=base_folder)


def _guess_object_from_dict(d):
    """
    When an object is read from json or pickle or zarr attr we can guess which object it is
    """

    # SortingAnalyzer and Motion
    if d.get("object") is not None:
        # the case is explicit.
        # SortingAnalyzer and Motion used to implement this from the start of implementing
        return d["object"]

    # Template
    is_template = True
    for k in ("templates_array", "sampling_frequency", "nbefore"):
        if k not in d:
            is_template = False
            break
    if is_template:
        return "Templates"

    # Recording | Sorting (A BaseExtractor)
    is_sorting_or_recording = True
    for k in ("kwargs", "class", "version", "annotations"):
        if k not in d:
            is_sorting_or_recording = False
            break
    if is_sorting_or_recording:
        return "Recording|Sorting"

    # Unknow
    return None


def _load_object_from_dict(d, object_type, base_folder=None):
    if object_type in ("Recording", "Sorting", "Recording|Sorting"):
        return BaseExtractor.from_dict(d, base_folder=base_folder)

    elif object_type == "Templates":
        from spikeinterface.core import Templates

        return Templates.from_dict(d)

    elif object_type == "Motion":
        from spikeinterface.core.motion import Motion

        return Motion.from_dict(d)


def _guess_object_from_local_folder(folder):
    folder = Path(folder)

    if folder.suffix == ".zarr":
        return _guess_object_from_zarr(folder)

    if (folder / "spikeinterface_info.json").is_file():
        # can be SortingAnalyzer | Motion
        with open(folder / "spikeinterface_info.json", "r") as f:
            spikeinterface_info = json.load(f)
            return _guess_object_from_dict(spikeinterface_info)
    elif (folder / "waveforms").is_dir():
        # before the SortingAnlazer, it was WaveformExtractor (v<0.101)
        return "WaveformExtractor"
    elif (folder / f"si_folder.json").is_file():
        # In later versions (0.94<v<0.102) we use the si_folder.json file
        # This should be Recording | Sorting
        return "Recording|Sorting"
    else:
        # For backward compatibility (v<=0.94) we check for the cached.json/pkl/pickle files
        for dump_ext in ("json", "pkl", "pickle"):
            f = folder / f"cached.{dump_ext}"
            if f.is_file():
                file = f
        return "Recording|Sorting"


def _load_object_from_folder(folder, object_type, **kwargs):
    if object_type == "SortingAnalyzer":
        from .sortinganalyzer import load_sorting_analyzer

        analyzer = load_sorting_analyzer(folder, **kwargs)
        return analyzer

    elif object_type == "Motion":
        from spikeinterface.core.motion import Motion

        motion = Motion.load(folder)
        return motion

    elif object_type == "WaveformExtractor":
        from .waveforms_extractor_backwards_compatibility import load_waveforms

        analyzer = load_waveforms(folder, output="SortingAnalyzer")
        return analyzer

    elif object_type in ("Recording", "Sorting", "Recording|Sorting"):
        if (folder / f"si_folder.json").is_file():
            # In later versions (0.94<v<0.102) we use the si_folder.json file
            # This should be Recording | Sorting
            si_file = folder / f"si_folder.json"
        else:
            # For backward compatibility (v<=0.94) we check for the cached.json/pkl/pickle files
            for dump_ext in ("json", "pkl", "pickle"):
                f = folder / f"cached.{dump_ext}"
                if f.is_file():
                    si_file = f
        return BaseExtractor.load(si_file, base_folder=folder)


def _guess_object_from_zarr(zarr_folder):
    # here it can be a zarr folder for Recording|Sorting|SortingAnalyzer|Template
    from .zarrextractors import super_zarr_open

    zarr_root = super_zarr_open(zarr_folder, mode="r")

    # can be SortingAnalyzer
    spikeinterface_info = zarr_root.attrs.get("spikeinterface_info")
    if spikeinterface_info is not None:
        return _guess_object_from_dict(spikeinterface_info)

    # here it is the old fashion and a bit ambiguous
    if "templates_array" in zarr_root.keys():
        return "Templates"
    elif "channel_ids" in zarr_root.keys() and "unit_ids" not in zarr_root.keys():
        return "Recording"
    elif "unit_ids" in zarr_root.keys() and "channel_ids" not in zarr_root.keys():
        return "Sorting"


def _load_object_from_zarr(folder_or_url, object_type, **kwargs):
    if object_type == "SortingAnalyzer":
        from .sortinganalyzer import load_sorting_analyzer

        backend_options = kwargs.get("backend_options", None)
        load_extensions = kwargs.get("load_extensions", True)
        analyzer = load_sorting_analyzer(
            folder_or_url, backend_options=backend_options, load_extensions=load_extensions
        )
        return analyzer
    elif object_type == "Templates":
        from .template import Templates

        templates = Templates.from_zarr(folder_or_url)
        return templates
    elif object_type == "Recording":
        from .zarrextractors import read_zarr_recording

        storage_options = kwargs.get("storage_options", None)
        recording = read_zarr_recording(folder_or_url, storage_options=storage_options)
        return recording
    elif object_type == "Sorting":
        from .zarrextractors import read_zarr_sorting

        storage_options = kwargs.get("storage_options", None)
        sorting = read_zarr_sorting(folder_or_url, storage_options=storage_options)
        return sorting
    elif object_type == "Recording|Sorting":
        # This case shoudl deprecated soon because the read_zarr is ultra ambiguous
        # just testing if the zarr contains unit_ids or channel_ids but many object also contains it (see template)!!!!
        from .zarrextractors import read_zarr

        storage_options = kwargs.get("storage_options", None)
        rec_or_sorting = read_zarr(folder_or_url, storage_options=storage_options)
        return rec_or_sorting
