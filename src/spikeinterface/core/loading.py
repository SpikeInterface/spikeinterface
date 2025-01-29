import warnings
from pathlib import Path


from .base import BaseExtractor
from .core_tools import is_path_remote


def load(
    file_or_folder_or_dict, base_folder=None, load_extensions=True, backend_options=None
) -> "BaseExtractor | SortingAnalyzer":
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
        - a `SortingAnalyzer` object from :
            * binary folder
            * zarr folder
            * remote zarr folder
            * WaveformExtractor folder

    Parameters
    ----------
    file_or_folder_or_dict : dictionary or folder or file (json, pickle)
        The file path, folder path, or dictionary to load the Recording, Sorting, or SortingAnalyzer from
    base_folder : str | Path | bool | None, default: None
        The base folder to make relative paths absolute. Only used to load Recording/Sorting objects.
        If True and file_or_folder_or_dict is a file, the parent folder of the file is used.
    load_extensions : bool, default: True
        If True, the SortingAnalyzer extensions are loaded. Only used to load SortingAnalyzer objects.
    backend_options : dict | None, default: None
        The backend options to use when loading the object. Only used to load SortingAnalyzer objects.
        The dictionary can contain the following keys:
        - storage_options: dict | None (fsspec storage options)
        - saving_options: dict | None (additional saving options for creating and saving datasets)

    Returns
    -------
    spikeinterface object: Recording or Sorting or SortingAnalyzer
        The loaded spikeinterface object
    """
    if isinstance(file_or_folder_or_dict, dict):
        assert not isinstance(base_folder, bool), "`base_folder` must be a string or Path when loading from dict"
        return BaseExtractor.from_dict(file_or_folder_or_dict, base_folder=base_folder)
    else:
        file_path = file_or_folder_or_dict
        error_msg = (
            f"{file_path} is not a file or a folder. It should point to either a json, pickle file or a "
            "folder that is the result of extractor.save(...) or sortinganalyzer.save_as(...)"
        )
        if not is_path_remote(file_path):
            file_path = Path(file_path)

            if base_folder is True:
                base_folder = file_path.parent

            if file_path.is_file():
                # standard case based on a file (json or pickle)
                if str(file_path).endswith(".json"):
                    import json

                    with open(file_path, "r") as f:
                        d = json.load(f)
                elif str(file_path).endswith(".pkl") or str(file_path).endswith(".pickle"):
                    import pickle

                    with open(file_path, "rb") as f:
                        d = pickle.load(f)
                else:
                    raise ValueError(error_msg)

                # this is for back-compatibility since now unserializable objects will not
                # be saved to file
                if "warning" in d:
                    print("The extractor was not serializable to file")
                    return None

                si_object = BaseExtractor.from_dict(d, base_folder=base_folder)

            elif file_path.is_dir():
                # this can be and extractor, SortingAnalyzer, or WaveformExtractor
                folder = file_path
                file = None

                if folder.suffix == ".zarr":
                    # here it can be a zarr folder for extractor or SortingAnalyzer
                    import zarr

                    zarr_root = zarr.open(str(folder), mode="r")

                    if "spikeinterface_info" in zarr_root.attrs:
                        # SortingAnalyzer case
                        from .sortinganalyzer import load_sorting_analyzer

                        si_object = load_sorting_analyzer(folder, load_extensions=load_extensions)
                    else:
                        # Extractor case
                        from .zarrextractors import read_zarr

                        si_object = read_zarr(folder)
                else:
                    if (folder / "spikeinterface_info.json").is_file():
                        # SortingAnalyzer case
                        from .sortinganalyzer import load_sorting_analyzer

                        si_object = load_sorting_analyzer(folder, load_extensions=load_extensions)
                    elif (folder / "waveforms").is_dir():
                        # WaveformExtractor case
                        from .waveforms_extractor_backwards_compatibility import load_waveforms

                        si_object = load_waveforms(folder, output="SortingAnalyzer")
                    else:
                        # Extractor case

                        # For backward compatibility (v<=0.94) we check for the cached.json/pkl/pickle files
                        # In later versions (v>0.94) we use the si_folder.json file
                        for dump_ext in ("json", "pkl", "pickle"):
                            f = folder / f"cached.{dump_ext}"
                            if f.is_file():
                                file = f

                        f = folder / f"si_folder.json"
                        if f.is_file():
                            file = f

                        if file is None:
                            raise ValueError(error_msg)
                        si_object = BaseExtractor.load(file, base_folder=folder)

            else:
                raise ValueError(error_msg)
        else:
            # remote case - currently, only zarr is supported
            if str(file_path).endswith((".zarr", ".zarr/")):
                # here we can have a zarr folder for extractor or SortingAnalyzer
                # since the data is remote, we need to try and see if it is an extractor or a SortingAnalyzer
                try:
                    from .zarrextractors import read_zarr

                    storage_options = (
                        backend_options.get("storage_options", None) if backend_options is not None else None
                    )
                    si_object = read_zarr(file_path, storage_options=storage_options)
                except Exception as e:
                    try:
                        # if it is not an extractor, it is a SortingAnalyzer
                        from .sortinganalyzer import load_sorting_analyzer

                        si_object = load_sorting_analyzer(
                            file_path, load_extensions=load_extensions, backend_options=backend_options
                        )
                    except Exception as e:
                        raise ValueError(f"Could not load the remote zarr folder at {file_path}")
            else:
                raise NotImplementedError(
                    "Only zarr format is supported for remote files and you should provide a path to a .zarr "
                    "remote path. You can save to a valid zarr folder using: "
                    "`extractor.save(folder='path/to/folder', format='zarr')` or `analyzer.save_as(folder='path/to/folder', format='zarr')`"
                )

        return si_object


def load_extractor(file_or_folder_or_dict, base_folder=None) -> "BaseExtractor | SortingAnalyzer":
    warnings.warn(
        "load_extractor() is deprecated and will be removed in the future. Please use load() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load(file_or_folder_or_dict, base_folder=base_folder)
