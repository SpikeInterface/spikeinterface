from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union
import importlib
import warnings
import weakref
import json
import pickle
import os
import random
import string
from packaging.version import parse
from copy import deepcopy

import numpy as np

from .globals import get_global_tmp_folder, is_set_global_tmp_folder
from .core_tools import check_json, is_dict_extractor, recursive_path_modifier, SIJsonEncoder
from .job_tools import _shared_job_kwargs_doc


class BaseExtractor:
    """
    Base class for Recording/Sorting

    Handle serialization save/load to/from a folder.

    """

    default_missing_property_values = {"f": np.nan, "O": None, "S": "", "U": ""}

    # This replaces the old key_properties
    # These are annotations/properties that always need to be
    # dumped (for instance locations, groups, is_fileterd, etc.)
    _main_annotations = []
    _main_properties = []

    # these properties are skipped by default in copy_metadata
    _skip_properties = []

    installed = True
    installation_mesg = ""

    def __init__(self, main_ids: Sequence) -> None:
        # store init kwargs for nested serialisation
        self._kwargs = {}

        # 'main_ids' will either be channel_ids or units_ids
        # They is used for properties
        self._main_ids = np.array(main_ids)

        # dict at object level
        self._annotations = {}

        # properties is a dict of arrays
        # array length is :
        #  * number of channel for recording
        #  * number of units for sorting
        self._properties = {}

        self._is_dumpable = True
        self._is_json_serializable = True

        # extractor specific list of pip extra requirements
        self.extra_requirements = []

        # preferred context for multiprocessing
        self._preferred_mp_context = None

    def get_num_segments(self) -> int:
        # This is implemented in BaseRecording or BaseSorting
        raise NotImplementedError

    def _check_segment_index(self, segment_index: Optional[int] = None) -> int:
        if segment_index is None:
            if self.get_num_segments() == 1:
                return 0
            else:
                raise ValueError("Multi-segment object. Provide 'segment_index'")
        else:
            return segment_index

    def ids_to_indices(self, ids: Iterable, prefer_slice: bool = False) -> Union[np.ndarray, slice]:
        """
        Transform a ids list (aka channel_ids or unit_ids)
        into a indices array.
        Useful to manipulate:
          * data
          * properties

        'prefer_slice' is an efficient option that tries to make a slice object
        when indices are consecutive.

        """
        if ids is None:
            if prefer_slice:
                indices = slice(None)
            else:
                indices = np.arange(len(self._main_ids))
        else:
            _main_ids = self._main_ids.tolist()
            indices = np.array([_main_ids.index(id) for id in ids], dtype=int)
            if prefer_slice:
                if np.all(np.diff(indices) == 1):
                    indices = slice(indices[0], indices[-1] + 1)
        return indices

    def id_to_index(self, id) -> int:
        ind = list(self._main_ids).index(id)
        return ind

    def annotate(self, **new_annotations) -> None:
        self._annotations.update(new_annotations)

    def set_annotation(self, annotation_key, value: Any, overwrite=False) -> None:
        """This function adds an entry to the annotations dictionary.

        Parameters
        ----------
        annotation_key: str
            An annotation stored by the Extractor
        value:
            The data associated with the given property name. Could be many
            formats as specified by the user
        overwrite: bool
            If True and the annotation already exists, it is overwritten
        """
        if annotation_key not in self._annotations.keys():
            self._annotations[annotation_key] = value
        else:
            if overwrite:
                self._annotations[annotation_key] = value
            else:
                raise ValueError(f"{annotation_key} is already an annotation key. Use 'overwrite=True' to overwrite it")

    def get_preferred_mp_context(self):
        """
        Get the preferred context for multiprocessing.
        If None, the context is set by the multiprocessing package.
        """
        return self._preferred_mp_context

    def get_annotation(self, key, copy: bool = True) -> Any:
        """
        Get a annotation.
        Return a copy by default
        """
        v = self._annotations.get(key, None)
        if copy:
            v = deepcopy(v)
        return v

    def get_annotation_keys(self) -> List:
        return list(self._annotations.keys())

    def set_property(self, key, values: Sequence, ids: Optional[Sequence] = None, missing_value: Any = None) -> None:
        """
        Set property vector for main ids.

        If ids is given AND property already exists,
        then it is modified only on a subset of channels/units.
        missing_values allows to specify the values of unset
        properties if ids is used


        Parameters
        ----------
        key : str
            The property name
        values : np.array
            Array of values for the property
        ids : list/np.array, optional
            List of subset of ids to set the values, by default None
        missing_value : object, optional
            In case the property is set on a subset of values ('ids' not None),
            it specifies the how the missing values should be filled, by default None.
            The missing_value has to be specified for types int and unsigned int.
        """

        if values is None:
            if key in self._properties:
                self._properties.pop(key)
            return

        size = self._main_ids.size
        values = np.asarray(values)
        dtype = values.dtype
        dtype_kind = dtype.kind

        if ids is None:
            assert values.shape[0] == size
            self._properties[key] = values
        else:
            ids = np.array(ids)
            assert np.unique(ids).size == ids.size, "'ids' are not unique!"

            if ids.size < size:
                if key not in self._properties:
                    # create the property with nan or empty
                    shape = (size,) + values.shape[1:]

                    if missing_value is None:
                        if dtype_kind not in self.default_missing_property_values.keys():
                            raise Exception(
                                "For values dtypes other than float, string, object or unicode, the missing value "
                                "cannot be automatically inferred. Please specify it with the 'missing_value' "
                                "argument."
                            )
                        else:
                            missing_value = self.default_missing_property_values[dtype_kind]
                    else:
                        assert dtype_kind == np.array(missing_value).dtype.kind, (
                            "Mismatch between values and "
                            "missing_value types. Provide a "
                            "missing_value with the same type "
                            "as the values."
                        )

                    empty_values = np.zeros(shape, dtype=dtype)
                    empty_values[:] = missing_value
                    self._properties[key] = empty_values
                    if ids.size == 0:
                        return
                else:
                    assert dtype_kind == self._properties[key].dtype.kind, (
                        "Mismatch between existing property dtype " "values dtype."
                    )

                indices = self.ids_to_indices(ids)
                self._properties[key][indices] = values
            else:
                indices = self.ids_to_indices(ids)
                self._properties[key] = np.zeros_like(values, dtype=values.dtype)
                self._properties[key][indices] = values

    def get_property(self, key, ids: Optional[Iterable] = None) -> np.ndarray:
        values = self._properties.get(key, None)
        if ids is not None and values is not None:
            inds = self.ids_to_indices(ids)
            values = values[inds]
        return values

    def get_property_keys(self) -> List:
        return list(self._properties.keys())

    def delete_property(self, key) -> None:
        if key in self._properties:
            del self._properties[key]
        else:
            raise Exception(f"{key} is not a property key")

    def copy_metadata(
        self,
        other: "BaseExtractor",
        only_main: bool = False,
        ids: Union[Iterable, slice, None] = None,
        skip_properties: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Copy metadata (annotations/properties) to another extractor (`other`).

        Parameters
        ----------
        other: BaseExtractor
            The extractor to copy the metadata to.
        only_main: bool
            If True, only the main annotations/properties are copied.
        ids: list
            List of ids to copy the metadata to. If None, all ids are copied.
        skip_properties: list
            List of properties to skip. Default is None.
        """

        if ids is None:
            inds = slice(None)
        elif len(ids) == 0:
            inds = slice(0, 0)
        else:
            inds = self.ids_to_indices(ids)

        if only_main:
            ann_keys = BaseExtractor._main_annotations
            prop_keys = BaseExtractor._main_properties
        else:
            ann_keys = self._annotations.keys()
            prop_keys = self._properties.keys()

        other._annotations = deepcopy({k: self._annotations[k] for k in ann_keys})

        # skip properties based on target "other" extractor
        skip_properties_all = other._skip_properties
        if skip_properties is not None:
            skip_properties_all = skip_properties_all + skip_properties

        for k in prop_keys:
            if k in skip_properties_all:
                continue
            values = self._properties[k]
            if values is not None:
                other.set_property(k, values[inds])

        other.extra_requirements.extend(self.extra_requirements)

        if self._preferred_mp_context is not None:
            other._preferred_mp_context = self._preferred_mp_context

    def to_dict(
        self,
        include_annotations: bool = False,
        include_properties: bool = False,
        relative_to: Union[str, Path, None] = None,
        folder_metadata=None,
        recursive: bool = False,
    ) -> dict:
        """
        Make a nested serialized dictionary out of the extractor. The dictionary produced can be used to re-initialize
        an extractor using load_extractor_from_dict(dump_dict)

        Parameters
        ----------
        include_annotations: bool
            If True, all annotations are added to the dict, by default False
        include_properties: bool
            If True, all properties are added to the dict, by default False
        relative_to: str, Path, or None
            If not None, files and folders are serialized relative to this path, by default None
            Used in waveform extractor to maintain relative paths to binary files even if the
            containing folder / diretory is moved
        folder_metadata: str, Path, or None
            Folder with numpy `npy` files containing additional information (e.g. probe in BaseRecording) and properties.
        recursive: bool
            If True, all dicitionaries in the kwargs are expanded with `to_dict` as well, by default False.

        Returns
        -------
        dump_dict: dict
            A dictionary representation of the extractor.
        """

        kwargs = self._kwargs

        if relative_to and not recursive:
            raise ValueError("`relative_to` is only posible when `recursive=True`")

        if recursive:
            to_dict_kwargs = dict(
                include_annotations=include_annotations,
                include_properties=include_properties,
                relative_to=None,  # '_make_paths_relative' is already recursive!
                folder_metadata=folder_metadata,
                recursive=recursive,
            )

            new_kwargs = dict()
            transform_extractors_to_dict = lambda x: x.to_dict(**to_dict_kwargs) if isinstance(x, BaseExtractor) else x

            for name, value in self._kwargs.items():
                if isinstance(value, list):
                    new_kwargs[name] = [transform_extractors_to_dict(element) for element in value]
                elif isinstance(value, dict):
                    new_kwargs[name] = {k: transform_extractors_to_dict(v) for k, v in value.items()}
                else:
                    new_kwargs[name] = transform_extractors_to_dict(value)

            kwargs = new_kwargs
        class_name = str(type(self)).replace("<class '", "").replace("'>", "")
        module = class_name.split(".")[0]
        imported_module = importlib.import_module(module)

        try:
            version = imported_module.__version__
        except AttributeError:
            version = "unknown"

        dump_dict = {
            "class": class_name,
            "module": module,
            "kwargs": kwargs,
            "version": version,
            "relative_paths": (relative_to is not None),
        }

        try:
            dump_dict["version"] = imported_module.__version__
        except AttributeError:
            dump_dict["version"] = "unknown"

        if include_annotations:
            dump_dict["annotations"] = self._annotations
        else:
            # include only main annotations
            dump_dict["annotations"] = {k: self._annotations.get(k, None) for k in self._main_annotations}

        if include_properties:
            dump_dict["properties"] = self._properties
        else:
            # include only main properties
            dump_dict["properties"] = {k: self._properties.get(k, None) for k in self._main_properties}

        if relative_to is not None:
            relative_to = Path(relative_to).resolve().absolute()
            assert relative_to.is_dir(), "'relative_to' must be an existing directory"
            dump_dict = _make_paths_relative(dump_dict, relative_to)

        if folder_metadata is not None:
            if relative_to is not None:
                folder_metadata = Path(folder_metadata).resolve().absolute().relative_to(relative_to)
            dump_dict["folder_metadata"] = str(folder_metadata)

        return dump_dict

    @staticmethod
    def from_dict(dictionary: dict, base_folder: Optional[Union[Path, str]] = None) -> "BaseExtractor":
        """
        Instantiate extractor from dictionary

        Parameters
        ----------
        d: dictionary
            Python dictionary
        base_folder: str, Path, or None
            If given, the parent folder of the file and folder paths

        Returns
        -------
        extractor: RecordingExtractor or SortingExtractor
            The loaded extractor object
        """
        if dictionary["relative_paths"]:
            assert base_folder is not None, "When  relative_paths=True, need to provide base_folder"
            dictionary = _make_paths_absolute(dictionary, base_folder)
        extractor = _load_extractor_from_dict(dictionary)
        folder_metadata = dictionary.get("folder_metadata", None)
        if folder_metadata is not None:
            folder_metadata = Path(folder_metadata)
            if dictionary["relative_paths"]:
                folder_metadata = base_folder / folder_metadata
            extractor.load_metadata_from_folder(folder_metadata)
        return extractor

    def load_metadata_from_folder(self, folder_metadata):
        # hack to load probe for recording
        folder_metadata = Path(folder_metadata)

        self._extra_metadata_from_folder(folder_metadata)

        # load properties
        prop_folder = folder_metadata / "properties"
        if prop_folder.is_dir():
            for prop_file in prop_folder.iterdir():
                if prop_file.suffix == ".npy":
                    values = np.load(prop_file, allow_pickle=True)
                    key = prop_file.stem
                    self.set_property(key, values)

    def save_metadata_to_folder(self, folder_metadata):
        self._extra_metadata_to_folder(folder_metadata)

        # save properties
        prop_folder = folder_metadata / "properties"
        prop_folder.mkdir(parents=True, exist_ok=False)
        for key in self.get_property_keys():
            values = self.get_property(key)
            np.save(prop_folder / (key + ".npy"), values)

    def clone(self) -> "BaseExtractor":
        """
        Clones an existing extractor into a new instance.
        """
        d = self.to_dict(include_annotations=True, include_properties=True)
        d = deepcopy(d)
        clone = BaseExtractor.from_dict(d)
        return clone

    def check_if_dumpable(self):
        """Check if the object is dumpable, including nested objects.

        Returns
        -------
        bool
            True if the object is dumpable, False otherwise.
        """
        kwargs = self._kwargs
        for value in kwargs.values():
            # here we check if the value is a BaseExtractor, a list of BaseExtractors, or a dict of BaseExtractors
            if isinstance(value, BaseExtractor):
                return value.check_if_dumpable()
            elif isinstance(value, list) and (len(value) > 0) and isinstance(value[0], BaseExtractor):
                return all([v.check_if_dumpable() for v in value])
            elif isinstance(value, dict) and isinstance(value[list(value.keys())[0]], BaseExtractor):
                return all([v.check_if_dumpable() for k, v in value.items()])
        return self._is_dumpable

    def check_if_json_serializable(self):
        """
        Check if the object is json serializable, including nested objects.

        Returns
        -------
        bool
            True if the object is json serializable, False otherwise.
        """
        kwargs = self._kwargs
        for value in kwargs.values():
            # here we check if the value is a BaseExtractor, a list of BaseExtractors, or a dict of BaseExtractors
            if isinstance(value, BaseExtractor):
                return value.check_if_json_serializable()
            elif isinstance(value, list) and (len(value) > 0) and isinstance(value[0], BaseExtractor):
                return all([v.check_if_json_serializable() for v in value])
            elif isinstance(value, dict) and isinstance(value[list(value.keys())[0]], BaseExtractor):
                return all([v.check_if_json_serializable() for k, v in value.items()])
        return self._is_json_serializable

    @staticmethod
    def _get_file_path(file_path: Union[str, Path], extensions: Sequence) -> Path:
        """
        Helper function to be used by various dump_to_file utilities.

        Returns default file_path (if not specified), makes sure that target
        directory exists, adds correct file extension if none, and checks
        that the provided file extension is allowed.

        Parameters
        ----------
        file_path: str or None
        extensions: list or tuple
            List of possible extensions. The first one provided is used as
            an extension for the default file_path.

        Returns
        -------
        Path
            Path object with file path to the file
        """
        ext = extensions[0]
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        folder_path = file_path.parent
        if Path(file_path).suffix == "":
            file_path = folder_path / (str(file_path) + ext)
        assert file_path.suffix in extensions, "'file_path' should have one of the following extensions:" " %s" % (
            ", ".join(extensions)
        )
        return file_path

    def dump(self, file_path: Union[str, Path], relative_to=None, folder_metadata=None) -> None:
        """
        Dumps extractor to json or pickle

        Parameters
        ----------
        file_path: str or Path
            The output file (either .json or .pkl/.pickle)
        relative_to: str, Path, True or None
            If not None, files and folders are serialized relative to this path. If True, the relative folder is the parent folder.
            This means that file and folder paths in extractor objects kwargs are changed to be relative rather than absolute.
        """
        if str(file_path).endswith(".json"):
            self.dump_to_json(file_path, relative_to=relative_to, folder_metadata=folder_metadata)
        elif str(file_path).endswith(".pkl") or str(file_path).endswith(".pickle"):
            self.dump_to_pickle(file_path, relative_to=relative_to, folder_metadata=folder_metadata)
        else:
            raise ValueError("Dump: file must .json or .pkl")

    def dump_to_json(self, file_path: Union[str, Path, None] = None, relative_to=None, folder_metadata=None) -> None:
        """
        Dump recording extractor to json file.
        The extractor can be re-loaded with load_extractor(json_file)

        Parameters
        ----------
        file_path: str
            Path of the json file
        relative_to: str, Path, True or None
            If not None, files and folders are serialized relative to this path. If True, the relative folder is the parent folder.
            This means that file and folder paths in extractor objects kwargs are changed to be relative rather than absolute.
        folder_metadata: str, Path, or None
            Folder with files containing additional information (e.g. probe in BaseRecording) and properties.
        """
        assert self.check_if_json_serializable(), "The extractor is not json serializable"

        # Writing paths as relative_to requires recursively expanding the dict
        if relative_to:
            relative_to = Path(file_path).parent if relative_to is True else Path(relative_to)
            relative_to = relative_to.resolve().absolute()

        dump_dict = self.to_dict(
            include_annotations=True,
            include_properties=False,
            relative_to=relative_to,
            folder_metadata=folder_metadata,
            recursive=True,
        )
        file_path = self._get_file_path(file_path, [".json"])

        file_path.write_text(
            json.dumps(dump_dict, indent=4, cls=SIJsonEncoder),
            encoding="utf8",
        )

    def dump_to_pickle(
        self,
        file_path: Union[str, Path, None] = None,
        include_properties: bool = True,
        folder_metadata=None,
    ):
        """
        Dump recording extractor to a pickle file.
        The extractor can be re-loaded with load_extractor(pickle_file)

        Parameters
        ----------
        file_path: str
            Path of the json file
        include_properties: bool
            If True, all properties are dumped
        folder_metadata: str, Path, or None
            Folder with files containing additional information (e.g. probe in BaseRecording) and properties.
        """
        assert self.check_if_dumpable(), "The extractor is not dumpable"

        dump_dict = self.to_dict(
            include_annotations=True,
            include_properties=include_properties,
            folder_metadata=folder_metadata,
            recursive=False,
        )
        file_path = self._get_file_path(file_path, [".pkl", ".pickle"])

        file_path.write_bytes(pickle.dumps(dump_dict))

    @staticmethod
    def load(file_path: Union[str, Path], base_folder: Optional[Union[Path, str, bool]] = None) -> "BaseExtractor":
        """
        Load extractor from file path (.json or .pkl)

        Used both after:
          * dump(...) json or pickle file
          * save (...)  a folder which contain data  + json (or pickle) + metadata.

        """

        file_path = Path(file_path)
        if base_folder is True:
            base_folder = file_path.parent

        if file_path.is_file():
            # standard case based on a file (json or pickle)
            if str(file_path).endswith(".json"):
                with open(file_path, "r") as f:
                    d = json.load(f)
            elif str(file_path).endswith(".pkl") or str(file_path).endswith(".pickle"):
                with open(file_path, "rb") as f:
                    d = pickle.load(f)
            else:
                raise ValueError(f"Impossible to load {file_path}")
            if "warning" in d and "not dumpable" in d["warning"]:
                print("The extractor was not dumpable")
                return None
            extractor = BaseExtractor.from_dict(d, base_folder=base_folder)
            return extractor

        elif file_path.is_dir():
            # case from a folder after a calling extractor.save(...)
            folder = file_path
            file = None

            if folder.suffix == ".zarr":
                from .zarrrecordingextractor import read_zarr

                extractor = read_zarr(folder)
            else:
                # the is spikeinterface<=0.94.0
                # a folder came with 'cached.json'
                for dump_ext in ("json", "pkl", "pickle"):
                    f = folder / f"cached.{dump_ext}"
                    if f.is_file():
                        file = f

                # spikeinterface>=0.95.0
                f = folder / f"si_folder.json"
                if f.is_file():
                    file = f

                if file is None:
                    raise ValueError(f"This folder is not a cached folder {file_path}")
                extractor = BaseExtractor.load(file, base_folder=folder)

            return extractor

        else:
            raise ValueError("spikeinterface.Base.load() file_path must be an existing folder or file")

    def __reduce__(self):
        """
        This function is used by pickle to serialize the object.
        """
        instance_constructor = self.from_dict
        intialization_args = (self.to_dict(),)
        return (instance_constructor, intialization_args)

    @staticmethod
    def load_from_folder(folder) -> "BaseExtractor":
        return BaseExtractor.load(folder)

    def _save(self, folder, **save_kwargs):
        # This implemented in BaseRecording or baseSorting
        # this is internally call by cache(...) main function
        raise NotImplementedError

    def _extra_metadata_from_folder(self, folder):
        # This implemented in BaseRecording for probe
        pass

    def _extra_metadata_to_folder(self, folder):
        # This implemented in BaseRecording for probe
        pass

    def save(self, **kwargs) -> "BaseExtractor":
        """
        Save a SpikeInterface object.

        Parameters
        ----------
        kwargs: Keyword arguments for saving.
            * format: "memory", "zarr", or "binary" (for recording) / "memory" or "npz" for sorting.
                In case format is not memory, the recording is saved to a folder. See format specific functions for
                more info (`save_to_memory()`, `save_to_folder()`, `save_to_zarr()`)
            * folder: if provided, the folder path where the object is saved
            * name: if provided and folder is not given, the name of the folder in the global temporary
                    folder (use set_global_tmp_folder() to change this folder) where the object is saved.
              If folder and name are not given, the object is saved in the global temporary folder with
              a random string
            * dump_ext: 'json' or 'pkl', default 'json' (if format is "folder")
            * verbose: if True output is verbose
            * **save_kwargs: additional kwargs format-dependent and job kwargs for recording
            {}

        Returns
        -------
        loaded_extractor: BaseRecording or BaseSorting
            The reference to the saved object after it is loaded back
        """
        format = kwargs.get("format", None)
        if format == "memory":
            loaded_extractor = self.save_to_memory(**kwargs)
        elif format == "zarr":
            loaded_extractor = self.save_to_zarr(**kwargs)
        else:
            loaded_extractor = self.save_to_folder(**kwargs)
        return loaded_extractor

    save.__doc__ = save.__doc__.format(_shared_job_kwargs_doc)

    def save_to_memory(self, **kwargs) -> "BaseExtractor":
        # used only by recording at the moment
        cached = self._save(**kwargs)
        self.copy_metadata(cached)
        return cached

    # TODO rename to saveto_binary_folder
    def save_to_folder(self, name=None, folder=None, verbose=True, **save_kwargs):
        """
        Save extractor to folder.

        The save consist of:
          * extracting traces by calling get_trace() method in chunks
          * saving data into file (memmap with BinaryRecordingExtractor)
          * dumping to json/pickle the original extractor for provenance
          * dumping to json/pickle the cached extractor (memmap with BinaryRecordingExtractor)

        This replaces the use of the old CacheRecordingExtractor and CacheSortingExtractor.

        There are 2 option for the 'folder' argument:
          * explicit folder: `extractor.save(folder="/path-for-saving/")`
          * explicit sub-folder, implicit base-folder : `extractor.save(name="extarctor_name")`
          * generated: `extractor.save()`

        The second option saves to subfolder "extarctor_name" in
        "get_global_tmp_folder()". You can set the global tmp folder with:
        "set_global_tmp_folder("path-to-global-folder")"

        The folder must not exist. If it exists, remove it before.

        Parameters
        ----------
        name: None str or Path
            Name of the subfolder in get_global_tmp_folder()
            If 'name' is given, 'folder' must be None.
        folder: None str or Path
            Name of the folder.
            If 'folder' is given, 'name' must be None.

        Returns
        -------
        cached: saved copy of the extractor.
        """

        if folder is None:
            cache_folder = get_global_tmp_folder()
            if name is None:
                name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
                folder = cache_folder / name
                if verbose:
                    print(f"Use cache_folder={folder}")
            else:
                folder = cache_folder / name
                if not is_set_global_tmp_folder():
                    if verbose:
                        print(f"Use cache_folder={folder}")
        else:
            folder = Path(folder)
        assert not folder.exists(), f"folder {folder} already exists, choose another name"
        folder.mkdir(parents=True, exist_ok=False)

        # dump provenance
        provenance_file = folder / f"provenance.json"
        if self.check_if_json_serializable():
            self.dump(provenance_file)
        else:
            provenance_file.write_text(json.dumps({"warning": "the provenace is not dumpable!!!"}), encoding="utf8")

        self.save_metadata_to_folder(folder)

        # save data (done the subclass)
        cached = self._save(folder=folder, verbose=verbose, **save_kwargs)

        # copy properties/
        self.copy_metadata(cached)

        # dump
        # cached.dump(folder / f'cached.json', relative_to=folder, folder_metadata=folder)
        cached.dump(folder / f"si_folder.json", relative_to=folder)

        return cached

    def save_to_zarr(
        self,
        name=None,
        folder=None,
        storage_options=None,
        channel_chunk_size=None,
        verbose=True,
        zarr_path=None,
        **save_kwargs,
    ):
        """
        Save extractor to zarr.

        The save consist of:
            * extracting traces by calling get_trace() method in chunks
            * saving data into a zarr file
            * dumping the original extractor for provenance in attributes

        Parameters
        ----------
        name: str or None
            Name of the subfolder in get_global_tmp_folder()
            If 'name' is given, 'folder' must be None.
        folder: str, Path, or None
            The folder used to save the zarr output. If the folder does not have a '.zarr' suffix,
            it will be automatically appended.
        storage_options: dict or None
            Storage options for zarr `store`. E.g., if "s3://" or "gcs://" they can provide authentication methods, etc.
            For cloud storage locations, this should not be None (in case of default values, use an empty dict)
        channel_chunk_size: int or None
            Channels per chunk. Default None (chunking in time only)
        verbose: bool
            If True (default), the output is verbose.
        zarr_path: str, Path, or None
            (Deprecated) Name of the zarr folder (.zarr).

        Returns
        -------
        cached: ZarrRecordingExtractor
            Saved copy of the extractor.
        """
        import zarr
        from .zarrrecordingextractor import read_zarr

        if zarr_path is not None:
            warnings.warn(
                "The 'zarr_path' argument is deprecated. " "Use 'folder' instead", DeprecationWarning, stacklevel=2
            )
            folder = zarr_path

        if folder is None:
            cache_folder = get_global_tmp_folder()
            if name is None:
                name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
                zarr_path = cache_folder / f"{name}.zarr"
                if verbose:
                    print(f"Use zarr_path={zarr_path}")
            else:
                zarr_path = cache_folder / f"{name}.zarr"
                if not is_set_global_tmp_folder():
                    if verbose:
                        print(f"Use zarr_path={zarr_path}")
        else:
            if storage_options is None:
                folder = Path(folder)
                if folder.suffix != ".zarr":
                    folder = folder.parent / f"{folder.stem}.zarr"
                zarr_path = folder
                zarr_path_init = str(zarr_path)
            else:
                zarr_path = folder
                zarr_path_init = zarr_path

        if isinstance(zarr_path, Path):
            assert not zarr_path.exists(), f"Path {zarr_path} already exists, choose another name"

        zarr_root = zarr.open(zarr_path_init, mode="w", storage_options=storage_options)

        if self.check_if_dumpable():
            zarr_root.attrs["provenance"] = check_json(self.to_dict())
        else:
            zarr_root.attrs["provenance"] = None

        # save data (done the subclass)
        save_kwargs["zarr_root"] = zarr_root
        save_kwargs["zarr_path"] = zarr_path
        save_kwargs["storage_options"] = storage_options
        save_kwargs["channel_chunk_size"] = channel_chunk_size
        cached = self._save(verbose=verbose, **save_kwargs)

        # save properties
        prop_group = zarr_root.create_group("properties")
        for key in self.get_property_keys():
            values = self.get_property(key)
            prop_group.create_dataset(name=key, data=values, compressor=None)

        # save annotations
        zarr_root.attrs["annotations"] = check_json(self._annotations)

        cached = read_zarr(zarr_path)

        return cached


def _make_paths_relative(d, relative) -> dict:
    relative = str(Path(relative).resolve().absolute())
    func = lambda p: os.path.relpath(str(p), start=relative)
    return recursive_path_modifier(d, func, target="path", copy=True)


def _make_paths_absolute(d, base):
    base = Path(base)
    func = lambda p: str((base / p).resolve().absolute())
    return recursive_path_modifier(d, func, target="path", copy=True)


def _load_extractor_from_dict(dic) -> BaseExtractor:
    """
    Convert a dictionary into an instance of BaseExtractor or its subclass.

    This function takes a dictionary that represents the state of an extractor,
    and reconstructs the extractor from the dictionary.

    Parameters
    ----------
    dic : dict
        A dictionary representation of the extractor which  must contain the following keys:
        - "kwargs": A dictionary of keyword arguments to be passed to the extractor class upon instantiation.
        - "class": The full name of the extractor class, including the module name.
        - "version": The version of the extractor class used when the dictionary was created.
        - "annotations": A dictionary of annotations.
        - "properties": A dictionary of properties.

    Returns
    -------
    BaseExtractor
        An instance of BaseExtractor or its subclass.
    """
    extractor_class = None
    class_name = None

    if "kwargs" not in dic:
        raise Exception(f"This dict cannot be load into extractor {dic}")

    # Create new kwargs to avoid modifying the original dict["kwargs"]
    new_kwargs = dict()
    transform_dict_to_extractor = lambda x: _load_extractor_from_dict(x) if is_dict_extractor(x) else x
    for name, value in dic["kwargs"].items():
        if is_dict_extractor(value):
            new_kwargs[name] = _load_extractor_from_dict(value)
        elif isinstance(value, dict):
            new_kwargs[name] = {k: transform_dict_to_extractor(v) for k, v in value.items()}
        elif isinstance(value, list):
            new_kwargs[name] = [transform_dict_to_extractor(e) for e in value]
        else:
            new_kwargs[name] = value

    class_name = dic["class"]
    extractor_class = _get_class_from_string(class_name)

    assert extractor_class is not None and class_name is not None, "Could not load spikeinterface class"
    if not _check_same_version(class_name, dic["version"]):
        warnings.warn(
            f"Versions are not the same. This might lead compatibility errors. "
            f"Using {class_name.split('.')[0]}=={dic['version']} is recommended"
        )

    # Initialize the extractor
    extractor = extractor_class(**new_kwargs)

    extractor._annotations.update(dic["annotations"])
    for k, v in dic["properties"].items():
        extractor.set_property(k, v)

    return extractor


def _get_class_from_string(class_string):
    class_name = class_string.split(".")[-1]
    module = ".".join(class_string.split(".")[:-1])
    imported_module = importlib.import_module(module)

    try:
        imported_class = getattr(imported_module, class_name)
    except:
        imported_class = None

    return imported_class


def _check_same_version(class_string, version):
    module = class_string.split(".")[0]
    imported_module = importlib.import_module(module)

    current_version = parse(imported_module.__version__)
    saved_version = parse(version)

    try:
        return current_version.major == saved_version.major and current_version.minor == saved_version.minor
    except AttributeError:
        return "unknown"


def load_extractor(file_or_folder_or_dict, base_folder=None) -> BaseExtractor:
    """
    Instantiate extractor from:
      * a dict
      * a json file
      * a pickle file
      * folder (after save)
      * a zarr folder (after save)

    Parameters
    ----------
    file_or_folder_or_dict : dictionary or folder or file (json, pickle)
        The file path, folder path, or dictionary to load the extractor from
    base_folder : str | Path | bool (optional)
        The base folder to make relative paths absolute.
        If True and file_or_folder_or_dict is a file, the parent folder of the file is used.

    Returns
    -------
    extractor: Recording or Sorting
        The loaded extractor object
    """
    if isinstance(file_or_folder_or_dict, dict):
        assert not isinstance(base_folder, bool), "`base_folder` must be a string or Path when loading from dict"
        return BaseExtractor.from_dict(file_or_folder_or_dict, base_folder=base_folder)
    else:
        return BaseExtractor.load(file_or_folder_or_dict, base_folder=base_folder)


def load_extractor_from_dict(d, base_folder=None) -> BaseExtractor:
    warnings.warn("Use load_extractor(..) instead")
    return BaseExtractor.from_dict(d, base_folder=base_folder)


def load_extractor_from_json(json_file, base_folder=None) -> "BaseExtractor":
    warnings.warn("Use load_extractor(..) instead")
    return BaseExtractor.load(json_file, base_folder=base_folder)


def load_extractor_from_pickle(pkl_file, base_folder=None) -> "BaseExtractor":
    warnings.warn("Use load_extractor(..) instead")
    return BaseExtractor.load(pkl_file, base_folder=base_folder)


class BaseSegment:
    def __init__(self):
        self._parent_extractor = None

    @property
    def parent_extractor(self) -> Union[BaseExtractor, None]:
        return self._parent_extractor()

    def set_parent_extractor(self, parent_extractor: BaseExtractor) -> None:
        self._parent_extractor = weakref.ref(parent_extractor)
