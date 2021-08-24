from pathlib import Path
import importlib
from copy import deepcopy
import weakref
import json
import pickle
import datetime
import random
import string

import numpy as np

from .default_folders import get_global_tmp_folder, is_set_global_tmp_folder
from .core_tools import check_json


class BaseExtractor:
    """
    Base class for Recording/Sorting
    
    Handle serialization save/load to/from a folder.
    
    """

    # This replaces the old key_properties
    # These are annotations/properties/features that always need to be
    # dumped (for instance locations, groups, is_fileterd, etc.)
    _main_annotations = []
    _main_properties = []
    _main_features = []

    def __init__(self, main_ids):
        # store init kwargs for nested serialisation
        self._kwargs = {}

        # 'main_ids' will either be channel_ids or units_ids
        # They is used for properties and features
        self._main_ids = np.array(main_ids)

        # dict at object level
        self._annotations = {}

        # properties is a dict of arrays
        # array length is :
        #  * number of channel for recording
        #  * number of units for sorting
        self._properties = {}

        # features is a dict of arrays (at spike level)
        self._features = {}

        self.is_dumpable = True

    def get_num_segments(self):
        # This is implemented in BaseRecording or BaseSorting
        raise NotImplementedError

    def _check_segment_index(self, segment_index=None):
        if segment_index is None:
            if self.get_num_segments() == 1:
                return 0
            else:
                raise ValueError()
        else:
            return segment_index

    def ids_to_indices(self, ids, prefer_slice=False):
        """
        Transform a ids list (aka channel_ids or unit_ids)
        into a indices array.
        Useful to manipulate:
          * data
          * properties
          * features
        
        'prefer_slice' is an efficient option that tries to make a slice object
        when indices are consecutive.
        
        """
        if ids is None:
            if prefer_slice:
                indices = slice(None)
            else:
                indices = self._main_ids
        else:
            _main_ids = self._main_ids.tolist()
            indices = np.array([_main_ids.index(id) for id in ids])
            if prefer_slice:
                if np.all(np.diff(indices) == 1):
                    indices = slice(indices[0], indices[-1] + 1)
        return indices

    def id_to_index(self, id):
        ind = list(self._main_ids).index(id)
        return ind

    def annotate(self, **new_annotations):
        self._annotations.update(new_annotations)

    def set_annotation(self, annotation_key, value, overwrite=False):
        '''This function adds an entry to the annotations dictionary.

        Parameters
        ----------
        annotation_key: str
            An annotation stored by the Extractor
        value:
            The data associated with the given property name. Could be many
            formats as specified by the user
        overwrite: bool
            If True and the annotation already exists, it is overwritten
        '''
        if annotation_key not in self._annotations.keys():
            self._annotations[annotation_key] = value
        else:
            if overwrite:
                self._annotations[annotation_key] = value
            else:
                raise ValueError(f"{annotation_key} is already an annotation key. Use 'overwrite=True' to overwrite it")

    def get_annotation(self, key, copy=True):
        """
        Get a annotation.
        Return a copy by default
        """
        v = self._annotations.get(key, None)
        if copy:
            v = deepcopy(v)
        return v

    def get_annotation_keys(self):
        return list(self._annotations.keys())

    def set_property(self, key, values, ids=None):
        """
        Set property vector:
          * channel_property
          * unit_property
        
        If ids is given AND property already exists,
        then it is modified only on a subset of channels/units
        
        """
        if values is None:
            if key in self._properties:
                self._properties.pop(key)
            return

        size = self._main_ids.size
        values = np.asarray(values)
        if ids is None:
            assert values.shape[0] == size
            self._properties[key] = values
        else:
            if key not in self._properties:
                # create the property with nan or empty
                shape = (size,) + values.shape[1:]
                if values.dtype.kind in ('i', 'f', 'S', 'U'):
                    dtype = values.dtype
                else:
                    dtype = object
                empty_values = np.zeros(shape, dtype=dtype)
                if values.dtype.kind == 'f':
                    empty_values = empty_values * np.nan
                # ~ elif values.dtype.kind == 'i':
                # ~ # TODO find a way to put missing values
                self._properties[key] = empty_values

            indices = self.ids_to_indices(ids)
            self._properties[key][indices] = values

    def get_property(self, key, ids=None):
        values = self._properties.get(key, None)
        if ids is not None and values is not None:
            inds = self.ids_to_indices(ids)
            values = values[inds]
        return values

    def get_property_keys(self):
        return list(self._properties.keys())

    def copy_metadata(self, other, only_main=False, ids=None):
        """
        Copy annotations/properties/features to another extractor.
        
        If 'only main' is True, then only "main" annotations/properties/features one are copied.
        """

        if ids is None:
            inds = slice(None)
        else:
            inds = self.ids_to_indices(ids)

        if only_main:
            ann_keys = BaseExtractor._main_annotations
            prop_keys = BaseExtractor._main_properties
            # feat_keys = BaseExtractor._main_features
        else:
            ann_keys = self._annotations.keys()
            prop_keys = self._properties.keys()
            # TODO include features
            # feat_keys = ExtractorBase._features.keys()

        other._annotations = deepcopy({k: self._annotations[k] for k in ann_keys})
        for k in prop_keys:
            values = self._properties[k]
            if values is not None:
                other.set_property(k, values[inds])
        # TODO: copy features also

    def to_dict(self, include_annotations=False, include_properties=False, include_features=False,
                relative_to=None, folder_metadata=None):
        '''
        Make a nested serialized dictionary out of the extractor. The dictionary be used to re-initialize an
        extractor with load_extractor_from_dict(dump_dict)

        Parameters
        ----------
        include_annotations: bool
            If True, all annotations are added to the dict
        include_properties: bool
            If True, all properties are added to the dict
        include_features: bool
            If True, all features are added to the dict
        relative_to: str, Path, or None
            If not None, file_paths are serialized relative to this path

        Returns
        -------
        dump_dict: dict
            Serialized dictionary
        '''
        class_name = str(type(self)).replace("<class '", "").replace("'>", '')
        module = class_name.split('.')[0]
        imported_module = importlib.import_module(module)

        try:
            version = imported_module.__version__
        except AttributeError:
            version = 'unknown'
        
        
        dump_dict = {
            'class': class_name,
            'module': module,
            'kwargs': self._kwargs,
            'dumpable': self.is_dumpable,
            'version': version,
            'relative_paths': (relative_to is not None),
        }

        try:
            dump_dict['version'] = imported_module.__version__
        except AttributeError:
            dump_dict['version'] = 'unknown'

        if include_annotations:
            dump_dict['annotations'] = self._annotations
        else:
            # include only main annotations
            dump_dict['annotations'] = {k: self._annotations.get(k, None) for k in self._main_annotations}

        if include_properties:
            dump_dict['properties'] = self._properties
        else:
            # include only main properties
            dump_dict['properties'] = {k: self._properties.get(k, None) for k in self._main_properties}

        # TODO include features

        if relative_to is not None:
            relative_to = Path(relative_to).absolute()
            assert relative_to.is_dir(), "'relative_to' must be an existing directory"
            dump_dict = _make_paths_relative(dump_dict, relative_to)
        
        if folder_metadata is not None:
            if relative_to is not None:
                folder_metadata = Path(folder_metadata).absolute().relative_to(relative_to)
            dump_dict['folder_metadata'] = str(folder_metadata)
        
        return dump_dict

    @staticmethod
    def from_dict(d, base_folder=None):
        '''
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
        '''
        if d['relative_paths']:
            assert base_folder is not None, 'When  relative_paths=True, need to provide base_folder'
            d = _make_paths_absolute(d, base_folder)
        extractor = _load_extractor_from_dict(d)
        folder_metadata = d.get('folder_metadata', None)
        if folder_metadata is not None:
            folder_metadata = Path(folder_metadata)
            if d['relative_paths']:
                folder_metadata = base_folder / folder_metadata
            extractor.load_metadata_from_folder(folder_metadata)
        return extractor
    
    def load_metadata_from_folder(self, folder_metadata):
        # hack to load probe for recording
        folder_metadata = Path(folder_metadata)
        
        self._extra_metadata_from_folder(folder_metadata)

        # load properties
        prop_folder = folder_metadata / 'properties'
        for prop_file in prop_folder.iterdir():
            if prop_file.suffix == '.npy':
                values = np.load(prop_file, allow_pickle=True)
                key = prop_file.stem
                self.set_property(key, values)
    
    def save_metadata_to_folder(self, folder_metadata):
        self._extra_metadata_to_folder(folder_metadata)
        
        # save properties
        prop_folder = folder_metadata / 'properties'
        prop_folder.mkdir(parents=True, exist_ok=False)
        for key in self.get_property_keys():
            values = self.get_property(key)
            np.save(prop_folder / (key + '.npy'), values)
        

    def clone(self):
        """
        Clones an existing extractor into a new instance.
        """
        d = self.to_dict(include_annotations=True, include_properties=True, include_features=True)
        clone = BaseExtractor.from_dict(d)
        return clone

    def check_if_dumpable(self):
        return _check_if_dumpable(self.to_dict())

    @staticmethod
    def _get_file_path(file_path, extensions):
        '''
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

        Raises
        ------
        NotDumpableExtractorError
        '''
        ext = extensions[0]
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        folder_path = file_path.parent
        if Path(file_path).suffix == '':
            file_path = folder_path / (str(file_path) + ext)
        assert file_path.suffix in extensions, \
            "'file_path' should have one of the following extensions:" \
            " %s" % (', '.join(extensions))
        return file_path

    def dump(self, file_path, relative_to=None, folder_metadata=None):
        """
        Dumps extractor to json or pickle

        Parameters
        ----------
        file_path: str or Path
            The output file (either .json or .pkl/.pickle)
        relative_to: str, Path, or None
            If not None, file_paths are serialized relative to this path
        """
        if str(file_path).endswith('.json'):
            self.dump_to_json(file_path, relative_to=relative_to, folder_metadata=folder_metadata)
        elif str(file_path).endswith('.pkl') or str(file_path).endswith('.pickle'):
            self.dump_to_pickle(file_path, relative_to=relative_to, folder_metadata=folder_metadata)
        else:
            raise ValueError('Dump: file must .json or .pkl')

    def dump_to_json(self, file_path=None, relative_to=None, folder_metadata=None):
        '''
        Dump recording extractor to json file.
        The extractor can be re-loaded with load_extractor_from_json(json_file)

        Parameters
        ----------
        file_path: str
            Path of the json file
        relative_to: str, Path, or None
            If not None, file_paths are serialized relative to this path
        '''
        assert self.check_if_dumpable()
        dump_dict = self.to_dict(include_annotations=True,
                                 include_properties=False,
                                 include_features=False,
                                 relative_to=relative_to,
                                 folder_metadata=folder_metadata)
        file_path = self._get_file_path(file_path, ['.json'])
        file_path.write_text(
            json.dumps(check_json(dump_dict), indent=4),
            encoding='utf8'
        )

    def dump_to_pickle(self, file_path=None, include_properties=True, include_features=True,
                       relative_to=None, folder_metadata=None):
        '''
        Dump recording extractor to a pickle file.
        The extractor can be re-loaded with load_extractor_from_json(json_file)

        Parameters
        ----------
        file_path: str
            Path of the json file
        include_properties: bool
            If True, all properties are dumped
        include_features: bool
            If True, all features are dumped
        relative_to: str, Path, or None
            If not None, file_paths are serialized relative to this path
        '''
        assert self.check_if_dumpable()
        dump_dict = self.to_dict(include_annotations=True,
                                 include_properties=False,
                                 include_features=False,
                                 relative_to=relative_to, 
                                 folder_metadata=folder_metadata)
        file_path = self._get_file_path(file_path, ['.pkl', '.pickle'])

        file_path.write_bytes(pickle.dumps(dump_dict))

    @staticmethod
    def load(file_path, base_folder=None):
        """
        Load extractor from file path (.json or .pkl)

        Used both after:
          * dump(...) json or pickle file
          * save (...)  a folder which contain data  + json (or pickle) + metadata.
        """

        file_path = Path(file_path)
        if file_path.is_file():
            # standard case based on a file (json or pickle)
            if str(file_path).endswith('.json'):
                with open(str(file_path), 'r') as f:
                    d = json.load(f)
            elif str(file_path).endswith('.pkl') or str(file_path).endswith('.pickle'):
                with open(str(file_path), 'rb') as f:
                    d = pickle.load(f)
            else:
                raise ValueError(f'Impossible to load {file_path}')
            extractor = BaseExtractor.from_dict(d, base_folder=base_folder)
            return extractor

        elif file_path.is_dir():
            # case from a folder after a calling extractor.save(...)
            folder = file_path
            file = None
            for dump_ext in ('json', 'pkl', 'pickle'):
                f = folder / f'cached.{dump_ext}'
                if f.is_file():
                    file = f
            if file is None:
                raise ValueError(f'This folder is not a cached folder {file_path}')
            extractor = BaseExtractor.load(file, base_folder=folder)


            return extractor

        else:
            raise ValueError('bad boy')

    @staticmethod
    def load_from_folder(folder):
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
    
    def save(self, **kwargs):
        """
        route save_to_folder() or save_to_mem()
        """
        if kwargs.get('format', None) == 'memory':
            return self.save_to_memory(**kwargs)
        else:
            return self.save_to_folder(**kwargs)

    def save_to_memory(self, **kwargs):
        # used only by recording at the moment
        cached = self._save(**kwargs)
        self.copy_metadata(cached)
        return cached

    def save_to_folder(self, name=None, folder=None, dump_ext='json', verbose=True, **save_kwargs):
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
                name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                folder = cache_folder / name
                if verbose:
                    print(f'Use cache_folder={folder}')
            else:
                folder = cache_folder / name
                if not is_set_global_tmp_folder():
                    if verbose:
                        print(f'Use cache_folder={folder}')
        else:
            folder = Path(folder)
        assert not folder.exists(), f'folder {folder} already exists, choose enother name'
        folder.mkdir(parents=True, exist_ok=False)

        # dump provenance
        provenance_file = folder / f'provenance.{dump_ext}'
        if self.check_if_dumpable():
            self.dump(provenance_file)
        else:
            provenance_file.write_text(
                json.dumps({'warning': 'the provenace is not dumpable!!!'}),
                encoding='utf8'
            )


        # save data (done the subclass)
        cached = self._save(folder=folder, verbose=verbose, **save_kwargs)
        
        self.save_metadata_to_folder(folder)

        # copy properties/
        self.copy_metadata(cached)

        # dump
        cached.dump(folder / f'cached.{dump_ext}', relative_to=folder, folder_metadata=folder)

        return cached


def _make_paths_relative(d, relative):
    dcopy = deepcopy(d)
    if "kwargs" in dcopy.keys():
        relative_kwargs = _make_paths_relative(dcopy["kwargs"], relative)
        dcopy["kwargs"] = relative_kwargs
        return dcopy
    else:
        for k in d.keys():
            # in SI, all input paths have the "path" keyword
            if "path" in k:
                # paths can be str or list of str
                if isinstance(d[k], str):
                    d[k] = str(Path(d[k]).relative_to(relative))
                else:
                    assert isinstance(d[k], list), "Paths can be strings or lists in kwargs"
                    relative_paths = []
                    for path in d[k]:
                        relative_paths.append(str(Path(path).relative_to(relative)))
                    d[k] = relative_paths
        return d


def _make_paths_absolute(d, base):
    base = Path(base)
    dcopy = deepcopy(d)
    if "kwargs" in dcopy.keys():
        base_kwargs = _make_paths_absolute(dcopy["kwargs"], base)
        dcopy["kwargs"] = base_kwargs
        return dcopy
    else:
        for k in d.keys():
            # in SI, all input paths have the "path" keyword
            if "path" in k:
                # paths can be str or list of str
                if isinstance(d[k], str):
                    if not Path(d[k]).exists():
                        d[k] = str(base / d[k])
                else:
                    assert isinstance(d[k], list), "Paths can be strings or lists in kwargs"
                    absolute_paths = []
                    for path in d[k]:

                        if not Path(path).exists():
                            absolute_paths.append(str(base / path))
                    d[k] = absolute_paths
        return d


def _check_if_dumpable(d):
    kwargs = d['kwargs']
    if np.any([isinstance(v, dict) and 'dumpable' in v.keys() for (k, v) in kwargs.items()]):
        # check nested
        for k, v in kwargs.items():
            if 'dumpable' in v.keys():
                return _check_if_dumpable(v)
    else:
        return d['dumpable']


def is_dict_extractor(d):
    """
    Check if a dict describe an extractor.
    """
    if not isinstance(d, dict):
        return False
    is_extractor = ('module' in d) and ('class' in d) and ('version' in d) and ('annotations' in d)
    return is_extractor


def _load_extractor_from_dict(dic):
    cls = None
    class_name = None

    if 'kwargs' not in dic:
        raise Exception(f'This dict cannot be load into extractor {dic}')
    kwargs = deepcopy(dic['kwargs'])

    # handle nested
    for k, v in kwargs.items():

        if isinstance(v, dict) and is_dict_extractor(v):
            kwargs[k] = _load_extractor_from_dict(v)

            # handle list of extractors list
    for k, v in kwargs.items():
        if isinstance(v, list):
            if all(is_dict_extractor(e) for e in v):
                kwargs[k] = [_load_extractor_from_dict(e) for e in v]

    class_name = dic['class']
    cls = _get_class_from_string(class_name)

    assert cls is not None and class_name is not None, "Could not load spikeinterface class"
    if not _check_same_version(class_name, dic['version']):
        print('Versions are not the same. This might lead to errors. Use ', class_name.split('.')[0],
              'version', dic['version'])

    # instantiate extrator object
    extractor = cls(**kwargs)

    extractor._annotations.update(dic['annotations'])
    for k, v in dic['properties'].items():
        # print(k, v)
        extractor.set_property(k, v)
    # TODO features

    return extractor


def _get_class_from_string(class_string):
    class_name = class_string.split('.')[-1]
    module = '.'.join(class_string.split('.')[:-1])
    imported_module = importlib.import_module(module)

    try:
        imported_class = getattr(imported_module, class_name)
    except:
        imported_class = None

    return imported_class


def _check_same_version(class_string, version):
    module = class_string.split('.')[0]
    imported_module = importlib.import_module(module)

    try:
        return imported_module.__version__ == version
    except AttributeError:
        return 'unknown'


def load_extractor(file_or_folder_or_dict, base_folder=None):
    """
    Instantiate extractor from:
      * a dict
      * a json file
      * a pickle file
      * folder (after save)

    Parameters
    ----------
    file_or_folder_or_dict: dictionary or folder or file (json, pickle)
        

    Returns
    -------
    extractor: Recording or Sorting
        The loaded extractor object
    """
    if isinstance(file_or_folder_or_dict, dict):
        return BaseExtractor.from_dict(file_or_folder_or_dict, base_folder=base_folder)
    else:
        return BaseExtractor.load(file_or_folder_or_dict, base_folder=base_folder)


def load_extractor_from_dict(d, base_folder=None):
    print('Use load_extractor(..) instead')
    return BaseExtractor.from_dict(d, base_folder=base_folder)


def load_extractor_from_json(json_file, base_folder=None):
    print('Use load_extractor(..) instead')
    return BaseExtractor.load(json_file, base_folder=base_folder)


def load_extractor_from_pickle(pkl_file, base_folder=None):
    print('Use load_extractor(..) instead')
    return BaseExtractor.load(pkl_file, base_folder=base_folder)


class BaseSegment:
    def __init__(self):
        self._parent_extractor = None

    @property
    def parent_extractor(self):
        return self._parent_extractor()

    def set_parent_extractor(self, parent_extractor):
        self._parent_extractor = weakref.ref(parent_extractor)
