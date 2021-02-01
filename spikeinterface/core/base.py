from typing import List, Union
from .mytypes import ChannelId, ChannelIndex, Order, SamplingFrequencyHz

from pathlib import Path
import importlib
from copy import deepcopy
import weakref
import json
import pickle
import datetime

import numpy as np

from .default import get_global_tmp_folder, is_set_global_tmp_folder

class BaseExtractor:
    """
    Base class for Recording/Sorting
    
    Handle serilization save/load to/from a folder.
    
    """
    
    # the replace the old key_properties
    # theses are annotations/properties/features that need to be
    # dumpc always (for instance locations, groups, is_fileterd...)
    _main_annotations = []
    _main_properties = []
    _main_features = []
    
    def __init__(self, main_ids):
        # store init kwargs for nested serialisation
        self._kwargs = {}
        
        # main_ids will either channel_ids or units_ids
        # it is used for properties and features
        #~ self._main_ids = np.array(main_ids, dtype=int)
        self._main_ids = np.array(main_ids)

        # dict at object level
        self._annotations = {}

        # properties is a dict of array
        # array length is :
        #  * number of channel for recording
        #  * number of units for sorting
        self._properties = {}
        
        # features dict of array (at spike level)
        self._features = {}

        # cache folder
        self._cache_folder = None

        
        self.is_dumpable = True        
        
        
    def get_num_segments(self):
        # This implemented in BaseRecording or baseSorting
        raise NotImplementedError

    def _check_segment_index(self, segment_index: Union[int, None]) -> int:
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
        Usefull to manipulate:
          * data
          * properties
          * features
        
        prefer_slice is an efficient option that try to make a slice object
        when channels are consecutive.
        
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
                    indices = slice(indices[0], indices[-1] +1)
        return indices
    
    def annotate(self, **new_nnotations):
        self._annotations.update(new_nnotations)
    
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
        Return a copy by dfault
        """
        v = self._annotations.get(key, None)
        if copy:
            v = deepcopy(v)
        return v
    
    def set_property(self, key, values, ids=None):
        """
        Set propery vector:
          * channel_property
          * unit_property
        
        if ids is given AND property already exists.
        Then is modify only a subset of channels/units
        
        """
        values = np.asarray(values)
        if ids is None:
            assert values.shape[0] == self._main_ids.size
            self._properties[key] = values
        else:
            assert key in self._properties, 'The key is not in properties'
            indices = self.ids_to_indices(ids)
            self._properties[key][indices] = values
    
    def get_property(self, key):
        return self._properties.get(key, None)
    
    def get_property_keys(self):
        return self._properties.keys()

    def copy_metadata(self, other, only_main=False):
        """
        Copy annotations/properties/features to other extractor
        
        If only main.
        Then only "main" one are copied
        """
        if only_main:
            other._annotations = deepcopy({self._annotations[k]
                            for k in ExtractorBase._main_annotations})
            other._properties = deepcopy({self._properties[k]
                            for k in ExtractorBase._main_properties})
            other._features = deepcopy({self._features[k]
                            for k in ExtractorBase._main_features})
        else:
            other._annotations = deepcopy(self._annotations)
            other._properties = deepcopy(self._properties)
            other._features = deepcopy(self._features)

    
    def set_cache_folder(self, folder):
        self._cache_folder = Path(folder)
    
    def to_dict(self, include_annotations=True, include_properties=True, include_features=True):
        '''
        Makes a nested serialized dictionary out of the extractor. The dictionary be used to re-initialize an
        extractor with spikeextractors.load_extractor_from_dict(dump_dict)

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
            }

        try:
            dump_dict['version'] = imported_module.__version__
        except AttributeError:
            dump_dict['version'] = 'unknown'
        
        if include_annotations:
            dump_dict['annotations'] = self._annotations
        
        if include_properties:
                dump_dict['properties'] = self._properties
        if include_features:
                dump_dict['features'] = self._features


        
        if include_annotations:
            dump_dict['annotations'] = self._annotations
        else:
            # include only main annotations
            dump_dict['annotations'] = {k:self._main_annotations[k] for k in self._main_annotations}
        
        if include_properties:
            dump_dict['properties'] = self._properties
        else:
            # include only main properties
            dump_dict['properties'] = {k:self._properties.get(k, None) for k in self._main_properties}
        
        if include_features:
            dump_dict['features'] = self._features
        else:
            # include only main features
            dump_dict['features'] = {k:self._features[k] for k in self._main_features}
        
        return dump_dict
    
    @staticmethod
    def from_dict(d):
        '''
        Instantiates extractor from dictionary

        Parameters
        ----------
        d: dictionary
            Python dictionary

        Returns
        -------
        extractor: RecordingExtractor or SortingExtractor
            The loaded extractor object
        '''        
        extractor = _load_extractor_from_dict(d)
        return extractor        

    def check_if_dumpable(self):
        return _check_if_dumpable(self.to_dict())

    def _get_file_path(self, file_path, extensions):
        '''
        Helper to be used by various dump_to_file utilities.

        Returns default file_path (if not specified), assures that target
        directory exists, adds correct file extension if none, and assures
        that provided file extension is one of the allowed.

        Parameters
        ----------
        file_path: str or None
        extensions: list or tuple
            First provided is used as an extension for the default file_path.
            All are tested against

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
    
    def dump(self, file_path):
        if str(file_path).endswith('.json'):
            self.dump_to_json(file_path)
        elif str(file_path).endswith('.pkl') or str(file_path).endswith('.pickle'):
            self.dump_to_pickle(file_path)
        else:
            raise ValueError('Dump: file must .json or .pkl')
    
    def dump_to_json(self, file_path=None):
        '''
        Dumps recording extractor to json file.
        The extractor can be re-loaded with spikeextractors.load_extractor_from_json(json_file)

        Parameters
        ----------
        file_path: str
            Path of the json file
        '''
        self.check_if_dumpable()
        dump_dict = self.to_dict( include_properties=False, include_features=False)
        file_path = self._get_file_path(file_path, ['.json'])
        
        file_path.write_text(
                json.dumps(_check_json(dump_dict), indent=4),
                encoding='utf8'
            )

    def dump_to_pickle(self, file_path=None, include_properties=True, include_features=True):
        '''
        Dumps recording extractor to a pickle file.
        The extractor can be re-loaded with spikeextractors.load_extractor_from_json(json_file)

        Parameters
        ----------
        file_path: str
            Path of the json file
        include_properties: bool
            If True, all properties are dumped
        include_features: bool
            If True, all features are dumped
        '''
        self.check_if_dumpable()
        dump_dict = self.to_dict( include_properties=include_properties, include_features=include_features)
        file_path = self._get_file_path(file_path, ['.pkl', '.pickle'])
        
        file_path.write_bytes(pickle.dumps(dump_dict))
    
    @staticmethod
    def load(file_path):
        """
        Used both after:
          * dump(...) json or pickle file
          * cache (...)  a folder which contain data  + json or pickle
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
                raise ValueError('Impossible to load {file_path}')
            extractor = BaseExtractor.from_dict(d)
            return extractor

        elif file_path.is_dir():
            # case from a folder after a calling extractor.cache(...)
            folder = file_path
            file = None
            for dump_ext in ('json', 'pkl', 'pickle'):
                f = folder / f'cached.{dump_ext}' 
                if f.is_file():
                    file = f
            if file is None:
                raise ValueError(f'This folder is not a cached folder {file_path}')
            return BaseExtractor.load(file)
            
        else:
            raise ValueError('bad boy')
    
    @staticmethod
    def load_from_cache(cache_folder, name):
        file_path = Path(cache_folder) / name
        return BaseExtractor.load(file_path)
    
    def _save_data(self, folder, **cache_kargs):
        # This implemented in BaseRecording or baseSorting
        raise NotImplementedError
    
    def cache(self, name=None, dump_ext='json', **cache_kargs):
        """
        Cache consist of:
          * compute the extractro by calling get_trace() in chunk
          * save data into file (memmap with BinaryRecordingExtractor)
          * dump to json/pickle the actual extractor for provenance
          * dump to json/pickle the cached extractor (memmap with BinaryRecordingExtractor)
        
        This replace the use of the old CacheRecordingExtractor and CacheSortingExtractor.
        """
        if name is None:
            raise ValueError('You must give a name for the cache')
        
        if self._cache_folder is None:
            cache_folder = get_global_tmp_folder()
            if not is_set_global_tmp_folder():
                print(f'Use cache_folder={cache_folder}')
        else:
            cache_folder = self._cache_folder
        
        folder = cache_folder / name
        assert not folder.exists(), f'folder {folder} already exists, choose other name'
        folder.mkdir(parents=True, exist_ok=False)
        
        # dump provenance
        provenance_file = folder / f'provenance.{dump_ext}' 
        if self.check_if_dumpable():
            self.dump(provenance_file)
        else:
            provenance_file.write_text(
                    json.dumps({'warning' : 'the provenace is not dumpable!!!'}),
                    encoding='utf8'
                )
        
        # save data (done the subclass)
        cached = self._save_data(folder, **cache_kargs)
        
        # copy properties/
        self.copy_metadata(cached)
        
        # dump
        cached.dump(folder / f'cached.{dump_ext}')
        
        return cached

    def allocate_array(self, engine, array_name, shape, dtype, ):
         raise NotImplemenetdError
        
    def allocate_array_from(self,  engine, array_name, existing_array):
        self.allocate_array(engine, array_name, existing_array.shape, existing_array.dtype)
        
    #~ get_sub_extractors_by_propertys


def _check_if_dumpable(d):
    kwargs = d['kwargs']
    if np.any([isinstance(v, dict) and 'dumpable' in v.keys() for (k, v) in kwargs.items()]):
        # check nested
        for k, v in kwargs.items():
            if 'dumpable' in v.keys():
                return _check_if_dumpable(v)
    else:
        return d['dumpable']

def _load_extractor_from_dict(dic):
    cls = None
    class_name = None
    
    kwargs = deepcopy(dic['kwargs'])

    if any(isinstance(v, dict) for v in kwargs.values()):
        # nested
        for k in kwargs.keys():
            if isinstance(kwargs[k], dict):
                if 'module' in kwargs[k].keys() and 'class' in kwargs[k].keys() and 'version' in kwargs[k].keys():
                    extractor = _load_extractor_from_dict(kwargs[k])
                    class_name = dic['class']
                    cls = _get_class_from_string(class_name)
                    kwargs[k] = extractor
                    break
    elif any(isinstance(v, list) and isinstance(v[0], dict) for v in kwargs.values()):
        # multi
        for k in kwargs.keys():
            if isinstance(kwargs[k], list) and isinstance(kwargs[k][0], dict):
                extractors = []
                for kw in kwargs[k]:
                    if 'module' in kw.keys() and 'class' in kw.keys() and 'version' in kw.keys():
                        extr = _load_extractor_from_dict(kw)
                        extractors.append(extr)
                #~ class_name = dic['class']
                #~ cls = _get_class_from_string(class_name)
                kwargs[k] = extractors
                break
    else:
        class_name = dic['class']
        cls = _get_class_from_string(class_name)

    assert cls is not None and class_name is not None, "Could not load spikeinterface class"
    if not _check_same_version(class_name, dic['version']):
        print('Versions are not the same. This might lead to errors. Use ', class_name.split('.')[0],
              'version', dic['version'])

    # instantiate extrator object
    extractor = cls(**kwargs)
    
    extractor._annotations.update(dic['annotations'])
    extractor._properties.update(dic['properties'])
    extractor._features.update(dic['features'])
    
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
    
    
def _check_json(d):
    # quick hack to ensure json writable
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _check_json(v)
        elif isinstance(v, Path):
            d[k] = str(v.absolute())
        elif isinstance(v, bool):
            d[k] = bool(v)
        elif isinstance(v, (np.int, np.int32, np.int64)):
            d[k] = int(v)
        elif isinstance(v, (np.float, np.float32, np.float64)):
            d[k] = float(v)
        elif isinstance(v, datetime.datetime):
            d[k] = v.isoformat()
        elif isinstance(v, (np.ndarray, list)):
            if len(v) > 0:
                if isinstance(v[0], dict):
                    # these must be extractors for multi extractors
                    d[k] = [_check_json(v_el) for v_el in v]
                else:
                    v_arr = np.array(v)
                    if len(v_arr.shape) == 1:
                        if 'int' in str(v_arr.dtype):
                            v_arr = [int(v_el) for v_el in v_arr]
                            d[k] = v_arr
                        elif 'float' in str(v_arr.dtype):
                            v_arr = [float(v_el) for v_el in v_arr]
                            d[k] = v_arr
                        elif isinstance(v_arr[0], str):
                            v_arr = [str(v_el) for v_el in v_arr]
                            d[k] = v_arr
                        else:
                            print(f'Skipping field {k}: only 1D arrays of int, float, or str types can be serialized')
                    elif len(v_arr.shape) == 2:
                        if 'int' in str(v_arr.dtype):
                            v_arr = [[int(v_el) for v_el in v_row] for v_row in v_arr]
                            d[k] = v_arr
                        elif 'float' in str(v_arr.dtype):
                            v_arr = [[float(v_el) for v_el in v_row] for v_row in v_arr]
                            d[k] = v_arr
                        else:
                            print(f'Skipping field {k}: only 2D arrays of int or float type can be serialized')
                    else:
                        print(f"Skipping field {k}: only 1D and 2D arrays can be serialized")
            else:
                d[k] = list(v)
    return d


def load_extractor(file_or_folder_or_dict):
    """
    Instantiates extractor from:
      * a dict
      * a json file
      * a pickle file
      * folder (after cahe)

    Parameters
    ----------
    file_or_folder_or_dict: dictionary or folder or file (json, pickle)
        

    Returns
    -------
    extractor: Recording or Sorting
        The loaded extractor object
    """
    if isinstance(file_or_folder_or_dict, dict):
        return BaseExtractor.from_dict(file_or_folder_or_dict)
    else:
        return BaseExtractor.load(file_or_folder_or_dict)

def load_extractor_from_dict(d):
    print('Use load_extractor(..) instead')
    return BaseExtractor.from_dict(d)


def load_extractor_from_json(json_file):
    print('Use load_extractor(..) instead')
    return BaseExtractor.load(json_file)


def load_extractor_from_pickle(pkl_file):
    print('Use load_extractor(..) instead')
    return BaseExtractor.load(pkl_file)


class BaseSegment:
    def __init__(self):
        self._parent_extractor = None
    
    @property
    def parent_extractor(self):
        return self._parent_extractor()
    
    def set_parent_extractor(self, parent_extractor):
        self._parent_extractor = weakref.ref(parent_extractor)
