from pathlib import Path
import importlib

from core_tools import get_global_temp_folder, is_set_global_temp_folder

class Base:
    """
    Base class for Recording/Sorting
    
    Handle serilization save/load to/from a folder.
    
    """
    
    def __init__(self):
        # store init kwargs for nested seriolisation
        self._kwargs = {}

        # properties is a dict of array
        # array length is :
        #  * number of channel for recording
        #  * number of units for sorting
        self.properties = {}
        
        # features
        self.features = {}
        
        # dump folder
        self._dump_folder = None
        
        # _name is for dumping.
        # name need to be unique in the _dump_folder
        self._name = None
    
    def set_dump_folder(self, folder):
        self._dump_folder = Path(folder)
    
    def set_name(self, name):
        self._name = name


    def make_serialized_dict(self):
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

        if self.is_dumpable:
            dump_dict = {'class': class_name, 'module': module, 'kwargs': self._kwargs,
                         'key_properties': self._key_properties, 'version': version,
                         'dumpable': True}
        else:
            dump_dict = {'class': class_name, 'module': module, 'kwargs': {}, 'key_properties': self._key_properties,
                         'version': imported_module.__version__, 'dumpable': False}
        return dump_dict
        
    def to_dict(self):
        return self.make_serialized_dict()
    
    dump_to_dict = to_dict
    
    def dump(self, name=None, engine='', **engine_kwargs):
        if name is not None:
            self.set_name(name)
        if self._name is None:
            raise ValueError('Please provide a name')
        
        if self._dump_folder is None:
            folder = get_global_temp_folder()
            if not is_set_global_temp_folder():
                print(f'temporary folder is {folder}, Use set_global_temp_folder() or extractor.set_dump_folder()')
        else:
            folder = self._dump_folder
        
        if engine == 'memmap':
            self._dump_to_memmap(engine_kwargs)
        elif engine == 'zarr':
            self._dump_to_memmap(engine_kwargs)
        elif engine == 'pickle':
            self._dump_to_pickle(engine_kwargs)
        else:
            raise ValueError('Bad boy')
    
    @staticmethod
    def load(folder, name):
        raise NotImplemenetdError
        return 
        
    
    def cache(self, **kargs):
        # This replace the old CacheRecordingExtractor and CacheSortingExtractor
        # it cache the extractor to a folder and load it immediatly!!
        self.dump(engine, **kargs)
        cached = SIBase.load(self._dump_folder / self._name)
        return cached
    
    def _dump_to_memmap(self):
        raise NotImplemenetdError
    
    def _dump_to_zarr(self):
        raise NotImplemenetdError
    
    def _dump_to_pickle(self):
        raise NotImplemenetdError
    
    def allocate_array(self, engine, array_name, shape, dtype, ):
         raise NotImplemenetdError
        
    def allocate_array_from(self,  engine, array_name, existing_array):
        self.allocate_array(engine, array_name, existing_array.shape, existing_array.dtype)

