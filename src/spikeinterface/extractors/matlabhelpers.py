from pathlib import Path
from collections import deque

import numpy as np

try:
    import h5py
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

try:
    from scipy.io.matlab import loadmat, savemat
    HAVE_LOADMAT = True
except ImportError:
    HAVE_LOADMAT = False

try:
    import hdf5storage
    HAVE_HDF5STORAGE = True
except ImportError:
    HAVE_HDF5STORAGE = False

HAVE_MAT = HAVE_H5PY & HAVE_LOADMAT


class MatlabHelper:
    extractor_name = "MATSortingExtractor"
    installed = HAVE_MAT  # check at class level if installed or not
    mode = "file"
    installation_mesg = "To use the MATSortingExtractor install h5py and scipy: " \
                        "\n\n pip install h5py scipy\n\n"  # error message when not installed

    def __init__(self, file_path):
        assert HAVE_MAT, self.installation_mesg

        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        if not isinstance(file_path, Path):
            raise TypeError(f"Expected a str or Path file_path but got '{type(file_path).__name__}'")

        file_path = file_path.resolve()  # get absolute path to this file
        if not file_path.is_file():
            raise ValueError(f"Specified file path '{file_path}' is not a file.")

        self._kwargs = {"file_path": str(file_path.absolute())}

        try:  # load old-style (up to 7.2) .mat file
            self._data = loadmat(file_path, matlab_compatible=True)
            self._old_style_mat = True
        except NameError:  # loadmat not defined
            raise ImportError("Old-style .mat file given, but `loadmat` is not defined.")
        except NotImplementedError:  # new style .mat file
            try:
                self._data = h5py.File(file_path, "r+")
                self._old_style_mat = False
            except NameError:
                raise ImportError("Version 7.2 .mat file given, but you don't have h5py installed.")

    def __del__(self):
        if hasattr(self, '_old_style_mat') and not self._old_style_mat:
            self._data.close()

    def _getfield(self, fieldname: str):
        def _drill(d: dict, keys: deque):
            if len(keys) == 1:
                return d[keys.popleft()]
            else:
                return _drill(d[keys.popleft()], keys)

        if self._old_style_mat:
            return _drill(self._data, deque(fieldname.split("/")))
        else:
            return self._data[fieldname][()]

    @classmethod
    def write_dict_to_mat(cls, mat_file_path, dict_to_write, version='7.3'):  # field must be a dict
        assert HAVE_HDF5STORAGE, "To use the MATSortingExtractor write_dict_to_mat function install hdf5storage: " \
                                 "\n\n pip install hdf5storage\n\n"
        if version == '7.3':
            hdf5storage.write(dict_to_write, '/', mat_file_path, matlab_compatible=True, options='w')
        elif version < '7.3' and version > '4':
            savemat(mat_file_path, dict_to_write)
