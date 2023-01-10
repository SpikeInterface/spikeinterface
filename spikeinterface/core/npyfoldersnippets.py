from pathlib import Path
import json

import numpy as np

from .base import _make_paths_absolute
from .npysnippetsextractor import NpySnippetsExtractor
from .core_tools import define_function_from_class


class NpyFolderSnippets(NpySnippetsExtractor):
    """
    NpyFolderSnippets is an internal format used in spikeinterface.
    It is a NpySnippetsExtractor + metadata contained in a folder.

    It is created with the function: `snippets.save(format='npy', folder='/myfolder')`

    Parameters
    ----------
    folder_path: str or Path

    Returns
    -------
    snippets: NpyFolderSnippets
        The snippets
    """
    extractor_name = 'NpyFolderSnippets'
    mode = 'folder'
    name = "npyfolder"

    def __init__(self,  folder_path):

        folder_path = Path(folder_path)

        with open(folder_path / 'npy.json', 'r') as f:
            d = json.load(f)

        if not d['class'].endswith('.NpySnippetsExtractor'):
            raise ValueError('This folder is not a binary spikeinterface folder')

        assert d['relative_paths']

        d = _make_paths_absolute(d, folder_path)

        NpySnippetsExtractor.__init__(self, **d['kwargs'])

        folder_metadata = folder_path
        self.load_metadata_from_folder(folder_metadata)

        self._kwargs = dict(folder_path=str(folder_path.absolute()))
        self._bin_kwargs = d['kwargs']


read_npy_snippets_folder = define_function_from_class(source_class=NpyFolderSnippets, name="read_npy_snippets_folder")
