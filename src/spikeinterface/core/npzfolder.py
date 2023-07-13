from pathlib import Path
import json

import numpy as np

from .base import _make_paths_absolute
from .npzsortingextractor import NpzSortingExtractor
from .core_tools import define_function_from_class


class NpzFolderSorting(NpzSortingExtractor):
    """
    NpzFolderSorting is an internal format used in spikeinterface.
    It is a NpzSortingExtractor + metadata contained in a folder.

    It is created with the function: `sorting.save(folder='/myfolder')`

    Parameters
    ----------
    folder_path: str or Path

    Returns
    -------
    sorting: NpzFolderSorting
        The sorting
    """

    extractor_name = "NpzFolder"
    mode = "folder"
    name = "npzfolder"

    def __init__(self, folder_path):
        folder_path = Path(folder_path)

        with open(folder_path / "npz.json", "r") as f:
            dictionary = json.load(f)

        if not dictionary["class"].endswith(".NpzSortingExtractor"):
            raise ValueError("This folder is not an npz spikeinterface folder")

        assert dictionary["relative_paths"]

        kwargs = dictionary["kwargs"]
        kwargs = _make_paths_absolute(kwargs, folder_path)

        NpzSortingExtractor.__init__(self, **kwargs)

        folder_metadata = folder_path
        self.load_metadata_from_folder(folder_metadata)

        self._kwargs = dict(folder_path=str(folder_path.absolute()))
        self._npz_kwargs = kwargs


read_npz_folder = define_function_from_class(source_class=NpzFolderSorting, name="read_npz_folder")
