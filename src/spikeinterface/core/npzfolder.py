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
            d = json.load(f)

        if not d["class"].endswith(".NpzSortingExtractor"):
            raise ValueError("This folder is not an npz spikeinterface folder")

        assert d["relative_paths"]

        d = _make_paths_absolute(d, folder_path)

        NpzSortingExtractor.__init__(self, **d["kwargs"])

        folder_metadata = folder_path
        self.load_metadata_from_folder(folder_metadata)

        self._kwargs = dict(folder_path=str(Path(folder_path).absolute()))
        self._npz_kwargs = d["kwargs"]


read_npz_folder = define_function_from_class(source_class=NpzFolderSorting, name="read_npz_folder")
