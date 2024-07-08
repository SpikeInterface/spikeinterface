from __future__ import annotations
from pathlib import Path
import json

import numpy as np

from .binaryrecordingextractor import BinaryRecordingExtractor
from .core_tools import define_function_from_class, make_paths_absolute


class BinaryFolderRecording(BinaryRecordingExtractor):
    """
    BinaryFolderRecording is an internal format used in spikeinterface.
    It is a BinaryRecordingExtractor + metadata contained in a folder.

    It is created with the function: `recording.save(format="binary", folder="/myfolder")`

    Parameters
    ----------
    folder_path : str or Path

    Returns
    -------
    recording : BinaryFolderRecording
        The recording
    """

    def __init__(self, folder_path):
        folder_path = Path(folder_path)

        with open(folder_path / "binary.json", "r") as f:
            d = json.load(f)

        if not d["class"].endswith(".BinaryRecordingExtractor"):
            raise ValueError("This folder is not a binary spikeinterface folder")

        assert d["relative_paths"]

        d = make_paths_absolute(d, folder_path)

        BinaryRecordingExtractor.__init__(self, **d["kwargs"])

        folder_metadata = folder_path
        self.load_metadata_from_folder(folder_metadata)

        self._kwargs = dict(folder_path=str(Path(folder_path).absolute()))
        self._bin_kwargs = d["kwargs"]
        if "num_channels" not in self._bin_kwargs:
            assert "num_chan" in self._bin_kwargs, "Cannot find num_channels or num_chan in binary.json"
            self._bin_kwargs["num_channels"] = self._bin_kwargs["num_chan"]

    def is_binary_compatible(self) -> bool:
        return True

    def get_binary_description(self):
        d = dict(
            file_paths=self._bin_kwargs["file_paths"],
            dtype=np.dtype(self._bin_kwargs["dtype"]),
            num_channels=self._bin_kwargs["num_channels"],
            time_axis=self._bin_kwargs["time_axis"],
            file_offset=self._bin_kwargs["file_offset"],
        )
        return d


read_binary_folder = define_function_from_class(source_class=BinaryFolderRecording, name="read_binary_folder")
