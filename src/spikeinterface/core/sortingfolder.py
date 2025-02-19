from __future__ import annotations
from pathlib import Path
import json

import numpy as np

from .basesorting import BaseSorting, SpikeVectorSortingSegment
from .npzsortingextractor import NpzSortingExtractor
from .core_tools import define_function_from_class, make_paths_absolute


class NumpyFolderSorting(BaseSorting):
    """
    NumpyFolderSorting is the new internal format used in spikeinterface (>=0.99.0) for caching sorting objects.

    It is a simple folder that contains:
      * a file "spike.npy" (numpy format) with all flatten spikes (using sorting.to_spike_vector())
      * a "numpysorting_info.json" containing sampling_frequency, unit_ids and num_segments
      * a metadata folder for units properties.

    It is created with the function: `sorting.save(folder="/myfolder", format="numpy_folder")`

    """

    mode = "folder"
    name = "NumpyFolder"

    def __init__(self, folder_path):
        folder_path = Path(folder_path)

        with open(folder_path / "numpysorting_info.json", "r") as f:
            info = json.load(f)

        sampling_frequency = info["sampling_frequency"]
        unit_ids = np.array(info["unit_ids"])
        num_segments = info["num_segments"]

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        self.spikes = np.load(folder_path / "spikes.npy")

        for segment_index in range(num_segments):
            self.add_sorting_segment(SpikeVectorSortingSegment(self.spikes, segment_index, unit_ids))

        # important trick : the cache is already spikes vector
        self._cached_spike_vector = self.spikes

        folder_metadata = folder_path
        self.load_metadata_from_folder(folder_metadata)

        self._kwargs = dict(folder_path=str(folder_path.absolute()))

    @staticmethod
    def write_sorting(sorting, save_path):
        # the folder can already exists but not contaning numpysorting_info.json
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        info_file = save_path / "numpysorting_info.json"
        if info_file.exists():
            raise ValueError("NumpyFolderSorting.write_sorting the folder already contains numpysorting_info.json")
        d = {
            "sampling_frequency": float(sorting.get_sampling_frequency()),
            "unit_ids": sorting.unit_ids.tolist(),
            "num_segments": sorting.get_num_segments(),
        }
        info_file.write_text(json.dumps(d), encoding="utf8")
        np.save(save_path / "spikes.npy", sorting.to_spike_vector())


class NpzFolderSorting(NpzSortingExtractor):
    """
    NpzFolderSorting is the old internal format used in spikeinterface (<=0.98.0)

    This a folder that contains:

      * "sorting_cached.npz" file in the NpzSortingExtractor format
      * "npz.json" which the json description of NpzSortingExtractor
      * a metadata folder for units properties.

    It is created with the function: `sorting.save(folder="/myfolder", format="npz_folder")`

    Parameters
    ----------
    folder_path : str or Path

    Returns
    -------
    sorting : NpzFolderSorting
        The sorting
    """

    mode = "folder"
    name = "npzfolder"

    def __init__(self, folder_path):
        folder_path = Path(folder_path)

        with open(folder_path / "npz.json", "r") as f:
            d = json.load(f)

        if not d["class"].endswith(".NpzSortingExtractor"):
            raise ValueError("This folder is not an npz spikeinterface folder")

        assert d["relative_paths"]

        d = make_paths_absolute(d, folder_path)

        NpzSortingExtractor.__init__(self, **d["kwargs"])

        folder_metadata = folder_path
        self.load_metadata_from_folder(folder_metadata)

        self._kwargs = dict(folder_path=str(folder_path.absolute()))
        self._npz_kwargs = d["kwargs"]

    @staticmethod
    def write_sorting(sorting, save_path):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        npz_file = save_path / "sorting_cached.npz"
        if npz_file.exists():
            raise ValueError("NpzFolderSorting.write_sorting the folder already contains sorting_cached.npz")
        NpzSortingExtractor.write_sorting(sorting, npz_file)
        cached = NpzSortingExtractor(npz_file)
        cached.dump(save_path / "npz.json", relative_to=save_path)


read_numpy_sorting_folder = define_function_from_class(
    source_class=NumpyFolderSorting, name="read_numpy_sorting_folder"
)
read_npz_folder = define_function_from_class(source_class=NpzFolderSorting, name="read_npz_folder")
