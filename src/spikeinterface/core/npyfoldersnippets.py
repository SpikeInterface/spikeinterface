from pathlib import Path
import json

from copy import deepcopy

from probeinterface import read_probeinterface, write_probeinterface

from .npysnippetsextractor import NpySnippetsExtractor
from .core_tools import define_function_from_class, make_paths_absolute, load_properties_from_binary_folder, save_properties_to_binary_folder



class NpyFolderSnippets(NpySnippetsExtractor):
    """
    NpyFolderSnippets is an internal format used in spikeinterface.
    It is a NpySnippetsExtractor + metadata contained in a folder.

    It is created with the function: `snippets.save(format="npy", folder="/myfolder")`

    Parameters
    ----------
    folder_path : str or Path
        The path to the folder

    Returns
    -------
    snippets : NpyFolderSnippets
        The snippets
    """

    mode = "folder"
    name = "npyfolder"

    def __init__(self, folder_path):
        folder_path = Path(folder_path)

        with open(folder_path / "npy.json", "r") as f:
            d = json.load(f)

        if not d["class"].endswith(".NpySnippetsExtractor"):
            raise ValueError("This folder is not a binary spikeinterface folder")

        assert d["relative_paths"]

        d = make_paths_absolute(d, folder_path)

        NpySnippetsExtractor.__init__(self, **d["kwargs"])

        probe_file = folder_path / "probegroup.json"
        if probe_file.is_file():
            self._probegroup = read_probeinterface(probe_file)

        load_properties_from_binary_folder(folder_path / "properties", self)

        self._kwargs = dict(folder_path=str(Path(folder_path).absolute()))
        self._bin_kwargs = d["kwargs"]
    
    @staticmethod
    def write_snippets(snippets, folder, dtype=None):

        folder = Path(folder)

        if dtype is None:
            dtype = snippets.dtype

        file_paths = [folder / f"traces_cached_seg{i}.npy" for i in range(snippets.get_num_segments())]

        if dtype is None:
            dtype = snippets.get_dtype()

        # This is weird but for backward compatibility
        # maybe this can be removed
        NpySnippetsExtractor.write_snippets(snippets=snippets, file_paths=file_paths, dtype=dtype)
        cached = NpySnippetsExtractor(
            file_paths=file_paths,
            sampling_frequency=snippets.get_sampling_frequency(),
            channel_ids=snippets.get_channel_ids(),
            nbefore=snippets.nbefore,
            gain_to_uV=snippets.get_channel_gains(),
            offset_to_uV=snippets.get_channel_offsets(),
        )
        cached.dump(folder / "npy.json", relative_to=folder)

        save_properties_to_binary_folder(folder / "properties", snippets)

        if snippets.has_probe():
            probegroup = snippets.get_probegroup()
            write_probeinterface(folder / "probegroup.json", probegroup)


        cached = NpyFolderSnippets(folder_path=folder)
        # important backward compatibility : annoations are handled (sadly) only is this file
        # so we need to set then here (sad hack)
        cached._annotations = deepcopy({k: snippets._annotations[k] for k in snippets._annotations.keys()})
        cached.dump(folder / "si_folder.json", relative_to=folder)

        return cached


read_npy_snippets_folder = define_function_from_class(source_class=NpyFolderSnippets, name="read_npy_snippets_folder")
