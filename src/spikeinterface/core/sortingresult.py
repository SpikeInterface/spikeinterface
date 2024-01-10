from __future__ import annotations
from typing import Literal, Optional

from pathlib import Path
import os
import json
import pickle
import weakref
import shutil

import numpy as np

import probeinterface

from .baserecording import BaseRecording
from .basesorting import BaseSorting

from .base import load_extractor
from .recording_tools import check_probe_do_not_overlap, get_rec_attributes
from .core_tools import check_json
from .numpyextractors import SharedMemorySorting
from .sparsity import ChannelSparsity
from .sortingfolder import NumpyFolderSorting


# TODO
#  * make info.json that contain some version info of spikeinterface
#  * same for zarr
#  * sample spikes and propagate in compute with option



# high level function
def start_sorting_result(sorting, recording, format="memory", folder=None, 
    sparse=True, sparsity=None,
    # **kwargs
    ):
    """
    Create a SortingResult by pairing a Sorting and the corresponding Recording.

    This object will handle a list of ResultExtension for all the post processing steps like: waveforms,
    templates, unit locations, spike locations, quality mertics ...

    This object will be also use used for ploting purpose.


    Parameters
    ----------
    sorting: Sorting
        The sorting object
    recording: Recording
        The recording object
    folder: str or Path or None, default: None
        The folder where waveforms are cached
    format: "memory | "binary_folder" | "zarr", default: "memory"
        The mode to store waveforms. If "folder", waveforms are stored on disk in the specified folder.
        The "folder" argument must be specified in case of mode "folder".
        If "memory" is used, the waveforms are stored in RAM. Use this option carefully!
    sparse: bool, default: True
        If True, then a sparsity mask is computed usingthe `precompute_sparsity()` function is run using
        a few spikes to get an estimate of dense templates to create a ChannelSparsity object.
        Then, the sparsity will be propagated to all ResultExtention that handle sparsity (like wavforms, pca, ...)
    sparsity: ChannelSparsity or None, default: None
        The sparsity used to compute waveforms. If this is given, `sparse` is ignored. Default None.
    sparsity_temp_folder: str or Path or None, default: None
        If sparse is True, this is the temporary folder where the dense waveforms are temporarily saved.
        If None, dense waveforms are extracted in memory in batches (which can be controlled by the `unit_batch_size`
        parameter. With a large number of units (e.g., > 400), it is advisable to use a temporary folder.
    num_spikes_for_sparsity: int, default: 100
        The number of spikes to use to estimate sparsity (if sparse=True).
    unit_batch_size: int, default: 200
        The number of units to process at once when extracting dense waveforms (if sparse=True and sparsity_temp_folder
        is None).

    sparsity kwargs:
    {}


    job kwargs:
    {}


    Returns
    -------
    sorting_result: SortingResult
        The SortingResult object

    Examples
    --------
    >>> import spikeinterface as si

    >>> # Extract dense waveforms and save to disk with binary_folder format.
    >>> sortres = si.start_sorting_result(sorting, recording, format="binary_folder", folder="/path/to_my/result")

    """


    # handle sparsity
    if sparsity is not None:
        assert isinstance(sparsity, ChannelSparsity), "'sparsity' must be a ChannelSparsity object"
        unit_id_to_channel_ids = sparsity.unit_id_to_channel_ids
        assert all(u in sorting.unit_ids for u in unit_id_to_channel_ids), "Invalid unit ids in sparsity"
        for channels in unit_id_to_channel_ids.values():
            assert all(ch in recording.channel_ids for ch in channels), "Invalid channel ids in sparsity"
    elif sparse:
        # TODO
        # raise NotImplementedError()
        sparsity = None
        # estimate_kwargs, job_kwargs = split_job_kwargs(kwargs)
        # sparsity = precompute_sparsity(
        #     recording,
        #     sorting,
        #     ms_before=ms_before,
        #     ms_after=ms_after,
        #     num_spikes_for_sparsity=num_spikes_for_sparsity,
        #     unit_batch_size=unit_batch_size,
        #     temp_folder=sparsity_temp_folder,
        #     allow_unfiltered=allow_unfiltered,
        #     **estimate_kwargs,
        #     **job_kwargs,
        # )
    else:
        sparsity = None

    sorting_result = SortingResult.create(
        sorting, recording, format=format, folder=folder, sparsity=sparsity)

    return sorting_result

def load_sorting_result(folder, load_extensions=True, format="auto"):
    """
    Load a SortingResult object from disk.

    Parameters
    ----------
    folder : str or Path
        The folder / zarr folder where the waveform extractor is stored
    load_extensions : bool, default: True
        Load all extensions or not.
    format: "auto" | "binary_folder" | "zarr"
        The format of the folder.

    Returns
    -------
    sorting_result: SortingResult
        The loaded SortingResult

    """

    return SortingResult.load(folder, load_extensions=load_extensions, format=format)


class SortingResult:
    """
    Class to make a pair of Recording-Sorting which will be used used for all post postprocessing,
    visualization and quality metric computation.

    This internaly maintain a list of computed ResultExtention (waveform, pca, unit position, spike poisition, ...).

    This can live in memory and/or can be be persistent to disk in 2 internal formats (folder/json/npz or zarr).

    This handle unit sparsity that can be propagated to ResultExtention.

    This handle spike sampling that can be propagated to ResultExtention : work only on a subset of spikes.

    This internally save a copy of the Sorting and extract main recording attributes (without traces) so
    the SortingResult object can be reload even if references to the original sorting and/or to the original recording
    are lost.
    """
    def __init__(self, sorting=None, recording=None, rec_attributes=None, format=None, sparsity=None):
        # very fast init because checks are done in load and create
        self.sorting = sorting
        # self.recorsding will be a property
        self._recording = recording
        self.rec_attributes = rec_attributes
        self.format = format
        self.sparsity = sparsity

        # extensions are not loaded at init
        self.extensions = dict()

    ## create and load zone

    @classmethod
    def create(cls,
            sorting: BaseSorting,
            recording: BaseRecording,
            format: Literal["memory", "binary_folder", "zarr", ] = "memory",
            folder=None,
            sparsity=None,
        ):
        # some checks
        assert sorting.sampling_frequency == recording.sampling_frequency
        # check that multiple probes are non-overlapping
        all_probes = recording.get_probegroup().probes
        check_probe_do_not_overlap(all_probes)

        if format == "memory":
            rec_attributes = get_rec_attributes(recording)
            rec_attributes["probegroup"] = recording.get_probegroup()
            # a copy of sorting is created directly in shared memory format to avoid further duplication of spikes.
            sorting_copy = SharedMemorySorting.from_sorting(sorting)
            sortres = SortingResult(sorting=sorting_copy, recording=recording, rec_attributes=rec_attributes, format=format, sparsity=sparsity)
        elif format == "binary_folder":
            cls.create_binary_folder(folder, sorting, recording, sparsity, rec_attributes=None)
            sortres = cls.load_from_binary_folder(folder, recording=recording)
        elif format == "zarr":
            cls.create_zarr(folder, sorting, recording, sparsity, rec_attributes=None)
            sortres = cls.load_from_zarr(folder, recording=recording)
        else:
            raise ValueError("SortingResult.create: wrong format")

        return sortres

    @classmethod
    def load(cls, folder, recording=None, load_extensions=True, format="auto"):
        """
        Load folder or zarr.
        The recording can be given if the recording location has changed.
        Otherwise the recording is loaded when possible.
        """
        folder = Path(folder)
        assert folder.is_dir(), "Waveform folder does not exists"
        if format == "auto":
            # make better assumption and check for auto guess format
            if folder.suffix == ".zarr":
                format = "zarr"
            else:
                format = "binary_folder"
        
        if format == "binary_folder":
            sortres = SortingResult.load_from_binary_folder(folder, recording=recording)
        elif format == "zarr":
            sortres = SortingResult.load_from_zarr(folder, recording=recording)
            
        
        sortres.folder = folder

        if load_extensions:
            sortres.load_all_saved_extension()

        return sortres

    @classmethod
    def create_binary_folder(cls, folder, sorting, recording, sparsity, rec_attributes):
        # used by create and save

        folder = Path(folder)
        if folder.is_dir():
            raise ValueError(f"Folder already exists {folder}")
        folder.mkdir(parents=True)

        # save a copy of the sorting
        NumpyFolderSorting.write_sorting(sorting, folder / "sorting")

        # save recording and sorting provenance
        if recording.check_serializability("json"):
            recording.dump(folder / "recording.json", relative_to=folder)
        elif recording.check_serializability("pickle"):
            recording.dump(folder / "recording.pickle", relative_to=folder)

        if sorting.check_serializability("json"):
            sorting.dump(folder / "sorting_provenance.json", relative_to=folder)
        elif sorting.check_serializability("pickle"):
            sorting.dump(folder / "sorting_provenance.pickle", relative_to=folder)

        # dump recording attributes
        probegroup = None
        rec_attributes_file = folder / "recording_info" / "recording_attributes.json"
        rec_attributes_file.parent.mkdir()
        if rec_attributes is None:
            assert recording is not None
            rec_attributes = get_rec_attributes(recording)
            rec_attributes_file.write_text(json.dumps(check_json(rec_attributes), indent=4), encoding="utf8")
            probegroup = recording.get_probegroup()
        else:
            rec_attributes_copy = rec_attributes.copy()
            probegroup = rec_attributes_copy.pop("probegroup")
            rec_attributes_file.write_text(json.dumps(check_json(rec_attributes_copy), indent=4), encoding="utf8")

        if probegroup is not None:
            probegroup_file = folder / "recording_info" / "probegroup.json"
            probeinterface.write_probeinterface(probegroup_file, probegroup)

        if sparsity is not None:
            with open(folder / "sparsity.json", mode="w") as f:
                json.dump(check_json(sparsity.to_dict()), f)

    @classmethod
    def load_from_binary_folder(cls, folder, recording=None):
        folder = Path(folder)
        assert folder.is_dir(), f"This folder does not exists {folder}"

        # load internal sorting copy and make it sharedmem
        sorting = SharedMemorySorting.from_sorting(NumpyFolderSorting(folder / "sorting"))
        
        # load recording if possible
        if recording is None:
            # try to load the recording if not provided
            for type in ("json", "pickle"):
                filename = folder / f"recording.{type}"
                if filename.exists():
                    try:
                        recording = load_extractor(filename, base_folder=folder)
                        break
                    except:
                        recording = None
        else:
            # TODO maybe maybe not??? : do we need to check  attributes match internal rec_attributes
            # Note this will make the loading too slow
            pass

        # recording attributes
        rec_attributes_file = folder / "recording_info" / "recording_attributes.json"
        if not rec_attributes_file.exists():
            raise ValueError("This folder is not a SortingResult folder")
        with open(rec_attributes_file, "r") as f:
            rec_attributes = json.load(f)
        # the probe is handle ouside the main json
        probegroup_file = folder / "recording_info" / "probegroup.json"
        print(probegroup_file, probegroup_file.is_file())
        if probegroup_file.is_file():
            rec_attributes["probegroup"] = probeinterface.read_probeinterface(probegroup_file)
        else:
            rec_attributes["probegroup"] = None
        
        # sparsity
        sparsity_file = folder / "sparsity.json"
        if sparsity_file.is_file():
            with open(sparsity_file, mode="r") as f:
                sparsity = ChannelSparsity.from_dict(json.load(f))
        else:
            sparsity = None

        sortres = SortingResult(
            sorting=sorting,
            recording=recording,
            rec_attributes=rec_attributes,
            format="binary_folder",
            sparsity=sparsity)

        return sortres

    def _get_zarr_root(self, mode="r+"):
        import zarr
        zarr_root = zarr.open(self.folder, mode=mode)
        return zarr_root

    @classmethod
    def create_zarr(cls, folder, sorting, recording, sparsity, rec_attributes):
        raise NotImplementedError

    @classmethod
    def load_from_zarr(cls, folder, recording=None):
        raise NotImplementedError


    def save_as(
        self, folder=None, format="binary_folder",
    ) -> "SortingResult":
        """
        Save SortingResult object into another format.
        Uselfull for memory to zarr or memory to binray.
        
        Note that the recording provenance or sorting provenance can be lost.

        Mainly propagate the copied sorting and recording property.

        Parameters
        ----------
        folder : str or Path
            The output waveform folder
        format : "binary_folder" | "zarr", default: "binary_folder"
            The backend to use for saving the waveforms
        """

        if self.has_recording():
            recording = self.recording
        else:
            recording = None
        
        # Note that the sorting is a copy we need to go back to the orginal sorting (if available)
        sorting_provenance = self.get_sorting_provenance()
        if sorting_provenance is None:
            # if the original sorting objetc is not available anymore (kilosort folder deleted, ....), take the copy
            sorting_provenance = self.sorting

        if format == "memory":
            # This make a copy of actual SortingResult
            # TODO
            raise NotImplementedError
        elif format == "binary_folder":
            # create  a new folder
            SortingResult.create_binary_folder(folder, sorting_provenance, recording, self.sparsity, self.rec_attributes)
            new_sortres = SortingResult.load_from_binary_folder(folder)
            new_sortres.folder = folder

        elif format == "zarr":
            # TODO
            raise NotImplementedError
        else:
            raise ValueError("SortingResult.save: wrong format")

        # make a copy of extensions
        for extension_name, extension in self.extensions.items():
            new_sortres.extensions[extension_name] = extension.copy(new_sortres)

        return new_sortres


    def is_read_only(self) -> bool:
        if self.format == "memory":
            return False
        return not os.access(self.folder, os.W_OK)


    ## map attribute and property zone

    @property
    def recording(self) -> BaseRecording:
        if not self.has_recording():
            raise ValueError("SortingResult could not load the recording")
        return self._recording

    @property
    def channel_ids(self) -> np.ndarray:
        return np.array(self.rec_attributes["channel_ids"])

    @property
    def sampling_frequency(self) -> float:
        return self.sorting.get_sampling_frequency()

    @property
    def unit_ids(self) -> np.ndarray:
        return self.sorting.unit_ids

    def has_recording(self) -> bool:
        return self._recording is not None

    def is_sparse(self) -> bool:
        return self.sparsity is not None
    
    def get_sorting_provenance(self):
        """
        Get the original sorting if possible otherwise return None
        """
        if self.format == "memory":
            # the orginal sorting provenance is not keps in that case
            sorting_provenance =  None

        elif self.format == "binary_folder":
            for type in ("json", "pickle"):
                filename = self.folder / f"sorting_provenance.{type}"
                if filename.exists():
                    try:
                        sorting_provenance = load_extractor(filename, base_folder=self.folder)
                        break
                    except:
                        sorting_provenance = None

        elif self.format == "zarr":
            # TODO
            raise NotImplementedError

        return sorting_provenance

    # def is_read_only(self) -> bool:
    #     return self._is_read_only

    def get_num_samples(self, segment_index: Optional[int] = None) -> int:
        # we use self.sorting to check segment_index
        segment_index = self.sorting._check_segment_index(segment_index)
        return self.rec_attributes["num_samples"][segment_index]

    def get_total_samples(self) -> int:
        s = 0
        for segment_index in range(self.get_num_segments()):
            s += self.get_num_samples(segment_index)
        return s

    def get_total_duration(self) -> float:
        duration = self.get_total_samples() / self.sampling_frequency
        return duration

    def get_num_channels(self) -> int:
        return self.rec_attributes["num_channels"]

    def get_num_segments(self) -> int:
        return self.sorting.get_num_segments()

    def get_probegroup(self):
        return self.rec_attributes["probegroup"]

    def get_probe(self):
        probegroup = self.get_probegroup()
        assert len(probegroup.probes) == 1, "There are several probes. Use `get_probegroup()`"
        return probegroup.probes[0]

    def get_channel_locations(self) -> np.ndarray:
        # important note : contrary to recording
        # this give all channel locations, so no kwargs like channel_ids and axes
        all_probes = self.get_probegroup().probes
        all_positions = np.vstack([probe.contact_positions for probe in all_probes])
        return all_positions

    def channel_ids_to_indices(self, channel_ids) -> np.ndarray:
        all_channel_ids = self.rec_attributes["channel_ids"]
        indices = np.array([all_channel_ids.index(id) for id in channel_ids], dtype=int)
        return indices

    def __repr__(self) -> str:
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nchan = self.get_num_channels()
        nunits = self.sorting.get_num_units()
        txt = f"{clsname}: {nchan} channels - {nunits} units - {nseg} segments - {self.format}"
        if self.is_sparse():
            txt += " - sparse"
        return txt

    ## extensions zone
    def compute(self, extension_name, **params):
        """
        Compute one extension

        Parameters
        ----------
        extension_name

        **params

        Returns
        -------
        sorting_result: SortingResult
            The SortingResult object

        Examples
        --------

        >>> extension = sortres.compute("unit_location", **some_params)
        >>> unit_location = extension.get_data()
        
        """
        # TODO check extension dependency

        extension_class = get_extension_class(extension_name)
        extension_instance = extension_class(self)
        extension_instance.set_params(**params)
        extension_instance.run()
        
        self.extensions[extension_name] = extension_instance

        return extension_instance

    def get_saved_extension_names(self):
        """
        Get extension saved in folder or zarr that can be loaded.
        """
        assert self.format != "memory"
        global _possible_extensions

        saved_extension_names = []
        for extension_class in _possible_extensions:
            extension_name = extension_class.extension_name
            if self.format == "binary_folder":
                is_saved = (self.folder / extension_name).is_dir() and (self.folder / extension_name / "params.json").is_file()
            elif self.format == "zarr":
                zarr_root = self._get_zarr_root(mode="r")
                is_saved = extension_name in zarr_root.keys() and "params" in zarr_root[extension_name].attrs.keys()
            if is_saved:
                saved_extension_names.append(extension_class.extension_name)
            return saved_extension_names

    def get_extension(self, extension_name: str):
        """
        Get a ResultExtension.
        If not loaded then load it before.

        
        """
        if extension_name in self.extensions:
            return self.extensions[extension_name]

        if self.has_extension(extension_name):
            self.load_extension(extension_name)
            return self.extensions[extension_name]
        
        return None

    def load_extension(self, extension_name: str):
        """
        Load an extensionet from folder or zarr into the `ResultSorting.extensions` dict.

        Parameters
        ----------
        extension_name: str
            The extension name.

        Returns
        -------
        ext_instanace:
            The loaded instance of the extension

        """
        assert self.format != "memory"

        extension_class = get_extension_class(extension_name)

        extension_instance = extension_class(self)
        extension_instance.load_prams()
        extension_instance.load_data()

        return extension_instance

    def load_all_saved_extension(self):
        """
        Load all saved extension in memory.
        """
        for extension_name in self.get_saved_extension_names():
            self.load_extension(extension_name)

    def delete_extension(self, extension_name) -> None:
        """
        Delete the extension from the dict and also in the persistent zarr or folder.
        """
        pass

    def get_loaded_extension_names(self):
        """
        Return the loaded or already computed extensions names.
        """
        return list(self.extensions.keys())
    
    def has_extension(self, extension_name: str) -> bool:
        """
        Check if the extension exists in memory (dict) or in the folder or in zarr.

        If force_load=True (the default) then the extension is automatically loaded if available.
        """
        if extension_name in self.extensions:
            return True
        elif extension_name in self.get_saved_extension_names():
            return True
        else:
            return False


global _possible_extensions
_possible_extensions = []

def register_result_extension(extension_class):
    """
    This maintains a list of possible extensions that are available.
    It depends on the imported submodules (e.g. for postprocessing module).

    For instance with:
    import spikeinterface as si
    only one extension will be available
    but with 
    import spikeinterface.postprocessing
    more extensions will be available
    """
    assert issubclass(extension_class, ResultExtension)
    assert extension_class.extension_name is not None, "extension_name must not be None"
    global _possible_extensions

    already_registered = any(extension_class is ext for ext in _possible_extensions)
    if not already_registered:
        assert all(
            extension_class.extension_name != ext.extension_name for ext in _possible_extensions
        ), "Extension name already exists"

        _possible_extensions.append(extension_class)


def get_extension_class(extension_name: str):
    """
    Get extension class from name and check if registered.

    Parameters
    ----------
    extension_name: str
        The extension name.

    Returns
    -------
    ext_class:
        The class of the extension.
    """
    global _possible_extensions
    extensions_dict = {ext.extension_name: ext for ext in _possible_extensions}
    assert extension_name in extensions_dict, "Extension is not registered, please import related module before"
    ext_class = extensions_dict[extension_name]
    return ext_class


class ResultExtension:
    """
    This the base class to extend the SortingResult.
    It can handle persistency to disk any computations related

    For instance:
      * waveforms
      * principal components
      * spike amplitudes
      * quality metrics

    Possible extension can be register on the fly at import time with register_result_extension() mechanism.
    It also enables any custum computation on top on SortingResult to be implemented by the user.

    An extension needs to inherit from this class and implement some abstract methods:
      * _set_params
      * _run
      * 

    The subclass must also set an `extension_name` class attribute which is not None by default.

    The subclass must also hanle an attribute `__data` which is a dict contain the results after the `run()`.
    """    
    extension_name = None

    def __init__(self, sorting_result):
        self._sorting_result = weakref.ref(sorting_result)

        self._params = None
        self._data = dict()

    @property
    def sorting_result(self):
        # Important : to avoid the SortingResult referencing a ResultExtension
        # and ResultExtension referencing a SortingResult we need a weakref.
        # Otherwise the garbage collector is not working properly.
        # and so the SortingResult + its recording are still alive even after deleting explicitly
        # the SortingResult which makes it impossible to delete the folder when using memmap.
        sorting_result = self._sorting_result()
        if sorting_result is None:
            raise ValueError(f"The extension {self.extension_name} has lost its SortingResult")
        return sorting_result

    # some attribuites come from sorting_result
    @property
    def format(self):
        return self.sorting_result.format
    
    @property
    def sparsity(self):
        return self.sorting_result.sparsity

    @property
    def folder(self):
        return self.sorting_result.folder
    
    def _get_binary_extension_folder(self):
        extension_folder = self.folder / "saved_extensions" /self.extension_name
        return extension_folder


    def _get_zarr_extension_group(self, mode='r+'):
        zarr_root = self.sorting_result._get_zarr_root(mode=mode)
        assert self.extension_name in zarr_root.keys(), (
            f"SortingResult: extension {self.extension_name} " f"is not in folder {self.folder}"
        )
        extension_group = zarr_root[self.extension_name]
        return extension_group
    

    @classmethod
    def load(cls, sorting_result):
        ext = cls(sorting_result)
        ext.load_params()
        ext.load_data()
        return ext

    def load_params(self):
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            params_file = extension_folder / "params.json"
            assert params_file.is_file(), f"No params file in extension {self.extension_name} folder"
            with open(str(params_file), "r") as f:
                params = json.load(f)

        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode='r')
            assert "params" in extension_group.attrs, f"No params file in extension {self.extension_name} folder"
            params = extension_group.attrs["params"]

        self._params = params

    def load_data(self):
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            for ext_data_file in extension_folder:
                if ext_data_file.name == "params.json":
                    continue
                ext_data_name = ext_data_file.stem
                if ext_data_file.suffix == ".json":
                    ext_data = json.load(ext_data_file.open("r"))
                elif ext_data_file.suffix == ".npy":
                    # The lazy loading of an extension is complicated because if we compute again
                    # and have a link to the old buffer on windows then it fails
                    # ext_data = np.load(ext_data_file, mmap_mode="r")
                    # so we go back to full loading
                    ext_data = np.load(ext_data_file)
                elif ext_data_file.suffix == ".csv":
                    import pandas as pd
                    ext_data = pd.read_csv(ext_data_file, index_col=0)
                elif ext_data_file.suffix == ".pkl":
                    ext_data = pickle.load(ext_data_file.open("rb"))
                else:
                    continue
                self._data[ext_data_name] = ext_data
        
        elif self.format == "zarr":
            raise NotImplementedError
            # TODO: decide if we make a copy or not
            # extension_group = self._get_zarr_extension_group(mode='r')
            # for ext_data_name in extension_group.keys():
            #     ext_data_ = extension_group[ext_data_name]
            #     if "dict" in ext_data_.attrs:
            #         ext_data = ext_data_[0]
            #     elif "dataframe" in ext_data_.attrs:
            #         import xarray
            #         ext_data = xarray.open_zarr(
            #             ext_data_.store, group=f"{extension_group.name}/{ext_data_name}"
            #         ).to_pandas()
            #         ext_data.index.rename("", inplace=True)
            #     else:
            #         ext_data = ext_data_
            #     self._data[ext_data_name] = ext_data

    def copy(self, new_sorting_result):
        new_extension = self.__class__(new_sorting_result)
        new_extension._params = self._params.copy()
        new_extension._data = self._data
        new_extension._save()

    def run(self, **kwargs):
        self._run(**kwargs)
        if not self.sorting_result.is_read_only():
            self._save(**kwargs)

    def _run(self, **kwargs):
        # must be implemented in subclass
        # must populate the self._data dictionary
        raise NotImplementedError

    def save(self, **kwargs):
        self._save(**kwargs)

    def _save(self, **kwargs):
        if self.format == "memory":
            return

        if self.sorting_result.is_read_only():
            raise ValueError("The SortingResult is read only save is not possible")
        
        # delete already saved
        self._reset_folder()
        self._save_params()


        if self.format == "binary_folder":
            import pandas as pd

            extension_folder = self._get_binary_extension_folder()

            for ext_data_name, ext_data in self._data.items():
                if isinstance(ext_data, dict):
                    with (extension_folder / f"{ext_data_name}.json").open("w") as f:
                        json.dump(ext_data, f)
                elif isinstance(ext_data, np.ndarray):
                    np.save(extension_folder / f"{ext_data_name}.npy", ext_data)
                elif isinstance(ext_data, pd.DataFrame):
                    ext_data.to_csv(extension_folder / f"{ext_data_name}.csv", index=True)
                else:
                    try:
                        with (extension_folder / f"{ext_data_name}.pkl").open("wb") as f:
                            pickle.dump(ext_data, f)
                    except:
                        raise Exception(f"Could not save {ext_data_name} as extension data")
        elif self.format == "zarr":
            from .zarrextractors import get_default_zarr_compressor
            import pandas as pd
            import numcodecs
            
            extension_group = self._get_zarr_extension_group(mode="r+")

            compressor = kwargs.get("compressor", None)
            if compressor is None:
                compressor = get_default_zarr_compressor()
            
            for ext_data_name, ext_data in self._data.items():
                if ext_data_name in extension_group:
                    del extension_group[ext_data_name]
                if isinstance(ext_data, dict):
                    extension_group.create_dataset(
                        name=ext_data_name, data=[ext_data], object_codec=numcodecs.JSON()
                    )
                    extension_group[ext_data_name].attrs["dict"] = True
                elif isinstance(ext_data, np.ndarray):
                    extension_group.create_dataset(name=ext_data_name, data=ext_data, compressor=compressor)
                elif isinstance(ext_data, pd.DataFrame):
                    ext_data.to_xarray().to_zarr(
                        store=extension_group.store,
                        group=f"{extension_group.name}/{ext_data_name}",
                        mode="a",
                    )
                    extension_group[ext_data_name].attrs["dataframe"] = True
                else:
                    try:
                        extension_group.create_dataset(
                            name=ext_data_name, data=ext_data, object_codec=numcodecs.Pickle()
                        )
                    except:
                        raise Exception(f"Could not save {ext_data_name} as extension data")

    def _reset_folder(self):
        """
        Delete the extension in folder (binary or zarr) and create an empty one.
        """
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            if extension_folder.is_dir():
                shutil.rmtree(extension_folder)
            extension_folder.mkdir(exist_ok=False, parents=True)

        elif self.format == "zarr":
            import zarr

            zarr_root = zarr.open(self.folder, mode="r+")
            self.extension_group = zarr_root.create_group(self.extension_name, overwrite=True)

    def reset(self):
        """
        Reset the waveform extension.
        Delete the sub folder and create a new empty one.
        """
        self._reset_folder()
        self._params = None
        self._data = dict()

    def _select_extension_data(self, unit_ids):
        # must be implemented in subclass
        raise NotImplementedError

    def set_params(self, **params):
        """
        Set parameters for the extension and
        make it persistent in json.
        """
        params = self._set_params(**params)
        self._params = params

        print(self.sorting_result.is_read_only())
        if self.sorting_result.is_read_only():
            return

        self._save_params()

    def _save_params(self):
        params_to_save = self._params.copy()
        if "sparsity" in params_to_save and params_to_save["sparsity"] is not None:
            assert isinstance(
                params_to_save["sparsity"], ChannelSparsity
            ), "'sparsity' parameter must be a ChannelSparsity object!"
            params_to_save["sparsity"] = params_to_save["sparsity"].to_dict()
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            extension_folder.mkdir(exist_ok=True)
            param_file = extension_folder / "params.json"
            param_file.write_text(json.dumps(check_json(params_to_save), indent=4), encoding="utf8")
        elif self.format == "zarr":
            self.extension_group.attrs["params"] = check_json(params_to_save)

    def _set_params(self, **params):
        # must be implemented in subclass
        # must return a cleaned version of params dict
        raise NotImplementedError

