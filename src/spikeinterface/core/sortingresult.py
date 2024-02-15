from __future__ import annotations
from typing import Literal, Optional

from pathlib import Path
import os
import json
import pickle
import weakref
import shutil
import warnings

import numpy as np

import probeinterface

import spikeinterface

from .baserecording import BaseRecording
from .basesorting import BaseSorting

from .base import load_extractor
from .recording_tools import check_probe_do_not_overlap, get_rec_attributes
from .sorting_tools import random_spikes_selection
from .core_tools import check_json
from .job_tools import split_job_kwargs
from .numpyextractors import SharedMemorySorting
from .sparsity import ChannelSparsity, estimate_sparsity
from .sortingfolder import NumpyFolderSorting
from .zarrextractors import get_default_zarr_compressor, ZarrSortingExtractor
from .node_pipeline import run_node_pipeline


# TODO make some_spikes a method of SortingResult


# high level function
def start_sorting_result(
    sorting, recording, format="memory", folder=None, sparse=True, sparsity=None, **sparsity_kwargs
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
        If True, then a sparsity mask is computed usingthe `estimate_sparsity()` function is run using
        a few spikes to get an estimate of dense templates to create a ChannelSparsity object.
        Then, the sparsity will be propagated to all ResultExtention that handle sparsity (like wavforms, pca, ...)
        You can control `estimate_sparsity()` : all extra arguments are propagated to it (included job_kwargs)
    sparsity: ChannelSparsity or None, default: None
        The sparsity used to compute waveforms. If this is given, `sparse` is ignored. Default None.

    Returns
    -------
    sorting_result: SortingResult
        The SortingResult object

    Examples
    --------
    >>> import spikeinterface as si

    >>> # Extract dense waveforms and save to disk with binary_folder format.
    >>> sortres = si.start_sorting_result(sorting, recording, format="binary_folder", folder="/path/to_my/result")

    >>> # Can be reload
    >>> sortres = si.load_sorting_result(folder="/path/to_my/result")

    >>> # Can run extension
    >>> sortres = si.compute("unit_locations", ...)

    >>> # Can be copy to another format (extensions are propagated)
    >>> sortres2 = sortres.save_as(format="memory")
    >>> sortres3 = sortres.save_as(format="zarr", folder="/path/to_my/result.zarr")

    >>> # Can make a copy with a subset of units (extensions are propagated for the unit subset)
    >>> sortres4 = sortres.select_units(unit_ids=sorting.units_ids[:5], format="memory")
    >>> sortres5 = sortres.select_units(unit_ids=sorting.units_ids[:5], format="binary_folder", folder="/result_5units")
    """

    # handle sparsity
    if sparsity is not None:
        # some checks
        assert isinstance(sparsity, ChannelSparsity), "'sparsity' must be a ChannelSparsity object"
        assert np.array_equal(
            sorting.unit_ids, sparsity.unit_ids
        ), "start_sorting_result(): if external sparsity is given unit_ids must correspond"
        assert np.array_equal(
            recording.channel_ids, recording.channel_ids
        ), "start_sorting_result(): if external sparsity is given unit_ids must correspond"
    elif sparse:
        sparsity = estimate_sparsity(recording, sorting, **sparsity_kwargs)
    else:
        sparsity = None

    sorting_result = SortingResult.create(sorting, recording, format=format, folder=folder, sparsity=sparsity)

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
    A SortingResult can be transfer to another format using `save_as()`

    This handle unit sparsity that can be propagated to ResultExtention.

    This handle spike sampling that can be propagated to ResultExtention : work only on a subset of spikes.

    This internally save a copy of the Sorting and extract main recording attributes (without traces) so
    the SortingResult object can be reload even if references to the original sorting and/or to the original recording
    are lost.

    SortingResult() should not never be used directly for creating: use instead start_sorting_result(sorting, resording, ...)
    or eventually SortingResult.create(...)
    """

    def __init__(
        self, sorting=None, recording=None, rec_attributes=None, format=None, sparsity=None, random_spikes_indices=None
    ):
        # very fast init because checks are done in load and create
        self.sorting = sorting
        # self.recorsding will be a property
        self._recording = recording
        self.rec_attributes = rec_attributes
        self.format = format
        self.sparsity = sparsity
        self.random_spikes_indices = random_spikes_indices

        # extensions are not loaded at init
        self.extensions = dict()

    def __repr__(self) -> str:
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nchan = self.get_num_channels()
        nunits = self.sorting.get_num_units()
        txt = f"{clsname}: {nchan} channels - {nunits} units - {nseg} segments - {self.format}"
        if self.is_sparse():
            txt += " - sparse"
        if self.has_recording():
            txt += " - has recording"
        ext_txt = f"Loaded {len(self.extensions)} extenstions: " + ", ".join(self.extensions.keys())
        txt += "\n" + ext_txt
        return txt

    ## create and load zone

    @classmethod
    def create(
        cls,
        sorting: BaseSorting,
        recording: BaseRecording,
        format: Literal[
            "memory",
            "binary_folder",
            "zarr",
        ] = "memory",
        folder=None,
        sparsity=None,
    ):
        # some checks
        assert sorting.sampling_frequency == recording.sampling_frequency
        # check that multiple probes are non-overlapping
        all_probes = recording.get_probegroup().probes
        check_probe_do_not_overlap(all_probes)

        if format == "memory":
            sortres = cls.create_memory(sorting, recording, sparsity, rec_attributes=None)
        elif format == "binary_folder":
            cls.create_binary_folder(folder, sorting, recording, sparsity, rec_attributes=None)
            sortres = cls.load_from_binary_folder(folder, recording=recording)
            sortres.folder = folder
        elif format == "zarr":
            cls.create_zarr(folder, sorting, recording, sparsity, rec_attributes=None)
            sortres = cls.load_from_zarr(folder, recording=recording)
            sortres.folder = folder
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
    def create_memory(cls, sorting, recording, sparsity, rec_attributes):
        # used by create and save_as

        if rec_attributes is None:
            assert recording is not None
            rec_attributes = get_rec_attributes(recording)
            rec_attributes["probegroup"] = recording.get_probegroup()
        else:
            # a copy is done to avoid shared dict between instances (which can block garbage collector)
            rec_attributes = rec_attributes.copy()

        # a copy of sorting is created directly in shared memory format to avoid further duplication of spikes.
        sorting_copy = SharedMemorySorting.from_sorting(sorting, with_metadata=True)
        sortres = SortingResult(
            sorting=sorting_copy, recording=recording, rec_attributes=rec_attributes, format="memory", sparsity=sparsity
        )
        return sortres

    @classmethod
    def create_binary_folder(cls, folder, sorting, recording, sparsity, rec_attributes):
        # used by create and save_as

        assert recording is not None, "To create a SortingResult you need recording not None"

        folder = Path(folder)
        if folder.is_dir():
            raise ValueError(f"Folder already exists {folder}")
        folder.mkdir(parents=True)

        info_file = folder / f"spikeinterface_info.json"
        info = dict(
            version=spikeinterface.__version__,
            dev_mode=spikeinterface.DEV_MODE,
            object="SortingResult",
        )
        with open(info_file, mode="w") as f:
            json.dump(check_json(info), f, indent=4)

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
            np.save(folder / "sparsity_mask.npy", sparsity.mask)
            # with open(folder / "sparsity.json", mode="w") as f:
            #     json.dump(check_json(sparsity.to_dict()), f)

    @classmethod
    def load_from_binary_folder(cls, folder, recording=None):
        folder = Path(folder)
        assert folder.is_dir(), f"This folder does not exists {folder}"

        # load internal sorting copy and make it sharedmem
        sorting = SharedMemorySorting.from_sorting(NumpyFolderSorting(folder / "sorting"), with_metadata=True)

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
            raise ValueError("This folder is not a SortingResult with format='binary_folder'")
        with open(rec_attributes_file, "r") as f:
            rec_attributes = json.load(f)
        # the probe is handle ouside the main json
        probegroup_file = folder / "recording_info" / "probegroup.json"

        if probegroup_file.is_file():
            rec_attributes["probegroup"] = probeinterface.read_probeinterface(probegroup_file)
        else:
            rec_attributes["probegroup"] = None

        # sparsity
        # sparsity_file = folder / "sparsity.json"
        sparsity_file = folder / "sparsity_mask.npy"
        if sparsity_file.is_file():
            sparsity_mask = np.load(sparsity_file)
            # with open(sparsity_file, mode="r") as f:
            #     sparsity = ChannelSparsity.from_dict(json.load(f))
            sparsity = ChannelSparsity(sparsity_mask, sorting.unit_ids, rec_attributes["channel_ids"])
        else:
            sparsity = None

        selected_spike_file = folder / "random_spikes_indices.npy"
        if selected_spike_file.is_file():
            random_spikes_indices = np.load(selected_spike_file)
        else:
            random_spikes_indices = None

        sortres = SortingResult(
            sorting=sorting,
            recording=recording,
            rec_attributes=rec_attributes,
            format="binary_folder",
            sparsity=sparsity,
            random_spikes_indices=random_spikes_indices,
        )

        return sortres

    def _get_zarr_root(self, mode="r+"):
        import zarr

        zarr_root = zarr.open(self.folder, mode=mode)
        return zarr_root

    @classmethod
    def create_zarr(cls, folder, sorting, recording, sparsity, rec_attributes):
        # used by create and save_as
        import zarr
        import numcodecs

        folder = Path(folder)
        # force zarr sufix
        if folder.suffix != ".zarr":
            folder = folder.parent / f"{folder.stem}.zarr"

        if folder.is_dir():
            raise ValueError(f"Folder already exists {folder}")

        zarr_root = zarr.open(folder, mode="w")

        info = dict(version=spikeinterface.__version__, dev_mode=spikeinterface.DEV_MODE, object="SortingResult")
        zarr_root.attrs["spikeinterface_info"] = check_json(info)

        # the recording
        rec_dict = recording.to_dict(relative_to=folder, recursive=True)
        zarr_rec = np.array([rec_dict], dtype=object)
        if recording.check_serializability("json"):
            # zarr_root.create_dataset("recording", data=rec_dict, object_codec=numcodecs.JSON())
            zarr_root.create_dataset("recording", data=zarr_rec, object_codec=numcodecs.JSON())
        elif recording.check_serializability("pickle"):
            # zarr_root.create_dataset("recording", data=rec_dict, object_codec=numcodecs.Pickle())
            zarr_root.create_dataset("recording", data=zarr_rec, object_codec=numcodecs.Pickle())
        else:
            warnings.warn(
                "SortingResult with zarr : the Recording is not json serializable, the recording link will be lost for futur load"
            )

        # sorting provenance
        sort_dict = sorting.to_dict(relative_to=folder, recursive=True)
        if sorting.check_serializability("json"):
            # zarr_root.attrs["sorting_provenance"] = check_json(sort_dict)
            zarr_sort = np.array([sort_dict], dtype=object)
            zarr_root.create_dataset("sorting_provenance", data=zarr_sort, object_codec=numcodecs.JSON())
        elif sorting.check_serializability("pickle"):
            # zarr_root.create_dataset("sorting_provenance", data=sort_dict, object_codec=numcodecs.Pickle())
            zarr_sort = np.array([sort_dict], dtype=object)
            zarr_root.create_dataset("sorting_provenance", data=zarr_sort, object_codec=numcodecs.Pickle())

        # else:
        #     warnings.warn("SortingResult with zarr : the sorting provenance is not json serializable, the sorting provenance link will be lost for futur load")

        recording_info = zarr_root.create_group("recording_info")

        if rec_attributes is None:
            assert recording is not None
            rec_attributes = get_rec_attributes(recording)
            probegroup = recording.get_probegroup()
        else:
            rec_attributes = rec_attributes.copy()
            probegroup = rec_attributes.pop("probegroup")

        recording_info.attrs["recording_attributes"] = check_json(rec_attributes)
        # recording_info.create_dataset("recording_attributes", data=check_json(rec_attributes), object_codec=numcodecs.JSON())

        if probegroup is not None:
            recording_info.attrs["probegroup"] = check_json(probegroup.to_dict())
            # recording_info.create_dataset("probegroup", data=check_json(probegroup.to_dict()), object_codec=numcodecs.JSON())

        if sparsity is not None:
            # zarr_root.attrs["sparsity"] = check_json(sparsity.to_dict())
            # zarr_root.create_dataset("sparsity", data=check_json(sparsity.to_dict()), object_codec=numcodecs.JSON())
            zarr_root.create_dataset("sparsity_mask", data=sparsity.mask)

        # write sorting copy
        from .zarrextractors import add_sorting_to_zarr_group

        # Alessio : we need to find a way to propagate compressor for all steps.
        # kwargs = dict(compressor=...)
        zarr_kwargs = dict()
        add_sorting_to_zarr_group(sorting, zarr_root.create_group("sorting"), **zarr_kwargs)

        recording_info = zarr_root.create_group("extensions")

    @classmethod
    def load_from_zarr(cls, folder, recording=None):
        import zarr

        folder = Path(folder)
        assert folder.is_dir(), f"This folder does not exists {folder}"

        zarr_root = zarr.open(folder, mode="r")

        # load internal sorting and make it sharedmem
        # TODO propagate storage_options
        sorting = SharedMemorySorting.from_sorting(
            ZarrSortingExtractor(folder, zarr_group="sorting"), with_metadata=True
        )

        # load recording if possible
        if recording is None:
            rec_dict = zarr_root["recording"][0]
            try:

                recording = load_extractor(rec_dict, base_folder=folder)
            except:
                recording = None
        else:
            # TODO maybe maybe not??? : do we need to check  attributes match internal rec_attributes
            # Note this will make the loading too slow
            pass

        # recording attributes
        rec_attributes = zarr_root["recording_info"].attrs["recording_attributes"]
        # rec_attributes = zarr_root["recording_info"]["recording_attributes"]
        if "probegroup" in zarr_root["recording_info"].attrs:
            probegroup_dict = zarr_root["recording_info"].attrs["probegroup"]
            # probegroup_dict = zarr_root["recording_info"]["probegroup"]
            rec_attributes["probegroup"] = probeinterface.ProbeGroup.from_dict(probegroup_dict)
        else:
            rec_attributes["probegroup"] = None

        # sparsity
        if "sparsity_mask" in zarr_root.attrs:
            # sparsity = zarr_root.attrs["sparsity"]
            sparsity = ChannelSparsity(zarr_root["sparsity_mask"], self.unit_ids, rec_attributes["channel_ids"])
        else:
            sparsity = None

        if "random_spikes_indices" in zarr_root.keys():
            random_spikes_indices = zarr_root["random_spikes_indices"]
        else:
            random_spikes_indices = None

        sortres = SortingResult(
            sorting=sorting,
            recording=recording,
            rec_attributes=rec_attributes,
            format="zarr",
            sparsity=sparsity,
            random_spikes_indices=random_spikes_indices,
        )

        return sortres

    def _save_or_select(self, format="binary_folder", folder=None, unit_ids=None) -> "SortingResult":
        """
        Internal used by both save_as(), copy() and select_units() which are more or less the same.
        """

        if self.has_recording():
            recording = self.recording
        else:
            recording = None

        if self.sparsity is not None and unit_ids is None:
            sparsity = self.sparsity
        elif self.sparsity is not None and unit_ids is not None:
            sparsity_mask = self.sparsity.mask[np.isin(self.unit_ids, unit_ids), :]
            sparsity = ChannelSparsity(sparsity_mask, unit_ids, self.channel_ids)
        else:
            sparsity = None

        # Note that the sorting is a copy we need to go back to the orginal sorting (if available)
        sorting_provenance = self.get_sorting_provenance()
        if sorting_provenance is None:
            # if the original sorting objetc is not available anymore (kilosort folder deleted, ....), take the copy
            sorting_provenance = self.sorting

        if unit_ids is not None:
            # when only some unit_ids then the sorting must be sliced
            # TODO check that unit_ids are in same order otherwise many extension do handle it properly!!!!
            sorting_provenance = sorting_provenance.select_units(unit_ids)

        if format == "memory":
            # This make a copy of actual SortingResult
            new_sortres = SortingResult.create_memory(sorting_provenance, recording, sparsity, self.rec_attributes)

        elif format == "binary_folder":
            # create  a new folder
            assert folder is not None, "For format='binary_folder' folder must be provided"
            SortingResult.create_binary_folder(folder, sorting_provenance, recording, sparsity, self.rec_attributes)
            new_sortres = SortingResult.load_from_binary_folder(folder, recording=recording)
            new_sortres.folder = folder

        elif format == "zarr":
            assert folder is not None, "For format='zarr' folder must be provided"
            SortingResult.create_zarr(folder, sorting_provenance, recording, sparsity, self.rec_attributes)
            new_sortres = SortingResult.load_from_zarr(folder, recording=recording)
            new_sortres.folder = folder
        else:
            raise ValueError("SortingResult.save: wrong format")

        # propagate random_spikes_indices is already done
        if self.random_spikes_indices is not None:
            if unit_ids is None:
                new_sortres.random_spikes_indices = self.random_spikes_indices.copy()
            else:
                # more tricky
                spikes = self.sorting.to_spike_vector()

                keep_unit_indices = np.flatnonzero(np.isin(self.unit_ids, unit_ids))
                keep_spike_mask = np.isin(spikes["unit_index"], keep_unit_indices)

                selected_mask = np.zeros(spikes.size, dtype=bool)
                selected_mask[self.random_spikes_indices] = True

                new_sortres.random_spikes_indices = np.flatnonzero(selected_mask[keep_spike_mask])

            # save it
            new_sortres._save_random_spikes_indices()

        # make a copy of extensions
        # note that the copy of extension handle itself the slicing of units when necessary and also the saveing
        for extension_name, extension in self.extensions.items():
            new_ext = new_sortres.extensions[extension_name] = extension.copy(new_sortres, unit_ids=unit_ids)

        return new_sortres

    def save_as(self, format="memory", folder=None) -> "SortingResult":
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
        return self._save_or_select(format=format, folder=folder, unit_ids=None)

    def select_units(self, unit_ids, format="memory", folder=None) -> "SortingResult":
        """
        This method is equivalent to `save_as()`but with a subset of units.
        Filters units by creating a new waveform extractor object in a new folder.

        Extensions are also updated to filter the selected unit ids.

        Parameters
        ----------
        unit_ids : list or array
            The unit ids to keep in the new SortingResult object
        folder : Path or None
            The new folder where selected waveforms are copied
        format:
        a
        Returns
        -------
        we :  SortingResult
            The newly create waveform extractor with the selected units
        """
        # TODO check that unit_ids are in same order otherwise many extension do handle it properly!!!!
        return self._save_or_select(format=format, folder=folder, unit_ids=unit_ids)

    def copy(self):
        """
        Create a a copy of SortingResult with format "memory".
        """
        return self._save_or_select(format="memory", folder=None, unit_ids=None)

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
            sorting_provenance = None

        elif self.format == "binary_folder":
            for type in ("json", "pickle"):
                filename = self.folder / f"sorting_provenance.{type}"
                sorting_provenance = None
                if filename.exists():
                    try:
                        sorting_provenance = load_extractor(filename, base_folder=self.folder)
                        break
                    except:
                        pass
                        # sorting_provenance = None

        elif self.format == "zarr":
            zarr_root = self._get_zarr_root(mode="r")
            if "sorting_provenance" in zarr_root.keys():
                sort_dict = zarr_root["sorting_provenance"][0]
                sorting_provenance = load_extractor(sort_dict, base_folder=self.folder)
            else:
                sorting_provenance = None

        return sorting_provenance

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
        all_channel_ids = list(self.rec_attributes["channel_ids"])
        indices = np.array([all_channel_ids.index(id) for id in channel_ids], dtype=int)
        return indices

    def get_recording_property(self, key) -> np.ndarray:
        values = np.array(self.rec_attributes["properties"].get(key, None))
        return values

    def get_sorting_property(self, key) -> np.ndarray:
        return self.sorting.get_property(key)

    def get_dtype(self):
        return self.rec_attributes["dtype"]

    ## extensions zone

    def compute(self, input, save=True, **kwargs):
        """
        Compute one extension or several extension.
        Internally calling compute_one_extension() or compute_several_extensions() depending th input type.

        Parameters
        ----------
        input: str or dict
            If the input is a string then compute one extension with compute_one_extension(extension_name=input, ...)
            If the input is a dict then compute several extension with compute_several_extensions(extensions=input)
        """
        if isinstance(input, str):
            return self.compute_one_extension(extension_name=input, save=save, **kwargs)
        elif isinstance(input, dict):
            params_, job_kwargs = split_job_kwargs(kwargs)
            assert len(params_) == 0, "Too many arguments for SortingResult.compute_several_extensions()"
            self.compute_several_extensions(extensions=input, save=save, **job_kwargs)

    def compute_one_extension(self, extension_name, save=True, **kwargs):
        """
        Compute one extension

        Parameters
        ----------
        extension_name: str
            The name of the extension.
            For instance "waveforms", "templates", ...
        save: bool, default True
            It the extension can be saved then it is saved.
            If not then the extension will only live in memory as long as the object is deleted.
            save=False is convinient to try some parameters without changing an already saved extension.

        **kwargs:
            All other kwargs are transimited to extension.set_params() or job_kwargs

        Returns
        -------
        sorting_result: SortingResult
            The SortingResult object

        Examples
        --------

        >>> extension = sortres.compute("waveforms", **some_params)
        >>> extension = sortres.compute_one_extension("waveforms", **some_params)
        >>> wfs = extension.data["waveforms"]

        """

        extension_class = get_extension_class(extension_name)

        if extension_class.need_job_kwargs:
            params, job_kwargs = split_job_kwargs(kwargs)
        else:
            params = kwargs
            job_kwargs = {}

        # check dependencies
        if extension_class.need_recording:
            assert self.has_recording(), f"Extension {extension_name} need the recording"
        for dependency_name in extension_class.depend_on:
            if "|" in dependency_name:
                # at least one extension must be done : usefull for "templates|fast_templates" for instance
                ok = any(self.get_extension(name) is not None for name in dependency_name.split("|"))
            else:
                ok = self.get_extension(dependency_name) is not None
            assert ok, f"Extension {extension_name} need {dependency_name} to be computed first"

        extension_instance = extension_class(self)
        extension_instance.set_params(save=save, **params)
        extension_instance.run(save=save, **job_kwargs)

        self.extensions[extension_name] = extension_instance

        # TODO : need discussion
        return extension_instance
        # OR
        return extension_instance.data

    def compute_several_extensions(self, extensions, save=True, **job_kwargs):
        """
        Compute several extensions

        Parameters
        ----------
        extensions: dict
            Key are extension_name and values are params.
        save: bool, default True
            It the extension can be saved then it is saved.
            If not then the extension will only live in memory as long as the object is deleted.
            save=False is convinient to try some parameters without changing an already saved extension.

        Returns
        -------
        No return

        Examples
        --------

        >>> sortres.compute({"waveforms": {"ms_before": 1.2}, "templates" : {"operators": ["average", "std", ]} })
        >>> sortres.compute_several_extensions({"waveforms": {"ms_before": 1.2}, "templates" : {"operators": ["average", "std"]}})

        """
        # TODO this is a simple implementation
        # this will be improved with nodepipeline!!!

        pipeline_mode = True
        for extension_name, extension_params in extensions.items():
            extension_class = get_extension_class(extension_name)
            if not extension_class.use_nodepipeline:
                pipeline_mode = False
                break

        if not pipeline_mode:
            # simple loop
            for extension_name, extension_params in extensions.items():
                extension_class = get_extension_class(extension_name)
                if extension_class.need_job_kwargs:
                    self.compute_one_extension(extension_name, save=save, **extension_params)
                else:
                    self.compute_one_extension(extension_name, save=save, **extension_params)
        else:

            all_nodes = []
            result_routage = []
            extension_instances = {}
            for extension_name, extension_params in extensions.items():
                extension_class = get_extension_class(extension_name)
                assert self.has_recording(), f"Extension {extension_name} need the recording"

                for variable_name in extension_class.nodepipeline_variables:
                    result_routage.append((extension_name, variable_name))

                extension_instance = extension_class(self)
                extension_instance.set_params(save=save, **extension_params)
                extension_instances[extension_name] = extension_instance

                nodes = extension_instance.get_pipeline_nodes()
                all_nodes.extend(nodes)

            job_name = "Compute : " + " + ".join(extensions.keys())
            results = run_node_pipeline(
                self.recording, all_nodes, job_kwargs=job_kwargs, job_name=job_name, gather_mode="memory"
            )

            for r, result in enumerate(results):
                extension_name, variable_name = result_routage[r]
                extension_instances[extension_name].data[variable_name] = result

            for extension_name, extension_instance in extension_instances.items():
                self.extensions[extension_name] = extension_instance
                if save:
                    extension_instance.save()

    def get_saved_extension_names(self):
        """
        Get extension saved in folder or zarr that can be loaded.
        """
        assert self.format != "memory"
        global _possible_extensions

        if self.format == "zarr":
            zarr_root = self._get_zarr_root(mode="r")
            if "extensions" in zarr_root.keys():
                extension_group = zarr_root["extensions"]
            else:
                extension_group = None

        saved_extension_names = []
        for extension_class in _possible_extensions:
            extension_name = extension_class.extension_name

            if self.format == "binary_folder":
                extension_folder = self.folder / "extensions" / extension_name
                is_saved = extension_folder.is_dir() and (extension_folder / "params.json").is_file()
            elif self.format == "zarr":
                if extension_group is not None:
                    is_saved = (
                        extension_name in extension_group.keys()
                        and "params" in extension_group[extension_name].attrs.keys()
                    )
                else:
                    is_saved = False
            if is_saved:
                saved_extension_names.append(extension_class.extension_name)

        return saved_extension_names

    def get_extension(self, extension_name: str):
        """
        Get a ResultExtension.
        If not loaded then load is automatic.

        Return None if the extension is not computed yet (this avoid the use of has_extension() and then get it)

        """
        if extension_name in self.extensions:
            return self.extensions[extension_name]

        elif self.format != "memory" and self.has_extension(extension_name):
            self.load_extension(extension_name)
            return self.extensions[extension_name]

        else:
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
        assert (
            self.format != "memory"
        ), "SortingResult.load_extension() do not work for format='memory' use SortingResult.get_extension()instead"

        extension_class = get_extension_class(extension_name)

        extension_instance = extension_class(self)
        extension_instance.load_params()
        extension_instance.load_data()

        self.extensions[extension_name] = extension_instance

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

        # delete from folder or zarr
        if self.format != "memory" and self.has_extension(extension_name):
            # need a reload to reset the folder
            ext = self.load_extension(extension_name)
            ext.reset()

        # remove from dict
        self.extensions.pop(extension_name, None)

    def get_loaded_extension_names(self):
        """
        Return the loaded or already computed extensions names.
        """
        return list(self.extensions.keys())

    def has_extension(self, extension_name: str) -> bool:
        """
        Check if the extension exists in memory (dict) or in the folder or in zarr.
        """
        if extension_name in self.extensions:
            return True
        elif self.format == "memory":
            return False
        elif extension_name in self.get_saved_extension_names():
            return True
        else:
            return False

    ## random_spikes_selection zone
    def select_random_spikes(self, **random_kwargs):
        # random_spikes_indices is a vector that refer to the spike vector of the sorting in absolut index
        assert self.random_spikes_indices is None, "select random spikes is already computed"

        self.random_spikes_indices = random_spikes_selection(
            self.sorting, self.rec_attributes["num_samples"], **random_kwargs
        )
        self._save_random_spikes_indices()

    def _save_random_spikes_indices(self):
        if self.format == "binary_folder":
            np.save(self.folder / "random_spikes_indices.npy", self.random_spikes_indices)
        elif self.format == "zarr":
            zarr_root = self._get_zarr_root()
            zarr_root.create_dataset("random_spikes_indices", data=self.random_spikes_indices)

    def get_selected_indices_in_spike_train(self, unit_id, segment_index):
        # usefull for Waveforms extractor backwars compatibility
        # In Waveforms extractor "selected_spikes" was a dict (key: unit_id) of list (segment_index) of indices of spikes in spiketrain
        assert self.random_spikes_indices is not None, "random spikes selection is not computeds"
        unit_index = self.sorting.id_to_index(unit_id)
        spikes = self.sorting.to_spike_vector()
        spike_indices_in_seg = np.flatnonzero(
            (spikes["segment_index"] == segment_index) & (spikes["unit_index"] == unit_index)
        )
        common_element, inds_left, inds_right = np.intersect1d(
            spike_indices_in_seg, self.random_spikes_indices, return_indices=True
        )
        selected_spikes_in_spike_train = inds_left
        return selected_spikes_in_spike_train


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
    assert (
        extension_name in extensions_dict
    ), f"Extension '{extension_name}' is not registered, please import related module before"
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

    An extension needs to inherit from this class and implement some attributes and abstract methods:
      * extension_name
      * depend_on
      * need_recording
      * use_nodepipeline
      * nodepipeline_variables only if use_nodepipeline=True
      * need_job_kwargs
      * _set_params()
      * _run()
      * _select_extension_data()
      * _get_data()

    The subclass must also set an `extension_name` class attribute which is not None by default.

    The subclass must also hanle an attribute `data` which is a dict contain the results after the `run()`.

    All ResultExtension will have a function associate for instance (this use the function_factory):
    comptute_unit_location(sorting_result, ...) will be equivalent to sorting_result.compute("unit_location", ...)


    """

    extension_name = None
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    nodepipeline_variables = None
    need_job_kwargs = False

    def __init__(self, sorting_result):
        self._sorting_result = weakref.ref(sorting_result)

        self.params = None
        self.data = dict()

    #######
    # This 3 methods must be implemented in the subclass!!!
    # See DummyResultExtension in test_sortingresult.py as a simple example
    def _run(self, **kwargs):
        # must be implemented in subclass
        # must populate the self.data dictionary
        raise NotImplementedError

    def _set_params(self, **params):
        # must be implemented in subclass
        # must return a cleaned version of params dict
        raise NotImplementedError

    def _select_extension_data(self, unit_ids):
        # must be implemented in subclass
        raise NotImplementedError

    def _get_pipeline_nodes(self):
        # must be implemented in subclass only if use_nodepipeline=True
        raise NotImplementedError

    def _get_data(self):
        # must be implemented in subclass
        raise NotImplementedError

    #
    #######

    @classmethod
    def function_factory(cls):
        # make equivalent
        # comptute_unit_location(sorting_result, ...) <> sorting_result.compute("unit_location", ...)
        # this also make backcompatibility
        # comptute_unit_location(we, ...)

        class FuncWrapper:
            def __init__(self, extension_name):
                self.extension_name = extension_name

            def __call__(self, sorting_result, load_if_exists=None, *args, **kwargs):
                from .waveforms_extractor_backwards_compatibility import MockWaveformExtractor

                if isinstance(sorting_result, MockWaveformExtractor):
                    # backward compatibility with WaveformsExtractor
                    sorting_result = sorting_result.sorting_result

                if not isinstance(sorting_result, SortingResult):
                    raise ValueError(f"compute_{self.extension_name}() need a SortingResult instance")

                if load_if_exists is not None:
                    # backward compatibility with "load_if_exists"
                    warnings.warn(
                        f"compute_{cls.extension_name}(..., load_if_exists=True/False) is kept for backward compatibility but should not be used anymore"
                    )
                    assert isinstance(load_if_exists, bool)
                    if load_if_exists:
                        ext = sorting_result.get_extension(self.extension_name)
                        return ext

                ext = sorting_result.compute(cls.extension_name, *args, **kwargs)
                return ext.get_data()

        func = FuncWrapper(cls.extension_name)
        func.__doc__ = cls.__doc__
        return func

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
        extension_folder = self.folder / "extensions" / self.extension_name
        return extension_folder

    def _get_zarr_extension_group(self, mode="r+"):
        zarr_root = self.sorting_result._get_zarr_root(mode=mode)
        extension_group = zarr_root["extensions"][self.extension_name]
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
            extension_group = self._get_zarr_extension_group(mode="r")
            assert "params" in extension_group.attrs, f"No params file in extension {self.extension_name} folder"
            params = extension_group.attrs["params"]

        self.params = params

    def load_data(self):
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            for ext_data_file in extension_folder.iterdir():
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
                self.data[ext_data_name] = ext_data

        elif self.format == "zarr":
            # Alessio
            # TODO: we need decide if we make a copy to memory or keep the lazy loading. For binary_folder it used to be lazy with memmap
            # but this make the garbage complicated when a data is hold by a plot but the o SortingResult is delete
            # lets talk
            extension_group = self._get_zarr_extension_group(mode="r")
            for ext_data_name in extension_group.keys():
                ext_data_ = extension_group[ext_data_name]
                if "dict" in ext_data_.attrs:
                    ext_data = ext_data_[0]
                elif "dataframe" in ext_data_.attrs:
                    import xarray

                    ext_data = xarray.open_zarr(
                        ext_data_.store, group=f"{extension_group.name}/{ext_data_name}"
                    ).to_pandas()
                    ext_data.index.rename("", inplace=True)
                elif "object" in ext_data_.attrs:
                    ext_data = ext_data_[0]
                else:
                    ext_data = ext_data_
                self.data[ext_data_name] = ext_data

    def copy(self, new_sorting_result, unit_ids=None):
        # alessio : please note that this also replace the old select_units!!!
        new_extension = self.__class__(new_sorting_result)
        new_extension.params = self.params.copy()
        if unit_ids is None:
            new_extension.data = self.data
        else:
            new_extension.data = self._select_extension_data(unit_ids)
        new_extension.save()
        return new_extension

    def run(self, save=True, **kwargs):
        if save and not self.sorting_result.is_read_only():
            # this also reset the folder or zarr group
            self._save_params()

        self._run(**kwargs)

        if save and not self.sorting_result.is_read_only():
            self._save_data(**kwargs)

    def save(self, **kwargs):
        self._save_params()
        self._save_data(**kwargs)

    def _save_data(self, **kwargs):
        if self.format == "memory":
            return

        if self.sorting_result.is_read_only():
            raise ValueError(f"The SortingResult is read only save extension {self.extension_name} is not possible")

        if self.format == "binary_folder":
            import pandas as pd

            extension_folder = self._get_binary_extension_folder()
            for ext_data_name, ext_data in self.data.items():
                if isinstance(ext_data, dict):
                    with (extension_folder / f"{ext_data_name}.json").open("w") as f:
                        json.dump(ext_data, f)
                elif isinstance(ext_data, np.ndarray):
                    data_file = extension_folder / f"{ext_data_name}.npy"
                    if isinstance(ext_data, np.memmap) and data_file.exists():
                        # important some SortingResult like ComputeWaveforms already run the computation with memmap
                        # so no need to save theses array
                        pass
                    else:
                        np.save(data_file, ext_data)
                elif isinstance(ext_data, pd.DataFrame):
                    ext_data.to_csv(extension_folder / f"{ext_data_name}.csv", index=True)
                else:
                    try:
                        with (extension_folder / f"{ext_data_name}.pkl").open("wb") as f:
                            pickle.dump(ext_data, f)
                    except:
                        raise Exception(f"Could not save {ext_data_name} as extension data")
        elif self.format == "zarr":

            import pandas as pd
            import numcodecs

            extension_group = self._get_zarr_extension_group(mode="r+")

            compressor = kwargs.get("compressor", None)
            if compressor is None:
                compressor = get_default_zarr_compressor()

            for ext_data_name, ext_data in self.data.items():
                if ext_data_name in extension_group:
                    del extension_group[ext_data_name]
                if isinstance(ext_data, dict):
                    extension_group.create_dataset(
                        name=ext_data_name, data=np.array([ext_data], dtype=object), object_codec=numcodecs.JSON()
                    )
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
                    # any object
                    try:
                        extension_group.create_dataset(
                            name=ext_data_name, data=np.array([ext_data], dtype=object), object_codec=numcodecs.Pickle()
                        )
                    except:
                        raise Exception(f"Could not save {ext_data_name} as extension data")
                    extension_group[ext_data_name].attrs["object"] = True

    def _reset_extension_folder(self):
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
            extension_group = zarr_root["extensions"].create_group(self.extension_name, overwrite=True)

    def reset(self):
        """
        Reset the waveform extension.
        Delete the sub folder and create a new empty one.
        """
        self._reset_extension_folder()
        self.params = None
        self.data = dict()

    def set_params(self, save=True, **params):
        """
        Set parameters for the extension and
        make it persistent in json.
        """
        # this ensure data is also deleted and corresponf to params
        # this also ensure the group is created
        self._reset_extension_folder()

        params = self._set_params(**params)
        self.params = params

        if self.sorting_result.is_read_only():
            return

        if save:
            self._save_params()

    def _save_params(self):
        params_to_save = self.params.copy()

        self._reset_extension_folder()

        # TODO make sparsity local Result specific
        # if "sparsity" in params_to_save and params_to_save["sparsity"] is not None:
        #     assert isinstance(
        #         params_to_save["sparsity"], ChannelSparsity
        #     ), "'sparsity' parameter must be a ChannelSparsity object!"
        #     params_to_save["sparsity"] = params_to_save["sparsity"].to_dict()

        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            extension_folder.mkdir(exist_ok=True, parents=True)
            param_file = extension_folder / "params.json"
            param_file.write_text(json.dumps(check_json(params_to_save), indent=4), encoding="utf8")
        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r+")
            extension_group.attrs["params"] = check_json(params_to_save)

    def get_pipeline_nodes(self):
        assert (
            self.use_nodepipeline
        ), "ResultExtension.get_pipeline_nodes() must be called only when use_nodepipeline=True"
        return self._get_pipeline_nodes()

    def get_data(self, *args, **kwargs):
        assert len(self.data) > 0, f"You must run the extension {self.extension_name} before retrieving data"
        return self._get_data(*args, **kwargs)
