from __future__ import annotations
from typing import Literal, Optional

from pathlib import Path
import os
import json
import pickle
import weakref
import shutil
import warnings
import importlib

import numpy as np

import probeinterface

import spikeinterface

from .baserecording import BaseRecording
from .basesorting import BaseSorting

from .base import load_extractor
from .recording_tools import check_probe_do_not_overlap, get_rec_attributes
from .core_tools import check_json, retrieve_importing_provenance
from .job_tools import split_job_kwargs
from .numpyextractors import NumpySorting
from .sparsity import ChannelSparsity, estimate_sparsity
from .sortingfolder import NumpyFolderSorting
from .zarrextractors import get_default_zarr_compressor, ZarrSortingExtractor
from .node_pipeline import run_node_pipeline


# high level function
def create_sorting_analyzer(
    sorting,
    recording,
    format="memory",
    folder=None,
    sparse=True,
    sparsity=None,
    return_scaled=True,
    overwrite=False,
    **sparsity_kwargs,
):
    """
    Create a SortingAnalyzer by pairing a Sorting and the corresponding Recording.

    This object will handle a list of AnalyzerExtension for all the post processing steps like: waveforms,
    templates, unit locations, spike locations, quality metrics ...

    This object will be also use used for plotting purpose.


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
        If True, then a sparsity mask is computed using the `estimate_sparsity()` function using
        a few spikes to get an estimate of dense templates to create a ChannelSparsity object.
        Then, the sparsity will be propagated to all ResultExtention that handle sparsity (like wavforms, pca, ...)
        You can control `estimate_sparsity()` : all extra arguments are propagated to it (included job_kwargs)
    sparsity: ChannelSparsity or None, default: None
        The sparsity used to compute waveforms. If this is given, `sparse` is ignored.
    return_scaled: bool, default: True
        All extensions that play with traces will use this global return_scaled: "waveforms", "noise_levels", "templates".
        This prevent return_scaled being differents from different extensions and having wrong snr for instance.

    Returns
    -------
    sorting_analyzer: SortingAnalyzer
        The SortingAnalyzer object

    Examples
    --------
    >>> import spikeinterface as si

    >>> # Extract dense waveforms and save to disk with binary_folder format.
    >>> sorting_analyzer = si.create_sorting_analyzer(sorting, recording, format="binary_folder", folder="/path/to_my/result")

    >>> # Can be reload
    >>> sorting_analyzer = si.load_sorting_analyzer(folder="/path/to_my/result")

    >>> # Can run extension
    >>> sorting_analyzer = si.compute("unit_locations", ...)

    >>> # Can be copy to another format (extensions are propagated)
    >>> sorting_analyzer2 = sorting_analyzer.save_as(format="memory")
    >>> sorting_analyzer3 = sorting_analyzer.save_as(format="zarr", folder="/path/to_my/result.zarr")

    >>> # Can make a copy with a subset of units (extensions are propagated for the unit subset)
    >>> sorting_analyzer4 = sorting_analyzer.select_units(unit_ids=sorting.units_ids[:5], format="memory")
    >>> sorting_analyzer5 = sorting_analyzer.select_units(unit_ids=sorting.units_ids[:5], format="binary_folder", folder="/result_5units")

    Notes
    -----

    By default creating a SortingAnalyzer can be slow because the sparsity is estimated by default.
    In some situation, sparsity is not needed, so to make it fast creation, you need to turn
    sparsity off (or give external sparsity) like this.
    """
    if format != "memory":
        if Path(folder).is_dir():
            if not overwrite:
                raise ValueError(f"Folder already exists {folder}! Use overwrite=True to overwrite it.")
            else:
                shutil.rmtree(folder)

    # handle sparsity
    if sparsity is not None:
        # some checks
        assert isinstance(sparsity, ChannelSparsity), "'sparsity' must be a ChannelSparsity object"
        assert np.array_equal(
            sorting.unit_ids, sparsity.unit_ids
        ), "create_sorting_analyzer(): if external sparsity is given unit_ids must correspond"
        assert np.array_equal(
            recording.channel_ids, sparsity.channel_ids
        ), "create_sorting_analyzer(): if external sparsity is given unit_ids must correspond"
    elif sparse:
        sparsity = estimate_sparsity(recording, sorting, **sparsity_kwargs)
    else:
        sparsity = None

    if return_scaled and not recording.has_scaled_traces() and recording.get_dtype().kind == "i":
        print("create_sorting_analyzer: recording does not have scaling to uV, forcing return_scaled=False")
        return_scaled = False

    sorting_analyzer = SortingAnalyzer.create(
        sorting, recording, format=format, folder=folder, sparsity=sparsity, return_scaled=return_scaled
    )

    return sorting_analyzer


def load_sorting_analyzer(folder, load_extensions=True, format="auto"):
    """
    Load a SortingAnalyzer object from disk.

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
    sorting_analyzer: SortingAnalyzer
        The loaded SortingAnalyzer

    """
    return SortingAnalyzer.load(folder, load_extensions=load_extensions, format=format)


class SortingAnalyzer:
    """
    Class to make a pair of Recording-Sorting which will be used used for all post postprocessing,
    visualization and quality metric computation.

    This internally maintains a list of computed ResultExtention (waveform, pca, unit position, spike position, ...).

    This can live in memory and/or can be be persistent to disk in 2 internal formats (folder/json/npz or zarr).
    A SortingAnalyzer can be transfer to another format using `save_as()`

    This handle unit sparsity that can be propagated to ResultExtention.

    This handle spike sampling that can be propagated to ResultExtention : works on only a subset of spikes.

    This internally saves a copy of the Sorting and extracts main recording attributes (without traces) so
    the SortingAnalyzer object can be reloaded even if references to the original sorting and/or to the original recording
    are lost.

    SortingAnalyzer() should not never be used directly for creating: use instead create_sorting_analyzer(sorting, resording, ...)
    or eventually SortingAnalyzer.create(...)
    """

    def __init__(
        self,
        sorting=None,
        recording=None,
        rec_attributes=None,
        format=None,
        sparsity=None,
        return_scaled=True,
    ):
        # very fast init because checks are done in load and create
        self.sorting = sorting
        # self.recorsding will be a property
        self._recording = recording
        self.rec_attributes = rec_attributes
        self.format = format
        self.sparsity = sparsity
        self.return_scaled = return_scaled

        # extensions are not loaded at init
        self.extensions = dict()

    def __repr__(self) -> str:
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nchan = self.get_num_channels()
        nunits = self.get_num_units()
        txt = f"{clsname}: {nchan} channels - {nunits} units - {nseg} segments - {self.format}"
        if self.is_sparse():
            txt += " - sparse"
        if self.has_recording():
            txt += " - has recording"
        ext_txt = f"Loaded {len(self.extensions)} extensions: " + ", ".join(self.extensions.keys())
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
        return_scaled=True,
    ):
        # some checks
        assert sorting.sampling_frequency == recording.sampling_frequency
        # check that multiple probes are non-overlapping
        all_probes = recording.get_probegroup().probes
        check_probe_do_not_overlap(all_probes)

        if format == "memory":
            sorting_analyzer = cls.create_memory(sorting, recording, sparsity, return_scaled, rec_attributes=None)
        elif format == "binary_folder":
            cls.create_binary_folder(folder, sorting, recording, sparsity, return_scaled, rec_attributes=None)
            sorting_analyzer = cls.load_from_binary_folder(folder, recording=recording)
            sorting_analyzer.folder = Path(folder)
        elif format == "zarr":
            cls.create_zarr(folder, sorting, recording, sparsity, return_scaled, rec_attributes=None)
            sorting_analyzer = cls.load_from_zarr(folder, recording=recording)
            sorting_analyzer.folder = Path(folder)
        else:
            raise ValueError("SortingAnalyzer.create: wrong format")

        return sorting_analyzer

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
            sorting_analyzer = SortingAnalyzer.load_from_binary_folder(folder, recording=recording)
        elif format == "zarr":
            sorting_analyzer = SortingAnalyzer.load_from_zarr(folder, recording=recording)

        sorting_analyzer.folder = folder

        if load_extensions:
            sorting_analyzer.load_all_saved_extension()

        return sorting_analyzer

    @classmethod
    def create_memory(cls, sorting, recording, sparsity, return_scaled, rec_attributes):
        # used by create and save_as

        if rec_attributes is None:
            assert recording is not None
            rec_attributes = get_rec_attributes(recording)
            rec_attributes["probegroup"] = recording.get_probegroup()
        else:
            # a copy is done to avoid shared dict between instances (which can block garbage collector)
            rec_attributes = rec_attributes.copy()

        # a copy of sorting is copied in memory for fast access
        sorting_copy = NumpySorting.from_sorting(sorting, with_metadata=True, copy_spike_vector=True)

        sorting_analyzer = SortingAnalyzer(
            sorting=sorting_copy,
            recording=recording,
            rec_attributes=rec_attributes,
            format="memory",
            sparsity=sparsity,
            return_scaled=return_scaled,
        )
        return sorting_analyzer

    @classmethod
    def create_binary_folder(cls, folder, sorting, recording, sparsity, return_scaled, rec_attributes):
        # used by create and save_as

        assert recording is not None, "To create a SortingAnalyzer you need recording not None"

        folder = Path(folder)
        if folder.is_dir():
            raise ValueError(f"Folder already exists {folder}")
        folder.mkdir(parents=True)

        info_file = folder / f"spikeinterface_info.json"
        info = dict(
            version=spikeinterface.__version__,
            dev_mode=spikeinterface.DEV_MODE,
            object="SortingAnalyzer",
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

        settings_file = folder / f"settings.json"
        settings = dict(
            return_scaled=return_scaled,
        )
        with open(settings_file, mode="w") as f:
            json.dump(check_json(settings), f, indent=4)

    @classmethod
    def load_from_binary_folder(cls, folder, recording=None):
        folder = Path(folder)
        assert folder.is_dir(), f"This folder does not exists {folder}"

        # load internal sorting copy in memory
        sorting = NumpySorting.from_sorting(
            NumpyFolderSorting(folder / "sorting"), with_metadata=True, copy_spike_vector=True
        )

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
            raise ValueError("This folder is not a SortingAnalyzer with format='binary_folder'")
        with open(rec_attributes_file, "r") as f:
            rec_attributes = json.load(f)
        # the probe is handle ouside the main json
        probegroup_file = folder / "recording_info" / "probegroup.json"

        if probegroup_file.is_file():
            rec_attributes["probegroup"] = probeinterface.read_probeinterface(probegroup_file)
        else:
            rec_attributes["probegroup"] = None

        # sparsity
        sparsity_file = folder / "sparsity_mask.npy"
        if sparsity_file.is_file():
            sparsity_mask = np.load(sparsity_file)
            sparsity = ChannelSparsity(sparsity_mask, sorting.unit_ids, rec_attributes["channel_ids"])
        else:
            sparsity = None

        # PATCH: Because SortingAnalyzer added this json during the development of 0.101.0 we need to save
        # this as a bridge for early adopters. The else branch can be removed in version 0.102.0/0.103.0
        # so that this can be simplified in the future
        # See https://github.com/SpikeInterface/spikeinterface/issues/2788

        settings_file = folder / f"settings.json"
        if settings_file.exists():
            with open(settings_file, "r") as f:
                settings = json.load(f)
        else:
            warnings.warn("settings.json not found for this folder writing one with return_scaled=True")
            settings = dict(return_scaled=True)
            with open(settings_file, "w") as f:
                json.dump(check_json(settings), f, indent=4)

        return_scaled = settings["return_scaled"]

        sorting_analyzer = SortingAnalyzer(
            sorting=sorting,
            recording=recording,
            rec_attributes=rec_attributes,
            format="binary_folder",
            sparsity=sparsity,
            return_scaled=return_scaled,
        )

        return sorting_analyzer

    def _get_zarr_root(self, mode="r+"):
        import zarr

        zarr_root = zarr.open(self.folder, mode=mode)
        return zarr_root

    @classmethod
    def create_zarr(cls, folder, sorting, recording, sparsity, return_scaled, rec_attributes):
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

        info = dict(version=spikeinterface.__version__, dev_mode=spikeinterface.DEV_MODE, object="SortingAnalyzer")
        zarr_root.attrs["spikeinterface_info"] = check_json(info)

        settings = dict(return_scaled=return_scaled)
        zarr_root.attrs["settings"] = check_json(settings)

        # the recording
        rec_dict = recording.to_dict(relative_to=folder, recursive=True)

        if recording.check_serializability("json"):
            # zarr_root.create_dataset("recording", data=rec_dict, object_codec=numcodecs.JSON())
            zarr_rec = np.array([check_json(rec_dict)], dtype=object)
            zarr_root.create_dataset("recording", data=zarr_rec, object_codec=numcodecs.JSON())
        elif recording.check_serializability("pickle"):
            # zarr_root.create_dataset("recording", data=rec_dict, object_codec=numcodecs.Pickle())
            zarr_rec = np.array([rec_dict], dtype=object)
            zarr_root.create_dataset("recording", data=zarr_rec, object_codec=numcodecs.Pickle())
        else:
            warnings.warn(
                "SortingAnalyzer with zarr : the Recording is not json serializable, the recording link will be lost for future load"
            )

        # sorting provenance
        sort_dict = sorting.to_dict(relative_to=folder, recursive=True)
        if sorting.check_serializability("json"):
            zarr_sort = np.array([check_json(sort_dict)], dtype=object)
            zarr_root.create_dataset("sorting_provenance", data=zarr_sort, object_codec=numcodecs.JSON())
        elif sorting.check_serializability("pickle"):
            zarr_sort = np.array([sort_dict], dtype=object)
            zarr_root.create_dataset("sorting_provenance", data=zarr_sort, object_codec=numcodecs.Pickle())

        # else:
        #     warnings.warn("SortingAnalyzer with zarr : the sorting provenance is not json serializable, the sorting provenance link will be lost for futur load")

        recording_info = zarr_root.create_group("recording_info")

        if rec_attributes is None:
            assert recording is not None
            rec_attributes = get_rec_attributes(recording)
            probegroup = recording.get_probegroup()
        else:
            rec_attributes = rec_attributes.copy()
            probegroup = rec_attributes.pop("probegroup")

        recording_info.attrs["recording_attributes"] = check_json(rec_attributes)

        if probegroup is not None:
            recording_info.attrs["probegroup"] = check_json(probegroup.to_dict())

        if sparsity is not None:
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
        assert folder.is_dir(), f"This folder does not exist {folder}"

        zarr_root = zarr.open(folder, mode="r")

        # load internal sorting in memory
        # TODO propagate storage_options
        sorting = NumpySorting.from_sorting(
            ZarrSortingExtractor(folder, zarr_group="sorting"), with_metadata=True, copy_spike_vector=True
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
        if "probegroup" in zarr_root["recording_info"].attrs:
            probegroup_dict = zarr_root["recording_info"].attrs["probegroup"]
            rec_attributes["probegroup"] = probeinterface.ProbeGroup.from_dict(probegroup_dict)
        else:
            rec_attributes["probegroup"] = None

        # sparsity
        if "sparsity_mask" in zarr_root.attrs:
            # sparsity = zarr_root.attrs["sparsity"]
            sparsity = ChannelSparsity(zarr_root["sparsity_mask"], cls.unit_ids, rec_attributes["channel_ids"])
        else:
            sparsity = None

        return_scaled = zarr_root.attrs["settings"]["return_scaled"]

        sorting_analyzer = SortingAnalyzer(
            sorting=sorting,
            recording=recording,
            rec_attributes=rec_attributes,
            format="zarr",
            sparsity=sparsity,
            return_scaled=return_scaled,
        )

        return sorting_analyzer

    def _save_or_select(self, format="binary_folder", folder=None, unit_ids=None) -> "SortingAnalyzer":
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
            # This make a copy of actual SortingAnalyzer
            new_sorting_analyzer = SortingAnalyzer.create_memory(
                sorting_provenance, recording, sparsity, self.return_scaled, self.rec_attributes
            )

        elif format == "binary_folder":
            # create  a new folder
            assert folder is not None, "For format='binary_folder' folder must be provided"
            folder = Path(folder)
            SortingAnalyzer.create_binary_folder(
                folder, sorting_provenance, recording, sparsity, self.return_scaled, self.rec_attributes
            )
            new_sorting_analyzer = SortingAnalyzer.load_from_binary_folder(folder, recording=recording)
            new_sorting_analyzer.folder = folder

        elif format == "zarr":
            assert folder is not None, "For format='zarr' folder must be provided"
            folder = Path(folder)
            if folder.suffix != ".zarr":
                folder = folder.parent / f"{folder.stem}.zarr"
            SortingAnalyzer.create_zarr(
                folder, sorting_provenance, recording, sparsity, self.return_scaled, self.rec_attributes
            )
            new_sorting_analyzer = SortingAnalyzer.load_from_zarr(folder, recording=recording)
            new_sorting_analyzer.folder = folder
        else:
            raise ValueError(f"SortingAnalyzer.save: unsupported format: {format}")

        # make a copy of extensions
        # note that the copy of extension handle itself the slicing of units when necessary and also the saveing
        for extension_name, extension in self.extensions.items():
            new_ext = new_sorting_analyzer.extensions[extension_name] = extension.copy(
                new_sorting_analyzer, unit_ids=unit_ids
            )

        return new_sorting_analyzer

    def save_as(self, format="memory", folder=None) -> "SortingAnalyzer":
        """
        Save SortingAnalyzer object into another format.
        Uselful for memory to zarr or memory to binary.

        Note that the recording provenance or sorting provenance can be lost.

        Mainly propagates the copied sorting and recording properties.

        Parameters
        ----------
        folder : str or Path
            The output waveform folder
        format : "binary_folder" | "zarr", default: "binary_folder"
            The backend to use for saving the waveforms
        """
        return self._save_or_select(format=format, folder=folder, unit_ids=None)

    def select_units(self, unit_ids, format="memory", folder=None) -> "SortingAnalyzer":
        """
        This method is equivalent to `save_as()`but with a subset of units.
        Filters units by creating a new sorting analyzer object in a new folder.

        Extensions are also updated to filter the selected unit ids.

        Parameters
        ----------
        unit_ids : list or array
            The unit ids to keep in the new SortingAnalyzer object
        folder : Path or None
            The new folder where selected waveforms are copied
        format:
        a
        Returns
        -------
        we :  SortingAnalyzer
            The newly create sorting_analyzer with the selected units
        """
        # TODO check that unit_ids are in same order otherwise many extension do handle it properly!!!!
        return self._save_or_select(format=format, folder=folder, unit_ids=unit_ids)

    def copy(self):
        """
        Create a a copy of SortingAnalyzer with format "memory".
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
            raise ValueError("SortingAnalyzer could not load the recording")
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

    def get_num_units(self) -> int:
        return self.sorting.get_num_units()

    ## extensions zone
    def compute(self, input, save=True, extension_params=None, **kwargs):
        """
        Compute one extension or several extensiosn.
        Internally calls compute_one_extension() or compute_several_extensions() depending on the input type.

        Parameters
        ----------
        input: str or dict or list
            The extensions to compute, which can be passed as:

            * a string: compute one extension. Additional parameters can be passed as key word arguments.
            * a dict: compute several extensions. The keys are the extension names and the values are dictiopnaries with the extension parameters.
            * a list: compute several extensions. The list contains the extension names. Additional parameters can be passed with the extension_params
              argument.
        save: bool, default: True
            If True the extension is saved to disk (only if sorting analyzer format is not "memory")
        extension_params: dict or None, default: None
            If input is a list, this parameter can be used to specify parameters for each extension.
            The extension_params keys must be included in the input list.
        **kwargs:
            All other kwargs are transmitted to extension.set_params() (if input is a string) or job_kwargs

        Returns
        -------
        extension: SortingAnalyzerExtension | None
            The extension instance if input is a string, None otherwise.

        Examples
        --------
        This function accepts the following possible signatures for flexibility:

        Compute one extension, with parameters:
        >>> analyzer.compute("waveforms", ms_before=1.5, ms_after=2.5)

        Compute two extensions with a list as input and with default parameters:
        >>> analyzer.compute(["random_spikes", "waveforms"])

        Compute two extensions with dict as input, one dict per extension
        >>> analyzer.compute({"random_spikes":{}, "waveforms":{"ms_before":1.5, "ms_after", "2.5"}})

        Compute two extensions with an input list specifying custom parameters for one
        (the other will use default parameters):
        >>> analyzer.compute(
            ["random_spikes", "waveforms"],
            extension_params={"waveforms":{"ms_before":1.5, "ms_after", "2.5"}}
        )
        """
        if isinstance(input, str):
            return self.compute_one_extension(extension_name=input, save=save, **kwargs)
        elif isinstance(input, dict):
            params_, job_kwargs = split_job_kwargs(kwargs)
            assert len(params_) == 0, "Too many arguments for SortingAnalyzer.compute_several_extensions()"
            self.compute_several_extensions(extensions=input, save=save, **job_kwargs)
        elif isinstance(input, list):
            params_, job_kwargs = split_job_kwargs(kwargs)
            assert len(params_) == 0, "Too many arguments for SortingAnalyzer.compute_several_extensions()"
            extensions = {k: {} for k in input}
            if extension_params is not None:
                for ext_name, ext_params in extension_params.items():
                    assert (
                        ext_name in input
                    ), f"SortingAnalyzer.compute(): Parameters specified for {ext_name}, which is not in the specified {input}"
                    extensions[ext_name] = ext_params
            self.compute_several_extensions(extensions=extensions, save=save, **job_kwargs)
        else:
            raise ValueError("SortingAnalyzer.compute() need str, dict or list")

    def compute_one_extension(self, extension_name, save=True, **kwargs):
        """
        Compute one extension.

        Important note: when computing again an extension, all extensions that depend on it
        will be automatically and silently deleted to keep a coherent data.

        Parameters
        ----------
        extension_name: str
            The name of the extension.
            For instance "waveforms", "templates", ...
        save: bool, default: True
            It the extension can be saved then it is saved.
            If not then the extension will only live in memory as long as the object is deleted.
            save=False is convenient to try some parameters without changing an already saved extension.

        **kwargs:
            All other kwargs are transmitted to extension.set_params() or job_kwargs

        Returns
        -------
        result_extension: AnalyzerExtension
            Return the extension instance.

        Examples
        --------

        >>> Note that the return is the instance extension.
        >>> extension = sorting_analyzer.compute("waveforms", **some_params)
        >>> extension = sorting_analyzer.compute_one_extension("waveforms", **some_params)
        >>> wfs = extension.data["waveforms"]
        >>> # Note this can be be done in the old way style BUT the return is not the same it return directly data
        >>> wfs = compute_waveforms(sorting_analyzer, **some_params)

        """
        for child in _get_children_dependencies(extension_name):
            self.delete_extension(child)

        extension_class = get_extension_class(extension_name)

        if extension_class.need_job_kwargs:
            params, job_kwargs = split_job_kwargs(kwargs)
        else:
            params = kwargs
            job_kwargs = {}

        # check dependencies
        if extension_class.need_recording:
            assert self.has_recording(), f"Extension {extension_name} requires the recording"
        for dependency_name in extension_class.depend_on:
            if "|" in dependency_name:
                ok = any(self.get_extension(name) is not None for name in dependency_name.split("|"))
            else:
                ok = self.get_extension(dependency_name) is not None
            assert ok, f"Extension {extension_name} requires {dependency_name} to be computed first"

        extension_instance = extension_class(self)
        extension_instance.set_params(save=save, **params)
        extension_instance.run(save=save, **job_kwargs)

        self.extensions[extension_name] = extension_instance

        return extension_instance

    def compute_several_extensions(self, extensions, save=True, **job_kwargs):
        """
        Compute several extensions

        Important note: when computing again an extension, all extensions that depend on it
        will be automatically and silently deleted to keep a coherent data.


        Parameters
        ----------
        extensions: dict
            Keys are extension_names and values are params.
        save: bool, default: True
            It the extension can be saved then it is saved.
            If not then the extension will only live in memory as long as the object is deleted.
            save=False is convenient to try some parameters without changing an already saved extension.

        Returns
        -------
        No return

        Examples
        --------

        >>> sorting_analyzer.compute({"waveforms": {"ms_before": 1.2}, "templates" : {"operators": ["average", "std", ]} })
        >>> sorting_analyzer.compute_several_extensions({"waveforms": {"ms_before": 1.2}, "templates" : {"operators": ["average", "std"]}})

        """
        for extension_name in extensions.keys():
            for child in _get_children_dependencies(extension_name):
                self.delete_extension(child)

        extensions_with_pipeline = {}
        extensions_without_pipeline = {}
        for extension_name, extension_params in extensions.items():
            extension_class = get_extension_class(extension_name)
            if extension_class.use_nodepipeline:
                extensions_with_pipeline[extension_name] = extension_params
            else:
                extensions_without_pipeline[extension_name] = extension_params

        # First extensions without pipeline
        for extension_name, extension_params in extensions_without_pipeline.items():
            extension_class = get_extension_class(extension_name)
            if extension_class.need_job_kwargs:
                self.compute_one_extension(extension_name, save=save, **extension_params, **job_kwargs)
            else:
                self.compute_one_extension(extension_name, save=save, **extension_params)
        # then extensions with pipeline
        if len(extensions_with_pipeline) > 0:
            all_nodes = []
            result_routage = []
            extension_instances = {}
            for extension_name, extension_params in extensions_with_pipeline.items():
                extension_class = get_extension_class(extension_name)
                assert self.has_recording(), f"Extension {extension_name} need the recording"

                for variable_name in extension_class.nodepipeline_variables:
                    result_routage.append((extension_name, variable_name))

                extension_instance = extension_class(self)
                extension_instance.set_params(save=save, **extension_params)
                extension_instances[extension_name] = extension_instance

                nodes = extension_instance.get_pipeline_nodes()
                all_nodes.extend(nodes)

            job_name = "Compute : " + " + ".join(extensions_with_pipeline.keys())
            results = run_node_pipeline(
                self.recording,
                all_nodes,
                job_kwargs=job_kwargs,
                job_name=job_name,
                gather_mode="memory",
                squeeze_output=False,
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
        Get extension names saved in folder or zarr that can be loaded.
        This do not load data, this only explores the directory.
        """
        saved_extension_names = []
        if self.format == "binary_folder":
            ext_folder = self.folder / "extensions"
            if ext_folder.is_dir():
                for extension_folder in ext_folder.iterdir():
                    is_saved = extension_folder.is_dir() and (extension_folder / "params.json").is_file()
                    if not is_saved:
                        continue
                    saved_extension_names.append(extension_folder.stem)

        elif self.format == "zarr":
            zarr_root = self._get_zarr_root(mode="r")
            if "extensions" in zarr_root.keys():
                extension_group = zarr_root["extensions"]
                for extension_name in extension_group.keys():
                    if "params" in extension_group[extension_name].attrs.keys():
                        saved_extension_names.append(extension_name)

        else:
            raise ValueError("SortingAnalyzer.get_saved_extension_names() works only with binary_folder and zarr")

        return saved_extension_names

    def get_extension(self, extension_name: str):
        """
        Get a AnalyzerExtension.
        If not loaded then load is automatic.

        Return None if the extension is not computed yet (this avoids the use of has_extension() and then get it)

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
        Load an extension from a folder or zarr into the `ResultSorting.extensions` dict.

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
        ), "SortingAnalyzer.load_extension() does not work for format='memory' use SortingAnalyzer.get_extension() instead"

        extension_class = get_extension_class(extension_name)

        extension_instance = extension_class(self)
        extension_instance.load_params()
        extension_instance.load_data()

        self.extensions[extension_name] = extension_instance

        return extension_instance

    def load_all_saved_extension(self):
        """
        Load all saved extensions in memory.
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

    def get_computable_extensions(self):
        """
        Get all extensions that can be computed by the analyzer.
        """
        return get_available_analyzer_extensions()

    def get_default_extension_params(self, extension_name: str):
        """
        Get the default params for an extension.

        Parameters
        ----------
        extension_name: str
            The extension name

        Returns
        -------
        default_params: dict
            The default parameters for the extension
        """
        return get_default_analyzer_extension_params(extension_name)


global _possible_extensions
_possible_extensions = []

global _extension_children
_extension_children = {}


def _get_children_dependencies(extension_name):
    """
    Extension classes have a `depend_on` attribute to declare on which class they
    depend. For instance "templates" depend on "waveforms". "waveforms depends on "random_spikes".

    This function is making the reverse way : get all children that depend of a
    particular extension.

    This is recurssive so this includes : children and so grand children and grand grand children

    This function is usefull for deleting on recompute.
    For instance recompute the "waveforms" need to delete "template"
    This make sens if "ms_before" is change in "waveforms" because the template also depends
    on this parameters.
    """
    names = []
    children = _extension_children[extension_name]
    for child in children:
        if child not in names:
            names.append(child)
        grand_children = _get_children_dependencies(child)
        names.extend(grand_children)
    return list(names)


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
    assert issubclass(extension_class, AnalyzerExtension)
    assert extension_class.extension_name is not None, "extension_name must not be None"
    global _possible_extensions

    already_registered = any(extension_class is ext for ext in _possible_extensions)
    if not already_registered:
        assert all(
            extension_class.extension_name != ext.extension_name for ext in _possible_extensions
        ), "Extension name already exists"

        _possible_extensions.append(extension_class)

        # create the children dpendencies to be able to delete on re-compute
        _extension_children[extension_class.extension_name] = []
        for parent_name in extension_class.depend_on:
            if "|" in parent_name:
                for name in parent_name.split("|"):
                    _extension_children[name].append(extension_class.extension_name)
            else:
                _extension_children[parent_name].append(extension_class.extension_name)


def get_extension_class(extension_name: str, auto_import=True):
    """
    Get extension class from name and check if registered.

    Parameters
    ----------
    extension_name: str
        The extension name.
    auto_import: bool, default: True
        Auto import the module if the extension class is not registered yet.

    Returns
    -------
    ext_class:
        The class of the extension.
    """
    global _possible_extensions
    extensions_dict = {ext.extension_name: ext for ext in _possible_extensions}

    if extension_name not in extensions_dict:
        if extension_name in _builtin_extensions:
            module = _builtin_extensions[extension_name]
            if auto_import:
                imported_module = importlib.import_module(module)
                extensions_dict = {ext.extension_name: ext for ext in _possible_extensions}
            else:
                raise ValueError(
                    f"Extension '{extension_name}' is not registered, please import related module before use: 'import {module}'"
                )
        else:
            raise ValueError(f"Extension '{extension_name}' is unknown maybe this is an external extension or a typo.")

    ext_class = extensions_dict[extension_name]
    return ext_class


def get_available_analyzer_extensions():
    """
    Get all extensions that can be computed by the analyzer.
    """
    return list(_builtin_extensions.keys())


def get_default_analyzer_extension_params(extension_name: str):
    """
    Get the default params for an extension.

    Parameters
    ----------
    extension_name: str
        The extension name

    Returns
    -------
    default_params: dict
        The default parameters for the extension
    """
    import inspect

    extension_class = get_extension_class(extension_name)

    sig = inspect.signature(extension_class._set_params)
    default_params = {
        k: v.default for k, v in sig.parameters.items() if k != "self" and v.default != inspect.Parameter.empty
    }

    return default_params


class AnalyzerExtension:
    """
    This the base class to extend the SortingAnalyzer.
    It can handle persistency to disk for any computations related to:

    For instance:
      * waveforms
      * principal components
      * spike amplitudes
      * quality metrics

    Possible extension can be registered on-the-fly at import time with register_result_extension() mechanism.
    It also enables any custom computation on top of the SortingAnalyzer to be implemented by the user.

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

    All AnalyzerExtension will have a function associate for instance (this use the function_factory):
    compute_unit_location(sorting_analyzer, ...) will be equivalent to sorting_analyzer.compute("unit_location", ...)


    """

    extension_name = None
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    nodepipeline_variables = None
    need_job_kwargs = False

    def __init__(self, sorting_analyzer):
        self._sorting_analyzer = weakref.ref(sorting_analyzer)

        self.params = None
        self.data = dict()

    #######
    # This 3 methods must be implemented in the subclass!!!
    # See DummyAnalyzerExtension in test_sortinganalyzer.py as a simple example
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
        # comptute_unit_location(sorting_analyzer, ...) <> sorting_analyzer.compute("unit_location", ...)
        # this also make backcompatibility
        # comptute_unit_location(we, ...)

        class FuncWrapper:
            def __init__(self, extension_name):
                self.extension_name = extension_name

            def __call__(self, sorting_analyzer, load_if_exists=None, *args, **kwargs):
                from .waveforms_extractor_backwards_compatibility import MockWaveformExtractor

                if isinstance(sorting_analyzer, MockWaveformExtractor):
                    # backward compatibility with WaveformsExtractor
                    sorting_analyzer = sorting_analyzer.sorting_analyzer

                if not isinstance(sorting_analyzer, SortingAnalyzer):
                    raise ValueError(f"compute_{self.extension_name}() needs a SortingAnalyzer instance")

                if load_if_exists is not None:
                    # backward compatibility with "load_if_exists"
                    warnings.warn(
                        f"compute_{cls.extension_name}(..., load_if_exists=True/False) is kept for backward compatibility but should not be used anymore"
                    )
                    assert isinstance(load_if_exists, bool)
                    if load_if_exists:
                        ext = sorting_analyzer.get_extension(self.extension_name)
                        return ext

                ext = sorting_analyzer.compute(cls.extension_name, *args, **kwargs)
                return ext.get_data()

        func = FuncWrapper(cls.extension_name)
        func.__doc__ = cls.__doc__
        return func

    @property
    def sorting_analyzer(self):
        # Important : to avoid the SortingAnalyzer referencing a AnalyzerExtension
        # and AnalyzerExtension referencing a SortingAnalyzer we need a weakref.
        # Otherwise the garbage collector is not working properly.
        # and so the SortingAnalyzer + its recording are still alive even after deleting explicitly
        # the SortingAnalyzer which makes it impossible to delete the folder when using memmap.
        sorting_analyzer = self._sorting_analyzer()
        if sorting_analyzer is None:
            raise ValueError(f"The extension {self.extension_name} has lost its SortingAnalyzer")
        return sorting_analyzer

    # some attribuites come from sorting_analyzer
    @property
    def format(self):
        return self.sorting_analyzer.format

    @property
    def sparsity(self):
        return self.sorting_analyzer.sparsity

    @property
    def folder(self):
        return self.sorting_analyzer.folder

    def _get_binary_extension_folder(self):
        extension_folder = self.folder / "extensions" / self.extension_name
        return extension_folder

    def _get_zarr_extension_group(self, mode="r+"):
        zarr_root = self.sorting_analyzer._get_zarr_root(mode=mode)
        extension_group = zarr_root["extensions"][self.extension_name]
        return extension_group

    @classmethod
    def load(cls, sorting_analyzer):
        ext = cls(sorting_analyzer)
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
            # but this make the garbage complicated when a data is hold by a plot but the o SortingAnalyzer is delete
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

    def copy(self, new_sorting_analyzer, unit_ids=None):
        # alessio : please note that this also replace the old select_units!!!
        new_extension = self.__class__(new_sorting_analyzer)
        new_extension.params = self.params.copy()
        if unit_ids is None:
            new_extension.data = self.data
        else:
            new_extension.data = self._select_extension_data(unit_ids)
        new_extension.save()
        return new_extension

    def run(self, save=True, **kwargs):
        if save and not self.sorting_analyzer.is_read_only():
            # this also reset the folder or zarr group
            self._save_params()
            self._save_importing_provenance()

        self._run(**kwargs)

        if save and not self.sorting_analyzer.is_read_only():
            self._save_data(**kwargs)

    def save(self, **kwargs):
        self._save_params()
        self._save_importing_provenance()
        self._save_data(**kwargs)

    def _save_data(self, **kwargs):
        if self.format == "memory":
            return

        if self.sorting_analyzer.is_read_only():
            raise ValueError(f"The SortingAnalyzer is read-only saving extension {self.extension_name} is not possible")

        try:
            # pandas is a weak dependency for spikeinterface.core
            import pandas as pd

            HAS_PANDAS = True
        except:
            HAS_PANDAS = False

        if self.format == "binary_folder":

            extension_folder = self._get_binary_extension_folder()
            for ext_data_name, ext_data in self.data.items():
                if isinstance(ext_data, dict):
                    with (extension_folder / f"{ext_data_name}.json").open("w") as f:
                        json.dump(ext_data, f)
                elif isinstance(ext_data, np.ndarray):
                    data_file = extension_folder / f"{ext_data_name}.npy"
                    if isinstance(ext_data, np.memmap) and data_file.exists():
                        # important some SortingAnalyzer like ComputeWaveforms already run the computation with memmap
                        # so no need to save theses array
                        pass
                    else:
                        np.save(data_file, ext_data)
                elif HAS_PANDAS and isinstance(ext_data, pd.DataFrame):
                    ext_data.to_csv(extension_folder / f"{ext_data_name}.csv", index=True)
                else:
                    try:
                        with (extension_folder / f"{ext_data_name}.pkl").open("wb") as f:
                            pickle.dump(ext_data, f)
                    except:
                        raise Exception(f"Could not save {ext_data_name} as extension data")
        elif self.format == "zarr":

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
                elif HAS_PANDAS and isinstance(ext_data, pd.DataFrame):
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
        Delete the extension in a folder (binary or zarr) and create an empty one.
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

        if self.sorting_analyzer.is_read_only():
            return

        if save:
            self._save_params()
            self._save_importing_provenance()

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

    def _save_importing_provenance(self):
        # this saves the class info, this is not uselful at the moment but could be useful in future
        # if some class changes the data model and if we need to make backwards compatibility
        # we have the same machanism in base.py for recording and sorting

        info = retrieve_importing_provenance(self.__class__)
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            extension_folder.mkdir(exist_ok=True, parents=True)
            info_file = extension_folder / "info.json"
            info_file.write_text(json.dumps(info, indent=4), encoding="utf8")
        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r+")
            extension_group.attrs["info"] = info

    def get_pipeline_nodes(self):
        assert (
            self.use_nodepipeline
        ), "AnalyzerExtension.get_pipeline_nodes() must be called only when use_nodepipeline=True"
        return self._get_pipeline_nodes()

    def get_data(self, *args, **kwargs):
        assert len(self.data) > 0, f"You must run the extension {self.extension_name} before retrieving data"
        return self._get_data(*args, **kwargs)


# this is a hardcoded list to to improve error message and auto_import mechanism
# this is important because extension are registered when the submodule is imported
_builtin_extensions = {
    # from core
    "random_spikes": "spikeinterface.core",
    "waveforms": "spikeinterface.core",
    "templates": "spikeinterface.core",
    # "fast_templates": "spikeinterface.core",
    "noise_levels": "spikeinterface.core",
    # from postprocessing
    "amplitude_scalings": "spikeinterface.postprocessing",
    "correlograms": "spikeinterface.postprocessing",
    "isi_histograms": "spikeinterface.postprocessing",
    "principal_components": "spikeinterface.postprocessing",
    "spike_amplitudes": "spikeinterface.postprocessing",
    "spike_locations": "spikeinterface.postprocessing",
    "template_metrics": "spikeinterface.postprocessing",
    "template_similarity": "spikeinterface.postprocessing",
    "unit_locations": "spikeinterface.postprocessing",
    # from quality metrics
    "quality_metrics": "spikeinterface.qualitymetrics",
}
