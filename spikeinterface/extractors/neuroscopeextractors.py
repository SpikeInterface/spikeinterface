import numpy as np
import re
import warnings
from pathlib import Path
from typing import Union, Optional

from spikeinterface.core import (BaseSorting, BaseSortingSegment, UnitsAggregationSorting)

try:
    from lxml import etree as et

    HAVE_LXML = True
except ImportError:
    HAVE_LXML = False

PathType = Union[str, Path]
OptionalPathType = Optional[PathType]
DtypeType = Union[str, np.dtype, None]


class NeuroScopeBaseSortingExtractor(BaseSorting):
    """
    Extracts spiking information from pair of .res and .clu files.

    The .res is a text file with a sorted list of spiketimes from all units displayed in sample (integer '%i') units.
    The .clu file is a file with one more row than the .res with the first row corresponding to
    the total number of unique ids in the file (and may exclude 0 & 1 from this count)
    with the rest of the rows indicating which unit id the corresponding entry in the
    .res file refers to.

    In the original Neuroscope format:
        Unit ID 0 is the cluster of unsorted spikes (noise).
        Unit ID 1 is a cluster of multi-unit spikes.

    The function defaults to returning multi-unit activity as the first index, and ignoring unsorted noise.
    To return only the fully sorted units, set keep_mua_units=False.

    The sorting extractor always returns unit IDs from 1, ..., number of chosen clusters.

    Parameters
    ----------
    resfile_path : PathType
        Optional. Path to a particular .res text file.
    clufile_path : PathType
        Optional. Path to a particular .clu text file.
    folder_path : PathType
        Optional. Path to the collection of .res and .clu text files. Will auto-detect format.
    keep_mua_units : bool
        Optional. Whether or not to return sorted spikes from multi-unit activity. Defaults to True.
    xml_file_path : PathType, optional
        Path to the .xml file referenced by this sorting.
    """

    extractor_name = "NeuroscopeSortingExtractor"
    installed = HAVE_LXML
    installation_mesg = "Please install lxml to use this extractor!"

    def __init__(
        self,
        resfile_path: OptionalPathType = None,
        clufile_path: OptionalPathType = None,
        folder_path: OptionalPathType = None,
        keep_mua_units: bool = True,
        xml_file_path: OptionalPathType = None,
    ):
        assert self.installed, self.installation_mesg
        assert not (folder_path is None and resfile_path is None and clufile_path is None), \
            "Either pass a single folder_path location, or a pair of resfile_path and clufile_path! None received."

        if resfile_path is not None:
            assert clufile_path is not None, "If passing resfile_path or clufile_path, both are required!"
            resfile_path = Path(resfile_path)
            clufile_path = Path(clufile_path)
            assert resfile_path.is_file() and clufile_path.is_file(), \
                f"The resfile_path ({resfile_path}) and clufile_path ({clufile_path}) must be .res and .clu files!"

            assert folder_path is None, "Pass either a single folder_path location, " \
                                        "or a pair of resfile_path and clufile_path! All received."
            folder_path_passed = False
            folder_path = resfile_path.parent
        else:
            assert folder_path is not None, "Either pass resfile_path and clufile_path, or folder_path!"
            folder_path = Path(folder_path)
            assert folder_path.is_dir(), "The folder_path must be a directory!"

            res_files = _get_single_files(
                folder_path=folder_path, suffix=".res")
            clu_files = _get_single_files(
                folder_path=folder_path, suffix=".clu")

            assert len(res_files) > 0 or len(clu_files) > 0, \
                "No .res or .clu files found in the folder_path!"
            assert len(res_files) == 1 and len(clu_files) == 1, \
                "NeuroscopeSortingExtractor expects a single pair of .res and .clu files in the folder_path. " \
                "For multiple .res and .clu files, use the NeuroscopeMultiSortingExtractor instead."

            folder_path_passed = True  # flag for setting kwargs for proper dumping
            resfile_path = res_files[0]
            clufile_path = clu_files[0]

        res_sorting_name = resfile_path.name[:resfile_path.name.find('.res')]
        clu_sorting_name = clufile_path.name[:clufile_path.name.find('.clu')]

        assert res_sorting_name == clu_sorting_name, "The .res and .clu files do not share the same name! " \
                                                     f"{res_sorting_name}  -- {clu_sorting_name}"

        xml_file_path = handle_xml_file_path(
            folder_path=folder_path, initial_xml_file_path=xml_file_path)
        xml_root = et.parse(str(xml_file_path)).getroot()
        sampling_frequency = float(xml_root.find(
            'acquisitionSystem').find('samplingRate').text)

        with open(resfile_path) as f:
            res = np.array([int(line) for line in f], np.int64)
        with open(clufile_path) as f:
            clu = np.array([int(line) for line in f], np.int64)

        n_spikes = len(res)
        if n_spikes > 0:
            # Extract the number of unique IDs from the first line of the clufile then remove it from the list
            n_clu = clu[0]
            clu = np.delete(clu, 0)
            unique_ids = np.unique(clu)
            assert len(unique_ids) == n_clu, (
                "First value of .clu file ({clufile_path}) does not match number of unique IDs!"
            )
            unit_map = dict(zip(unique_ids, list(range(1, n_clu + 1))))

            if 0 in unique_ids:
                unit_map.pop(0)
            if not keep_mua_units and 1 in unique_ids:
                unit_map.pop(1)
            unit_ids = np.array([u for u in unit_map.values()])
            spiketrains = []
            for s_id in unit_map:
                spiketrains.append(res[(clu == s_id).nonzero()])
        else:
            warnings.warn(f"No spikes found for the given files:"
                          f"{res_sorting_name} - {clu_sorting_name}")

        BaseSorting.__init__(self, sampling_frequency=sampling_frequency,
                             unit_ids=unit_ids)
        self.add_sorting_segment(NeuroscopeSortingSegment(unit_ids, spiketrains))

        if folder_path_passed:
            self._kwargs = dict(
                resfile_path=None,
                clufile_path=None,
                folder_path=str(folder_path.absolute()),
                keep_mua_units=keep_mua_units,
            )
        else:
            self._kwargs = dict(
                resfile_path=str(resfile_path.absolute()),
                clufile_path=str(clufile_path.absolute()),
                folder_path=None,
                keep_mua_units=keep_mua_units,
            )


class NeuroscopeSortingSegment(BaseSortingSegment):
    def __init__(self, unit_ids, spiketrains):
        BaseSortingSegment.__init__(self)
        self._unit_ids = list(unit_ids)
        self._spiketrains = spiketrains

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        unit_index = self._unit_ids.index(unit_id)
        times = self._spiketrains[unit_index]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times


class NeuroScopeSortingExtractor(UnitsAggregationSorting):
    """
    Extracts spiking information from an arbitrary number of .res.%i and .clu.%i files in the general folder path.

    The .res is a text file with a sorted list of spiketimes from all units displayed in sample (integer '%i') units.
    The .clu file is a file with one more row than the .res with the first row corresponding to the total number of
    unique ids in the file (and may exclude 0 & 1 from this count)
    with the rest of the rows indicating which unit id the corresponding entry in the .res file refers to.
    The group id is loaded as unit property 'group'.

    In the original Neuroscope format:
        Unit ID 0 is the cluster of unsorted spikes (noise).
        Unit ID 1 is a cluster of multi-unit spikes.

    The function defaults to returning multi-unit activity as the first index, and ignoring unsorted noise.
    To return only the fully sorted units, set keep_mua_units=False.

    The sorting extractor always returns unit IDs from 1, ..., number of chosen clusters.

    Parameters
    ----------
    folder_path : str
        Optional. Path to the collection of .res and .clu text files. Will auto-detect format.
    keep_mua_units : bool
        Optional. Whether or not to return sorted spikes from multi-unit activity. Defaults to True.
    exclude_shanks : list
        Optional. List of indices to ignore. The set of all possible indices is chosen by default, extracted as the
        final integer of all the .res.%i and .clu.%i pairs.
    xml_file_path : PathType, optional
        Path to the .xml file referenced by this sorting.
    """

    extractor_name = "NeuroscopeMultiSortingExtractor"
    installed = HAVE_LXML
    installation_mesg = "Please install lxml to use this extractor!"

    def __init__(
        self,
        folder_path: PathType,
        keep_mua_units: bool = True,
        exclude_shanks: Optional[list] = None,
        xml_file_path: OptionalPathType = None,
    ):
        assert self.installed, self.installation_mesg

        folder_path = Path(folder_path)

        if exclude_shanks is not None:  # dumping checks do not like having an empty list as default
            assert all([isinstance(x, (int, np.integer)) and x >= 0 for x in
                        exclude_shanks]), "Optional argument 'exclude_shanks' must contain positive integers only!"
            exclude_shanks_passed = True
        else:
            exclude_shanks = []
            exclude_shanks_passed = False
        xml_file_path = handle_xml_file_path(
            folder_path=folder_path, initial_xml_file_path=xml_file_path)
        xml_root = et.parse(str(xml_file_path)).getroot()
        self._sampling_frequency = float(xml_root.find(
            'acquisitionSystem').find('samplingRate').text)

        res_files = _get_shank_files(folder_path=folder_path, suffix=".res")
        clu_files = _get_shank_files(folder_path=folder_path, suffix=".clu")

        assert len(res_files) > 0 or len(
            clu_files) > 0, "No .res or .clu files found in the folder_path!"
        assert len(res_files) == len(clu_files)

        res_ids = [int(x.suffix[1:]) for x in res_files]
        clu_ids = [int(x.suffix[1:]) for x in clu_files]
        assert sorted(res_ids) == sorted(
            clu_ids), "Unmatched .clu.%i and .res.%i files detected!"
        if any([x not in res_ids for x in exclude_shanks]):
            warnings.warn(
                "Detected indices in exclude_shanks that are not in the directory! These will be ignored.")

        resfile_names = [x.name[:x.name.find('.res')] for x in res_files]
        clufile_names = [x.name[:x.name.find('.clu')] for x in clu_files]
        assert np.all(r == c for (r, c) in zip(resfile_names, clufile_names)), \
            "Some of the .res.%i and .clu.%i files do not share the same name!"
        sorting_name = resfile_names[0]

        all_shanks_list_se = []
        unit_shank_ids = []
        for shank_id in list(set(res_ids) - set(exclude_shanks)):
            nse_args = dict(
                resfile_path=folder_path / f"{sorting_name}.res.{shank_id}",
                clufile_path=folder_path / f"{sorting_name}.clu.{shank_id}",
                keep_mua_units=keep_mua_units,
                xml_file_path=xml_file_path,
            )
            shank_sorting = NeuroScopeBaseSortingExtractor(**nse_args)
            all_shanks_list_se.append(shank_sorting)
            unit_shank_ids.append([shank_id] * len(shank_sorting.unit_ids))

        UnitsAggregationSorting.__init__(self, sorting_list=all_shanks_list_se)

        # set "group" property based on shank ids
        self.set_property("group", unit_shank_ids)

        self._kwargs = dict(
            folder_path=str(folder_path.absolute()),
            keep_mua_units=keep_mua_units,
            exclude_shanks=exclude_shanks,
        )


def _get_single_files(folder_path: Path, suffix: str):
    return [
        f for f in folder_path.iterdir() if f.is_file() and suffix in f.suffixes and not f.name.endswith("~")
        and len(f.suffixes) == 1
    ]


def _get_shank_files(folder_path: Path, suffix: str):
    return [
        f for f in folder_path.iterdir() if f.is_file() and suffix in f.suffixes
        and re.search(r"\d+$", f.name) is not None and len(f.suffixes) == 2
    ]


def find_xml_file_path(folder_path: PathType):
    xml_files = [f for f in folder_path.iterdir() if f.is_file()
                 if f.suffix == ".xml"]
    assert any(xml_files), "No .xml files found in the folder_path."
    assert len(
        xml_files) == 1, "More than one .xml file found in the folder_path! Specify xml_file_path."
    xml_file_path = xml_files[0]
    return xml_file_path


def handle_xml_file_path(folder_path: PathType, initial_xml_file_path: PathType):
    if initial_xml_file_path is None:
        xml_file_path = find_xml_file_path(folder_path=folder_path)
    else:
        assert Path(initial_xml_file_path).is_file(
        ), f".xml file ({initial_xml_file_path}) not found!"
        xml_file_path = initial_xml_file_path
    return xml_file_path


def read_neuroscope_sorting(*args, **kwargs):
    sorting = NeuroScopeSortingExtractor(*args, **kwargs)
    return sorting


read_neuroscope_sorting.__doc__ = NeuroScopeSortingExtractor.__doc__
