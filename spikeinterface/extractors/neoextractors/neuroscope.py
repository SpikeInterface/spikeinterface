import warnings
from pathlib import Path
from typing import Union, Optional

import numpy as np

from spikeinterface.core import (BaseSorting, BaseSortingSegment)
from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor

try:
    from lxml import etree as et
    HAVE_LXML = True
except ImportError:
    HAVE_LXML = False

PathType = Union[str, Path]
OptionalPathType = Optional[PathType]


class NeuroScopeRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from neuroscope
    Ref: http://neuroscope.sourceforge.net

    Based on :py:class:`neo.rawio.NeuroScopeRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'NeuroScopeRawIO'
    name = "neuroscope"

    def __init__(self, file_path, stream_id=None, stream_name=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, 
                                           stream_name=stream_name,
                                           all_annotations=all_annotations, 
                                           **neo_kwargs)
        self._kwargs.update(dict(file_path=str(file_path)))

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {'filename': str(file_path)}
        return neo_kwargs


class NeuroScopeSortingExtractor(BaseSorting):
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
    resfile_path : PathType
        Optional. Path to a particular .res text file. If given, only the single .res file
        (and the respective .clu file) are loaded
    clufile_path : PathType
        Optional. Path to a particular .clu text file. If given, only the single .clu file
        (and the respective .res file) are loaded
    keep_mua_units : bool
        Optional. Whether or not to return sorted spikes from multi-unit activity. Defaults to True.
    exclude_shanks : list
        Optional. List of indices to ignore. The set of all possible indices is chosen by default, extracted as the
        final integer of all the .res.%i and .clu.%i pairs.
    xml_file_path : PathType, optional
        Path to the .xml file referenced by this sorting.
    """

    extractor_name = "NeuroscopeSortingExtractor"
    installed = HAVE_LXML
    installation_mesg = "Please install lxml to use this extractor!"
    name = "neuroscope"

    def __init__(
        self,
        folder_path: OptionalPathType = None,
        resfile_path: OptionalPathType = None,
        clufile_path: OptionalPathType = None,
        keep_mua_units: bool = True,
        exclude_shanks: Optional[list] = None,
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
            res_files = [resfile_path]
            clu_files = [clufile_path]
        else:
            assert folder_path is not None, "Either pass resfile_path and clufile_path, or folder_path!"
            folder_path = Path(folder_path)
            assert folder_path.is_dir(), "The folder_path must be a directory!"

            res_files = sorted([f for f in folder_path.iterdir() if f.is_file() and ".res" in f.suffixes and not
                                f.name.endswith("~")])
            clu_files = sorted([f for f in folder_path.iterdir() if f.is_file() and ".clu" in f.suffixes and not
                                f.name.endswith("~")])

            assert len(res_files) > 0 or len(clu_files) > 0, \
                "No .res or .clu files found in the folder_path!"

            folder_path_passed = True  # flag for setting kwargs for proper dumping

        if exclude_shanks is not None:  # dumping checks do not like having an empty list as default
            assert all([isinstance(x, (int, np.integer)) and x >= 0 for x in
                        exclude_shanks]), "Optional argument 'exclude_shanks' must contain positive integers only!"
        else:
            exclude_shanks = []
        xml_file_path = _handle_xml_file_path(
            folder_path=folder_path, initial_xml_file_path=xml_file_path)
        xml_root = et.parse(str(xml_file_path)).getroot()
        sampling_frequency = float(xml_root.find(
            'acquisitionSystem').find('samplingRate').text)

        if len(res_files) > 1:
            res_ids = [int(x.suffix[1:]) for x in res_files]
            clu_ids = [int(x.suffix[1:]) for x in clu_files]
            assert sorted(res_ids) == sorted(
                clu_ids), "Unmatched .clu.%i and .res.%i files detected!"
            if any([x not in res_ids for x in exclude_shanks]):
                warnings.warn(
                    "Detected indices in exclude_shanks that are not in the directory! These will be ignored.")
            shank_ids = res_ids
        else:
            shank_ids = None

        resfile_names = [x.name[:x.name.find('.res')] for x in res_files]
        clufile_names = [x.name[:x.name.find('.clu')] for x in clu_files]
        assert np.all(r == c for (r, c) in zip(resfile_names, clufile_names)), \
            "Some of the .res.%i and .clu.%i files do not share the same name!"

        all_unit_ids = []
        all_spiketrains = []
        all_unit_shank_ids = []
        for i, (res_file, clu_file) in enumerate(zip(res_files, clu_files)):
            if shank_ids is not None:
                shank_id = shank_ids[i]
                if shank_id in exclude_shanks:
                    continue

            with open(res_file) as f:
                res = np.array([int(line) for line in f], np.int64)
            with open(clu_file) as f:
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
                if len(all_unit_ids) > 0:
                    last_unit_id = all_unit_ids[-1]
                else:
                    last_unit_id = 0
                new_unit_ids = [u + last_unit_id for u in unit_map.values()]
                all_unit_ids += new_unit_ids
                for s_id in unit_map:
                    all_spiketrains.append(res[(clu == s_id).nonzero()])

                if shank_ids is not None:
                    all_unit_shank_ids += [shank_id] * len(new_unit_ids)

        BaseSorting.__init__(self, sampling_frequency=sampling_frequency,
                             unit_ids=all_unit_ids)
        self.add_sorting_segment(
            NeuroScopeSortingSegment(all_unit_ids, all_spiketrains))

        self.extra_requirements.append('lxml')

        # set "group" property based on shank ids
        if len(all_unit_shank_ids) > 0:
            self.set_property("group", all_unit_shank_ids)

        if folder_path_passed:
            self._kwargs = dict(
                folder_path=str(folder_path.absolute()),
                resfile_path=None,
                clufile_path=None,
                keep_mua_units=keep_mua_units,
                exclude_shanks=exclude_shanks,
                xml_file_path=xml_file_path
            )
        else:
            self._kwargs = dict(
                folder_path=None,
                resfile_path=str(resfile_path.absolute()),
                clufile_path=str(clufile_path.absolute()),
                keep_mua_units=keep_mua_units,
                exclude_shanks=exclude_shanks,
                xml_file_path=xml_file_path
            )


class NeuroScopeSortingSegment(BaseSortingSegment):
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


def _find_xml_file_path(folder_path: PathType):
    xml_files = [f for f in folder_path.iterdir() if f.is_file()
                 if f.suffix == ".xml"]
    assert any(xml_files), "No .xml files found in the folder_path."
    assert len(
        xml_files) == 1, "More than one .xml file found in the folder_path! Specify xml_file_path."
    xml_file_path = xml_files[0]
    return xml_file_path


def _handle_xml_file_path(folder_path: PathType, initial_xml_file_path: PathType):
    if initial_xml_file_path is None:
        xml_file_path = _find_xml_file_path(folder_path=folder_path)
    else:
        assert Path(initial_xml_file_path).is_file(
        ), f".xml file ({initial_xml_file_path}) not found!"
        xml_file_path = initial_xml_file_path
    return xml_file_path


read_neuroscope_recording = define_function_from_class(source_class=NeuroScopeRecordingExtractor,
                                                   name="read_neuroscope_recording")
read_neuroscope_sorting = define_function_from_class(source_class=NeuroScopeSortingExtractor,
                                                 name="read_neuroscope_sorting")


def read_neuroscope(file_path, stream_id=None, keep_mua_units=False,
                    exclude_shanks=None, load_recording=True, load_sorting=False):
    """
    Read neuroscope recording and sorting.
    This function assumses that all .res and .clu files are in the same folder as
    the .xml file.

    Parameters
    ----------
    file_path: str
        The xml file.
    stream_id: str or None
    keep_mua_units: bool
        Optional. Whether or not to return sorted spikes from multi-unit activity. Defaults to True.
    exclude_shanks: list
        Optional. List of indices to ignore. The set of all possible indices is chosen by default, extracted as the
        final integer of all the .res. % i and .clu. % i pairs.
    load_recording: bool
        If True, the recording is loaded (default True)
    load_sorting: bool
        If True, the sorting is loaded (default False)
    """
    outputs = ()
    # TODO add checks for recording and sorting existence
    if load_recording:
        recording = NeuroScopeRecordingExtractor(file_path=file_path, stream_id=stream_id)
        outputs = outputs + (recording,)
    if load_sorting:
        folder_path = Path(file_path).parent
        sorting = NeuroScopeSortingExtractor(folder_path=folder_path, keep_mua_units=keep_mua_units,
                                             exclude_shanks=exclude_shanks)
        outputs = outputs + (sorting,)

    if len(outputs) == 1:
        outputs = outputs[0]

    return outputs
