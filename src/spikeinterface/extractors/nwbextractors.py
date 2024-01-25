from __future__ import annotations
from pathlib import Path
from typing import Union, List, Optional, Literal, Dict, BinaryIO
import warnings

import numpy as np

from spikeinterface import get_global_tmp_folder
from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


def import_lazily():
    "Makes annotations / typing available lazily"
    global NWBFile, ElectricalSeries, Units, NWBHDF5IO
    from pynwb import NWBFile
    from pynwb.ecephys import ElectricalSeries
    from pynwb.misc import Units
    from pynwb import NWBHDF5IO


def retrieve_electrical_series(nwbfile: NWBFile, electrical_series_name: Optional[str] = None) -> ElectricalSeries:
    """
    Get an ElectricalSeries object from an NWBFile.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWBFile object from which to extract the ElectricalSeries.
    electrical_series_name : str, default: None
        The name of the ElectricalSeries to extract. If not specified, it will return the first found ElectricalSeries
        if there's only one; otherwise, it raises an error.

    Returns
    -------
    ElectricalSeries
        The requested ElectricalSeries object.

    Raises
    ------
    ValueError
        If no acquisitions are found in the NWBFile or if multiple acquisitions are found but no electrical_series_name
        is provided.
    AssertionError
        If the specified electrical_series_name is not present in the NWBFile.
    """
    from pynwb.ecephys import ElectricalSeries

    electrical_series_dict: Dict[str, ElectricalSeries] = {}

    for item in nwbfile.all_children():
        if isinstance(item, ElectricalSeries):
            # remove data and skip first "/"
            electrical_series_path = item.data.name.replace("/data", "")[1:]
            electrical_series_dict[electrical_series_path] = item

    if electrical_series_name is not None:
        if electrical_series_name not in electrical_series_dict:
            raise ValueError(f"{electrical_series_name} not found in the NWBFile. ")
        electrical_series = electrical_series_dict[electrical_series_name]
    else:
        electrical_series_list = list(electrical_series_dict.keys())
        if len(electrical_series_list) > 1:
            raise ValueError(
                f"More than one acquisition found! You must specify 'electrical_series_name'. \n"
                f"Options in current file are: {[e for e in electrical_series_list]}"
            )
        if len(electrical_series_list) == 0:
            raise ValueError("No acquisitions found in the .nwb file.")
        electrical_series = electrical_series_dict[electrical_series_list[0]]

    return electrical_series


def retrieve_unit_table(nwbfile: NWBFile, unit_table_name: Optional[str] = None) -> Units:
    """
    Get an Units object from an NWBFile.
    Units tables can be either the main unit table (nwbfile.units) or in the processing module.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWBFile object from which to extract the Units.
    unit_table_name : str, default: None
        The path of the Units to extract. If not specified, it will return the first found Units
        if there's only one; otherwise, it raises an error.

    Returns
    -------
    Units
        The requested Units object.

    Raises
    ------
    ValueError
        If no unit tables are found in the NWBFile or if multiple unit tables are found but no unit_table_name
        is provided.
    AssertionError
        If the specified unit_table_name is not present in the NWBFile.
    """
    from pynwb.misc import Units

    unit_table_dict: Dict[str:Units] = {}

    for item in nwbfile.all_children():
        if isinstance(item, Units):
            # retrieve name of "id" column and skip first "/"
            electrical_series_path = item.id.data.name.replace("/id", "")[1:]
            unit_table_dict[electrical_series_path] = item

    if unit_table_name is not None:
        if unit_table_name not in unit_table_dict:
            raise ValueError(f"{unit_table_name} not found in the NWBFile. ")
        unit_table = unit_table_dict[unit_table_name]
    else:
        unit_table_list: List[Units] = list(unit_table_dict.keys())

        if len(unit_table_list) > 1:
            raise ValueError(
                f"More than one unit table found! You must specify 'unit_table_list_name'. \n"
                f"Options in current file are: {[e for e in unit_table_list]}"
            )
        if len(unit_table_list) == 0:
            raise ValueError("No unit table found in the .nwb file.")
        unit_table = unit_table_dict[unit_table_list[0]]

    return unit_table


def read_file_from_backend(
    *,
    file_path: str | Path | None,
    file: BinaryIO | None = None,
    stream_mode: Literal["ffspec", "ros3", "remfile"] | None = None,
    cache: bool = False,
    stream_cache_path: str | Path | None = None,
):
    import h5py

    if stream_mode == "fsspec":
        import fsspec
        from fsspec.implementations.cached import CachingFileSystem

        assert file_path is not None, "file_path must be specified when using stream_mode='fsspec'"

        fsspec_file_system = fsspec.filesystem("http")

        if cache:
            stream_cache_path = stream_cache_path if stream_cache_path is not None else str(get_global_tmp_folder())
            caching_file_system = CachingFileSystem(
                fs=fsspec_file_system,
                cache_storage=str(stream_cache_path),
            )
            ffspec_file = caching_file_system.open(path=file_path, mode="rb")
        else:
            ffspec_file = fsspec_file_system.open(file_path, "rb")

        hdf5_file = h5py.File(name=ffspec_file, mode="r")

    elif stream_mode == "ros3":
        assert file_path is not None, "file_path must be specified when using stream_mode='ros3'"

        drivers = h5py.registered_drivers()
        assertion_msg = "ROS3 support not enbabled, use: install -c conda-forge h5py>=3.2 to enable streaming"
        assert "ros3" in drivers, assertion_msg
        hdf5_file = h5py.File(name=file_path, mode="r", driver="ros3")

    elif stream_mode == "remfile":
        import remfile

        assert file_path is not None, "file_path must be specified when using stream_mode='remfile'"
        rfile = remfile.File(file_path)
        hdf5_file = h5py.File(rfile, "r")

    elif file_path is not None:
        file_path = str(Path(file_path).resolve())
        hdf5_file = h5py.File(name=file_path, mode="r")
    else:
        assert file is not None, "Unexpected, file is None"
        hdf5_file = h5py.File(file, "r")

    return hdf5_file


def read_nwbfile(
    *,
    file_path: str | Path | None,
    file: BinaryIO | None = None,
    stream_mode: Literal["ffspec", "ros3", "remfile"] | None = None,
    cache: bool = False,
    stream_cache_path: str | Path | None = None,
) -> NWBFile:
    """
    Read an NWB file and return the NWBFile object.

    Parameters
    ----------
    file_path : Path, str or None
        The path to the NWB file. Either provide this or file.
    file : file-like object or None
        The file-like object to read from. Either provide this or file_path.
    stream_mode : "fsspec" | "ros3" | "remfile" | None, default: None
        The streaming mode to use. If None it assumes the file is on the local disk.
    cache: bool, default: False
        If True, the file is cached in the file passed to stream_cache_path
        if False, the file is not cached.
    stream_cache_path : str or None, default: None
        The path to the cache storage, when default to None it uses the a temporary
        folder.
    Returns
    -------
    nwbfile : NWBFile
        The NWBFile object.

    Raises
    ------
    AssertionError
        If ROS3 support is not enabled.

    Notes
    -----
    This function can stream data from the "fsspec", "ros3" and "rem" protocols.


    Examples
    --------
    >>> nwbfile = read_nwbfile("data.nwb", stream_mode="ros3")
    """
    from pynwb import NWBHDF5IO

    if file_path is not None and file is not None:
        raise ValueError("Provide either file_path or file, not both")
    if file_path is None and file is None:
        raise ValueError("Provide either file_path or file")

    hdf5_file = read_file_from_backend(
        file_path=file_path,
        file=file,
        stream_mode=stream_mode,
        cache=cache,
        stream_cache_path=stream_cache_path,
    )
    io = NWBHDF5IO(file=hdf5_file, mode="r", load_namespaces=True)
    nwbfile = io.read()
    return nwbfile


def find_neurodata_type_from_backend(group, path="", result=None, neurodata_type="ElectricalSeries"):
    """
    Recursively searches for groups with neurodata_type 'ElectricalSeries' in the HDF5 group or file,
    and returns a list with their paths.
    """
    import h5py

    if result is None:
        result = []

    for neurodata_name, value in group.items():
        # Check if it's a group and if it has the neurodata_type
        if isinstance(value, h5py.Group):  # add Zarr
            current_path = f"{path}/{neurodata_name}" if path else neurodata_name
            if value.attrs.get("neurodata_type") == neurodata_type:
                result.append(current_path)
            find_neurodata_type_from_backend(
                value, current_path, result, neurodata_type
            )  # Recursive call for sub-groups
    return result


def extract_time_info_pynwb(electrical_series, samples_for_rate_estimation, load_time_vector=False):
    sampling_frequency = None
    if hasattr(electrical_series, "rate"):
        sampling_frequency = electrical_series.rate

    if hasattr(electrical_series, "starting_time"):
        t_start = electrical_series.starting_time
    else:
        t_start = None

    timestamps = None
    if hasattr(electrical_series, "timestamps"):
        if electrical_series.timestamps is not None:
            timestamps = electrical_series.timestamps
            t_start = electrical_series.timestamps[0]

    # TimeSeries need to have either timestamps or rate
    if sampling_frequency is None:
        sampling_frequency = 1.0 / np.median(np.diff(timestamps[:samples_for_rate_estimation]))

    if load_time_vector and timestamps is not None:
        times_kwargs = dict(time_vector=electrical_series.timestamps)
    else:
        times_kwargs = dict(sampling_frequency=sampling_frequency, t_start=t_start)

    return sampling_frequency, times_kwargs


class NwbRecordingExtractor(BaseRecording):
    """Load an NWBFile as a RecordingExtractor.

    Parameters
    ----------
    file_path: str, Path, or None
        Path to the NWB file or an s3 URL. Use this parameter to specify the file location
        if not using the `file` parameter.
    electrical_series_name: str or None, default: None
        The name of the ElectricalSeries object within the NWB file. This parameter is crucial
        when the NWB file contains multiple ElectricalSeries objects. It helps in identifying
        which specific series to extract data from. If there is only one ElectricalSeries and
        this parameter is not set, that unique series will be used by default.
        If multiple ElectricalSeries are present and this parameter is not set, an error is raised.
        The `electrical_series_name` corresponds to the path within the NWB file, e.g.,
        'acquisition/MyElectricalSeries`.
    file: file-like object or None, default: None
        A file-like object representing the NWB file. Use this parameter if you have an in-memory
        representation of the NWB file instead of a file path.
    load_time_vector: bool, default: False
        If set to True, the time vector is also loaded into the recording object. Useful for
        cases where precise timing information is required.
    samples_for_rate_estimation: int, default: 1000
        The number of timestamp samples used for estimating the sampling rate. This is relevant
        when the 'rate' attribute is not available in the ElectricalSeries.
    stream_mode : "fsspec", "ros3", "remfile", or None, default: None
        Determines the streaming mode for reading the file. Use this for optimized reading from
        different sources, such as local disk or remote servers.
    cache: bool, default: False
        Indicates whether to cache the file locally when using streaming. Caching can improve performance for
        remote files.
    stream_cache_path: str, Path, or None, default: None
        Specifies the local path for caching the file. Relevant only if `cache` is True.
    use_pynwb: bool, default: False
        Uses the pynwb library to read the NWB file. Setting this to False, the default, uses h5py
        to read the file. Using h5py can improve performance by bypassing some of the PyNWB validations.

    Returns
    -------
    recording : NwbRecordingExtractor
        The recording extractor for the NWB file.

    Examples
    --------
    Run on local file:

    >>> from spikeinterface.extractors.nwbextractors import NwbRecordingExtractor
    >>> rec = NwbRecordingExtractor(filepath)

    Run on s3 URL from the DANDI Archive:

    >>> from spikeinterface.extractors.nwbextractors import NwbRecordingExtractor
    >>> from dandi.dandiapi import DandiAPIClient
    >>>
    >>> # get s3 path
    >>> dandiset_id, filepath = "101116", "sub-001/sub-001_ecephys.nwb"
    >>> with DandiAPIClient("https://api-staging.dandiarchive.org/api") as client:
    >>>     asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    >>>     s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    >>>
    >>> rec = NwbRecordingExtractor(s3_url, stream_mode="fsspec", stream_cache_path="cache")
    """

    extractor_name = "NwbRecording"
    mode = "file"
    name = "nwb"
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(
        self,
        file_path: str | Path | None = None,  # provide either this or file
        electrical_series_name: str | None = None,
        load_time_vector: bool = False,
        samples_for_rate_estimation: int = 1000,
        stream_mode: Optional[Literal["fsspec", "ros3", "remfile"]] = None,
        stream_cache_path: str | Path | None = None,
        *,
        file: BinaryIO | None = None,  # file-like - provide either this or file_path
        cache: bool = False,
        use_pynwb: bool = False,
    ):
        try:
            from pynwb.ecephys import ElectrodeGroup
        except ImportError:
            raise ImportError(self.installation_mesg)

        if file_path is not None and file is not None:
            raise ValueError("Provide either file_path or file, not both")
        if file_path is None and file is None:
            raise ValueError("Provide either file_path or file")

        self.file_path = file_path
        self.stream_mode = stream_mode
        self.stream_cache_path = stream_cache_path
        self.electrical_series_name = electrical_series_name

        # extract info
        if use_pynwb:
            try:
                import pynwb
            except ImportError:
                raise ImportError(self.installation_mesg)
            channel_ids, sampling_frequency, dtype, segment_data, times_kwargs = self.extract_info_pynwb(
                file, cache, load_time_vector, samples_for_rate_estimation
            )
        else:
            channel_ids, sampling_frequency, dtype, segment_data, times_kwargs = self.extract_info_backend(
                file, cache, load_time_vector, samples_for_rate_estimation
            )

        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        recording_segment = NwbRecordingSegment(
            electrical_series_data=segment_data,
            times_kwargs=times_kwargs,
        )
        self.add_recording_segment(recording_segment)

        # add properties and info
        if use_pynwb:
            self.add_extractor_info_pynwb()
        else:
            self.add_extractor_info_backend()

        if stream_mode is None and file_path is not None:
            file_path = str(Path(file_path).resolve())

        if stream_mode == "fsspec" and stream_cache_path is not None:
            stream_cache_path = str(Path(self.stream_cache_path).absolute())

        # set serializability bools
        if file is not None:
            # not json serializable if file arg is provided
            self._serializability["json"] = False

        self._kwargs = {
            "file_path": file_path,
            "electrical_series_name": self.electrical_series_name,
            "load_time_vector": load_time_vector,
            "samples_for_rate_estimation": samples_for_rate_estimation,
            "stream_mode": stream_mode,
            "cache": cache,
            "stream_cache_path": stream_cache_path,
            "file": file,
        }

    def extract_info_pynwb(self, file, cache, load_time_vector, samples_for_rate_estimation):
        self._nwbfile = read_nwbfile(
            file_path=self.file_path,
            file=file,
            stream_mode=self.stream_mode,
            cache=cache,
            stream_cache_path=self.stream_cache_path,
        )
        electrical_series = retrieve_electrical_series(self._nwbfile, self.electrical_series_name)
        # The indices in the electrode table corresponding to this electrical series
        electrodes_indices = electrical_series.electrodes.data[:]
        # The table for all the electrodes in the nwbfile
        electrodes_table = self._nwbfile.electrodes

        sampling_frequency, times_kwargs = extract_time_info_pynwb(
            electrical_series=electrical_series,
            samples_for_rate_estimation=samples_for_rate_estimation,
            load_time_vector=load_time_vector,
        )

        # Fill channel properties dictionary from electrodes table
        if "channel_name" in electrodes_table.colnames:
            channel_ids = [
                electrical_series.electrodes["channel_name"][electrodes_index]
                for electrodes_index in electrodes_indices
            ]
        else:
            channel_ids = [electrical_series.electrodes.table.id[x] for x in electrodes_indices]
        electrical_series_data = electrical_series.data
        dtype = electrical_series_data.dtype

        # need this later
        self.electrical_series = electrical_series
        electrodes_table = electrodes_table
        electrodes_indices = electrodes_indices

        return channel_ids, sampling_frequency, dtype, electrical_series_data, times_kwargs

    def extract_info_backend(self, file, cache, load_time_vector, samples_for_rate_estimation):
        hdf5_file = read_file_from_backend(
            file_path=self.file_path,
            file=file,
            stream_mode=self.stream_mode,
            cache=cache,
            stream_cache_path=self.stream_cache_path,
        )

        # If the electrical_series_name is not given, `find_neurodata_type_from_backend` will be called
        # And returns a list with the electrical_series_names available in the file.
        # If there is only one electrical series, the electrical_series_name is set to the name of the series,
        # otherwise an error is raised.
        if self.electrical_series_name is None:
            available_electrical_series = find_neurodata_type_from_backend(hdf5_file, neurodata_type="ElectricalSeries")
            # if electrical_series_name is None:
            if len(available_electrical_series) == 1:
                self.electrical_series_name = available_electrical_series[0]
            else:
                raise ValueError(
                    "Multiple ElectricalSeries found in the file. "
                    "Please specify the 'electrical_series_name' argument:"
                    f"Available options are: {available_electrical_series}."
                )

        # The indices in the electrode table corresponding to this electrical series
        try:
            electrical_series = hdf5_file[self.electrical_series_name]
        except KeyError:
            available_electrical_series = find_neurodata_type_from_backend(hdf5_file, neurodata_type="ElectricalSeries")
            raise ValueError(
                f"{self.electrical_series_name} not found in the NWB file!"
                f"Available options are: {available_electrical_series}."
            )
        electrodes_indices = electrical_series["electrodes"][:]
        # The table for all the electrodes in the nwbfile
        electrodes_table = hdf5_file["/general/extracellular_ephys/electrodes"]
        electrode_table_columns = electrodes_table.attrs["colnames"]

        # Get sampling frequency
        if "starting_time" in electrical_series.keys():
            t_start = electrical_series["starting_time"][()]
            sampling_frequency = electrical_series["starting_time"].attrs["rate"]
        elif "timestamps" in electrical_series.keys():
            timestamps = electrical_series["timestamps"][:]
            t_start = timestamps[0]
            sampling_frequency = 1.0 / np.median(np.diff(timestamps[:samples_for_rate_estimation]))

        if load_time_vector and timestamps is not None:
            times_kwargs = dict(time_vector=electrical_series.timestamps)
        else:
            times_kwargs = dict(sampling_frequency=sampling_frequency, t_start=t_start)

        # If channel names are present, use them as channel_ids instead of the electrode ids
        if "channel_name" in electrode_table_columns:
            channel_names = electrodes_table["channel_name"]
            channel_ids = channel_names[electrodes_indices]
            # Decode if bytes with utf-8
            channel_ids = [x.decode("utf-8") if isinstance(x, bytes) else x for x in channel_ids]

        else:
            channel_ids = [electrodes_table["id"][x] for x in electrodes_indices]

        dtype = electrical_series["data"].dtype
        electrical_series_data = electrical_series["data"]

        # need this for later
        self.electrical_series = electrical_series
        self._file = hdf5_file

        return channel_ids, sampling_frequency, dtype, electrical_series_data, times_kwargs

    def add_extractor_info_pynwb(self):
        from pynwb.ecephys import ElectrodeGroup

        electrodes_indices = self.electrical_series.electrodes.data[:]
        electrodes_table = self._nwbfile.electrodes
        if "group_name" in electrodes_table.colnames:
            unique_electrode_group_names = list(np.unique(electrodes_table["group_name"][:]))

        # Channels gains - for RecordingExtractor, these are values to cast traces to uV
        gains = self.electrical_series.conversion * 1e6
        if self.electrical_series.channel_conversion is not None:
            gains = self.electrical_series.conversion * self.electrical_series.channel_conversion[:] * 1e6

        # Set gains
        self.set_channel_gains(gains)

        # Set offsets
        offset = self.electrical_series.offset if hasattr(self.electrical_series, "offset") else 0
        if offset == 0 and "offset" in electrodes_table:
            offset = electrodes_table["offset"].data[electrodes_indices]

        self.set_channel_offsets(offset * 1e6)

        #########
        # Extract and re-name properties from nwbfile TODO: Should be a function
        ########

        properties = dict()
        # Extract rel_x, rel_y and rel_z and assign to location

        # TODO: Refactor ALL of this and add tests. This is difficult to read.
        if "rel_x" in electrodes_table:
            ndim = 3 if "rel_z" in electrodes_table else 2
            properties["location"] = np.zeros((self.get_num_channels(), ndim), dtype=float)

        for electrical_series_index, (channel_id, electrode_table_index) in enumerate(
            zip(self.channel_ids, electrodes_indices)
        ):
            if "rel_x" in electrodes_table:
                properties["location"][electrical_series_index, 0] = electrodes_table["rel_x"][electrode_table_index]
                if "rel_y" in electrodes_table:
                    properties["location"][electrical_series_index, 1] = electrodes_table["rel_y"][
                        electrode_table_index
                    ]
                if "rel_z" in electrodes_table:
                    properties["location"][electrical_series_index, 2] = electrodes_table["rel_z"][
                        electrode_table_index
                    ]

        for electrical_series_index, (channel_id, electrode_table_index) in enumerate(
            zip(self.channel_ids, electrodes_indices)
        ):
            for column in electrodes_table.colnames:
                if isinstance(electrodes_table[column][electrode_table_index], ElectrodeGroup):
                    continue
                elif column == "channel_name":
                    # channel_names are already set as channel ids!
                    continue
                elif column == "group_name":
                    group = unique_electrode_group_names.index(electrodes_table[column][electrode_table_index])
                    if "group" not in properties:
                        properties["group"] = np.zeros(self.get_num_channels(), dtype=type(group))
                    properties["group"][electrical_series_index] = group
                elif column == "location":
                    brain_area = electrodes_table[column][electrode_table_index]
                    if "brain_area" not in properties:
                        properties["brain_area"] = np.zeros(self.get_num_channels(), dtype=type(brain_area))
                    properties["brain_area"][electrical_series_index] = brain_area
                elif column == "offset":
                    offset = electrodes_table[column][electrode_table_index]
                    if "offset" not in properties:
                        properties["offset"] = np.zeros(self.get_num_channels(), dtype=type(offset))
                    properties["offset"][electrical_series_index] = offset
                elif column in ["x", "y", "z", "rel_x", "rel_y", "rel_z"]:
                    continue
                else:
                    val = electrodes_table[column][electrode_table_index]
                    if column not in properties:
                        properties[column] = np.zeros(self.get_num_channels(), dtype=type(val))
                    properties[column][electrical_series_index] = val

        # Set the properties in the recorder
        for property_name, values in properties.items():
            if property_name == "location":
                self.set_dummy_probe_from_locations(values)
            elif property_name == "group":
                if np.isscalar(values):
                    groups = [values] * len(self.channel_ids)
                else:
                    groups = values
                self.set_channel_groups(groups)
            else:
                self.set_property(property_name, values)

    def add_extractor_info_backend(self):
        electrodes_indices = self.electrical_series["electrodes"][:]
        electrodes_table = self._file["/general/extracellular_ephys/electrodes"]
        electrode_table_columns = electrodes_table.attrs["colnames"]
        if "group_name" in electrode_table_columns:
            unique_electrode_group_names = list(np.unique(electrodes_table["group_name"][:]))
        # Channels gains - for RecordingExtractor, these are values to cast traces to uV
        data_attributes = self.electrical_series["data"].attrs
        electrical_series_conversion = data_attributes["conversion"]
        gains = electrical_series_conversion * 1e6
        if "channel_conversion" in data_attributes:
            gains *= self.electrical_series["channel_conversion"][:]

        # Set gains
        self.set_channel_gains(gains)

        # Set offsets
        offset = data_attributes["offset"] if "offset" in data_attributes else 0
        if offset == 0 and "offset" in electrode_table_columns:
            offset = electrodes_table["offset"][electrodes_indices]

        self.set_channel_offsets(offset * 1e6)

        #########
        # Extract and re-name properties from nwbfile TODO: Should be a function
        ########

        properties = dict()
        # Extract rel_x, rel_y and rel_z and assign to location

        # TODO: Refactor ALL of this and add tests. This is difficult to read.
        if "rel_x" in electrodes_table:
            ndim = 3 if "rel_z" in electrodes_table else 2
            properties["location"] = np.zeros((self.get_num_channels(), ndim), dtype=float)

        for electrical_series_index, (channel_id, electrode_table_index) in enumerate(
            zip(self.channel_ids, electrodes_indices)
        ):
            if "rel_x" in electrodes_table:
                properties["location"][electrical_series_index, 0] = electrodes_table["rel_x"][electrode_table_index]
                if "rel_y" in electrodes_table:
                    properties["location"][electrical_series_index, 1] = electrodes_table["rel_y"][
                        electrode_table_index
                    ]
                if "rel_z" in electrodes_table:
                    properties["location"][electrical_series_index, 2] = electrodes_table["rel_z"][
                        electrode_table_index
                    ]

        # Extract all the other properties
        for electrical_series_index, (channel_id, electrode_table_index) in enumerate(
            zip(self.channel_ids, electrodes_indices)
        ):
            for column in electrode_table_columns:
                if column == "channel_name":
                    # channel_names are already set as channel ids!
                    continue
                elif column == "group_name":
                    group = unique_electrode_group_names.index(electrodes_table[column][electrode_table_index])
                    if "group" not in properties:
                        properties["group"] = np.zeros(self.get_num_channels(), dtype=type(group))
                    properties["group"][electrical_series_index] = group
                elif column == "location":
                    brain_area = electrodes_table[column][electrode_table_index]
                    if "brain_area" not in properties:
                        properties["brain_area"] = np.zeros(self.get_num_channels(), dtype=type(brain_area))
                    properties["brain_area"][electrical_series_index] = brain_area
                elif column == "offset":
                    offset = electrodes_table[column][electrode_table_index]
                    if "offset" not in properties:
                        properties["offset"] = np.zeros(self.get_num_channels(), dtype=type(offset))
                    properties["offset"][electrical_series_index] = offset
                elif column in ["x", "y", "z", "rel_x", "rel_y", "rel_z"]:
                    continue
                else:
                    val = electrodes_table[column][electrode_table_index]
                    if column not in properties:
                        properties[column] = np.zeros(self.get_num_channels(), dtype=type(val))
                    properties[column][electrical_series_index] = val

        # Set the properties in the recorder
        for property_name, values in properties.items():
            if property_name == "location":
                self.set_dummy_probe_from_locations(values)
            elif property_name == "group":
                if np.isscalar(values):
                    groups = [values] * len(self.channel_ids)
                else:
                    groups = values
                self.set_channel_groups(groups)
            else:
                # If any properties is bytes, decode with utf-8
                values = [x.decode("utf-8") if isinstance(x, bytes) else x for x in values]
                self.set_property(property_name, values)


class NwbRecordingSegment(BaseRecordingSegment):
    def __init__(self, electrical_series_data, times_kwargs):
        BaseRecordingSegment.__init__(self, **times_kwargs)
        self.electrical_series_data = electrical_series_data
        self._num_samples = self.electrical_series_data.shape[0]

    def get_num_samples(self):
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex: Number of samples in the signal block
        """
        return self._num_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        electrical_series_data = self.electrical_series_data
        if electrical_series_data.ndim == 1:
            traces = electrical_series_data[start_frame:end_frame][:, np.newaxis]
        elif isinstance(channel_indices, slice):
            traces = electrical_series_data[start_frame:end_frame, channel_indices]
        else:
            # channel_indices is np.ndarray
            if np.array(channel_indices).size > 1 and np.any(np.diff(channel_indices) < 0):
                # get around h5py constraint that it does not allow datasets
                # to be indexed out of order
                sorted_channel_indices = np.sort(channel_indices)
                resorted_indices = np.array([list(sorted_channel_indices).index(ch) for ch in channel_indices])
                recordings = electrical_series_data[start_frame:end_frame, sorted_channel_indices]
                traces = recordings[:, resorted_indices]
            else:
                traces = electrical_series_data[start_frame:end_frame, channel_indices]

        return traces


class NwbSortingExtractor(BaseSorting):
    """Load an NWBFile as a SortingExtractor.
    Parameters
    ----------
    file_path: str or Path
        Path to NWB file.
    electrical_series_name: str or None, default: None
        The name of the ElectricalSeries (if multiple ElectricalSeries are present).
    sampling_frequency: float or None, default: None
        The sampling frequency in Hz (required if no ElectricalSeries is available).
    unit_table_name: str or None, default: "units"
        The path of the unit table in the NWB file.
    samples_for_rate_estimation: int, default: 100000
        The number of timestamp samples to use to estimate the rate.
        Used if "rate" is not specified in the ElectricalSeries.
    stream_mode : "fsspec" | "ros3" | "remfile" | None, default: None
        The streaming mode to use. If None it assumes the file is on the local disk.
    cache: bool, default: False
        If True, the file is cached in the file passed to stream_cache_path
        if False, the file is not cached.
    stream_cache_path: str or Path or None, default: None
        Local path for caching. If None it uses the system temporary directory.
    t_start: float or None, default: None
        This is the time at which the corresponding ElectricalSeries start. NWB stores its spikes as times
        and the `t_start` is used to convert the times to seconds. Concrently, the returned frames are computed as:

        `frames = (times - t_start) * sampling_frequency`.

        As SpikeInterface always considers the first frame to be at the beginning of the recording independently
        of the `t_start`.

        When a `t_start` is not provided it will be inferred from the corresponding ElectricalSeries with name equal
        to `electrical_series_name`. The `t_start` then will be either the `ElectricalSeries.starting_time` or the
        first timestamp in the `ElectricalSeries.timestamps`.

    use_pynwb: bool, default: False
        Uses the pynwb library to read the NWB file. Setting this to False, the default, uses h5py
        to read the file. Using h5py can improve performance by bypassing some of the PyNWB validations.
    Returns
    -------
    sorting: NwbSortingExtractor
        The sorting extractor for the NWB file.
    """

    extractor_name = "NwbSorting"
    mode = "file"
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"
    name = "nwb"

    def __init__(
        self,
        file_path: str | Path,
        electrical_series_name: str | None = None,
        sampling_frequency: float | None = None,
        samples_for_rate_estimation: int = 1000,
        unit_table_name: str = "units",
        stream_mode: str | None = None,
        stream_cache_path: str | Path | None = None,
        *,
        t_start: float | None = None,
        cache: bool = False,
        use_pynwb: bool = False,
    ):
        self.stream_mode = stream_mode
        self.stream_cache_path = stream_cache_path
        self.electrical_series_name = electrical_series_name
        self.file_path = file_path
        self.t_start = t_start
        self.internal_sampling_frequency = sampling_frequency

        if use_pynwb:
            try:
                import pynwb
            except ImportError:
                raise ImportError(self.installation_mesg)

            unit_ids, spike_times_data, spike_times_index_data = self.extract_info_pynwb(
                unit_table_name=unit_table_name, samples_for_rate_estimation=samples_for_rate_estimation, cache=cache
            )
        else:
            unit_ids, spike_times_data, spike_times_index_data = self.extract_info_backend(
                unit_table_name=unit_table_name, samples_for_rate_estimation=samples_for_rate_estimation, cache=cache
            )

        BaseSorting.__init__(self, sampling_frequency=self.internal_sampling_frequency, unit_ids=unit_ids)

        sorting_segment = NwbSortingSegment(
            spike_times_data=spike_times_data,
            spike_times_index_data=spike_times_index_data,
            sampling_frequency=self.sampling_frequency,
            t_start=self.t_start,
        )
        self.add_sorting_segment(sorting_segment)

        if use_pynwb:
            self.add_extractor_info_pynwb()
        else:
            self.add_extractor_info_backend()

        if stream_mode is None and file_path is not None:
            file_path = str(Path(file_path).resolve())

        self._kwargs = {
            "file_path": file_path,
            "electrical_series_name": self.electrical_series_name,
            "sampling_frequency": sampling_frequency,
            "samples_for_rate_estimation": samples_for_rate_estimation,
            "cache": cache,
            "stream_mode": stream_mode,
            "stream_cache_path": stream_cache_path,
            "t_start": self.t_start,
        }

    def extract_info_pynwb(
        self, unit_table_name: str = None, samples_for_rate_estimation: int = 1000, cache: bool = False
    ):
        self._nwbfile = read_nwbfile(
            file_path=self.file_path,
            stream_mode=self.stream_mode,
            cache=cache,
            stream_cache_path=self.stream_cache_path,
        )

        timestamps = None
        if self.internal_sampling_frequency is None:
            # defines the electrical series from where the sorting came from
            # important to know the sampling_frequency
            self.electrical_series = retrieve_electrical_series(self._nwbfile, self.electrical_series_name)
            # get rate
            if self.electrical_series.rate is not None:
                self.internal_sampling_frequency = self.electrical_series.rate
                self.t_start = self.electrical_series.starting_time
            else:
                if hasattr(self.electrical_series, "timestamps"):
                    if self.electrical_series.timestamps is not None:
                        timestamps = self.electrical_series.timestamps
                        self.internal_sampling_frequency = 1 / np.median(
                            np.diff(timestamps[:samples_for_rate_estimation])
                        )
                        self.t_start = timestamps[0]
        assert (
            self.internal_sampling_frequency is not None
        ), "Couldn't load sampling frequency. Please provide it with the 'sampling_frequency' argument"
        assert (
            self.t_start is not None
        ), "Couldn't load a starting time for the sorting. Please provide it with the 't_start' argument"

        units_table = retrieve_unit_table(self._nwbfile, unit_table_name=unit_table_name)

        name_to_column_data = {c.name: c for c in units_table.columns}
        spike_times_data = name_to_column_data.pop("spike_times").data
        spike_times_index_data = name_to_column_data.pop("spike_times_index").data

        units_ids = name_to_column_data.pop("unit_name", None)
        if units_ids is None:
            units_ids = units_table["id"].data

        # need this for later
        self.unit_table = units_table

        return units_ids, spike_times_data, spike_times_index_data

    def extract_info_backend(
        self, unit_table_name: str = None, samples_for_rate_estimation: int = 1000, cache: bool = False
    ):
        hdf5_file = read_file_from_backend(
            file_path=self.file_path,
            stream_mode=self.stream_mode,
            cache=cache,
            stream_cache_path=self.stream_cache_path,
        )

        timestamps = None

        if self.internal_sampling_frequency is None or self.t_start is None:
            # defines the electrical series from where the sorting came from
            # important to know the sampling_frequency
            available_electrical_series = find_neurodata_type_from_backend(hdf5_file, neurodata_type="ElectricalSeries")
            if self.electrical_series_name is None:
                if len(available_electrical_series) == 1:
                    self.electrical_series_name = available_electrical_series[0]
                else:
                    raise ValueError(
                        "Multiple ElectricalSeries found in the file. "
                        "Please specify the 'electrical_series_name' argument:"
                        f"Available options are: {available_electrical_series}."
                    )
            else:
                if self.electrical_series_name not in available_electrical_series:
                    raise ValueError(
                        f"'{self.electrical_series_name}' not found in the file. "
                        f"Available options are: {available_electrical_series}"
                    )
            electrical_series = hdf5_file[self.electrical_series_name]

            # Get sampling frequency
            if "starting_time" in electrical_series.keys():
                self.t_start = electrical_series["starting_time"][()]
                self.internal_sampling_frequency = electrical_series["starting_time"].attrs["rate"]
            elif "timestamps" in electrical_series.keys():
                timestamps = electrical_series["timestamps"][:]
                self.t_start = timestamps[0]
                self.internal_sampling_frequency = 1.0 / np.median(np.diff(timestamps[:samples_for_rate_estimation]))

        assert (
            self.internal_sampling_frequency is not None
        ), "Couldn't load sampling frequency. Please provide it with the 'sampling_frequency' argument"
        assert (
            self.t_start is not None
        ), "Couldn't load a starting time for the sorting. Please provide it with the 't_start' argument"

        available_unit_table_names = find_neurodata_type_from_backend(hdf5_file, neurodata_type="Units")
        if unit_table_name is None:
            if len(available_unit_table_names) == 1:
                unit_table_name = available_unit_table_names[0]
            else:
                raise ValueError(
                    "Multiple Units tables found in the file. "
                    "Please specify the 'unit_table_name' argument:"
                    f"Available options are: {available_unit_table_names}."
                )
        else:
            if unit_table_name not in available_unit_table_names:
                raise ValueError(
                    f"'{unit_table_name}' not found in the file. "
                    f"Available options are: {available_unit_table_names}"
                )
        self.unit_table_location = unit_table_name
        units_table = hdf5_file[self.unit_table_location]

        spike_times_data = units_table["spike_times"]
        spike_times_index_data = units_table["spike_times_index"]

        if "unit_name" in units_table:
            unit_ids = units_table["unit_name"]
        else:
            unit_ids = units_table["id"]

        decode_to_string = lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        unit_ids = [decode_to_string(id) for id in unit_ids]

        # need this for later
        self.unit_table = units_table

        return unit_ids, spike_times_data, spike_times_index_data

    def add_extractor_info_pynwb(self):
        name_to_column_data = {c.name: c for c in self.unit_table.columns}
        # Add only the columns that are not indices
        index_columns = [name for name in name_to_column_data if name.endswith("_index")]
        properties_to_add = [name for name in name_to_column_data if name not in index_columns]
        # Filter those properties that are nested ragged arrays
        properties_to_add = [name for name in properties_to_add if f"{name}_index_index" not in name_to_column_data]

        for property_name in properties_to_add:
            data = name_to_column_data.pop(property_name).data
            data_index = name_to_column_data.get(f"{property_name}_index", None)
            not_ragged_array = data_index is None
            if not_ragged_array:
                values = data[:]
            else:  # TODO if we want we could make this recursive to handle nested ragged arrays
                data_index = data_index.data
                index_spacing = np.diff(data_index, prepend=0)
                all_index_spacing_are_the_same = np.unique(index_spacing).size == 1
                if all_index_spacing_are_the_same:
                    start_indices = [0] + list(data_index[:-1])
                    end_indices = list(data_index)
                    values = [data[start_index:end_index] for start_index, end_index in zip(start_indices, end_indices)]
                    self.set_property(property_name, np.asarray(values))

                else:
                    warnings.warn(f"Skipping {property_name} because of unequal shapes across units")
                    continue

            self.set_property(property_name, np.asarray(values))

    def add_extractor_info_backend(self):
        units_table = self.unit_table
        # Skip canonical properties and indices
        caonical_properties = ["spike_times", "spike_times_index", "unit_name"]
        index_properties = [name for name in units_table if name.endswith("_index")]
        nested_ragged_array_properties = [name for name in units_table if f"{name}_index_index" in units_table]

        skip_properties = caonical_properties + index_properties + nested_ragged_array_properties
        properties_to_add = [name for name in units_table if name not in skip_properties]

        for property_name in properties_to_add:
            data = units_table[property_name]
            corresponding_index_name = f"{property_name}_index"
            not_ragged_array = corresponding_index_name not in units_table
            if not_ragged_array:
                values = data[:]
            else:
                data_index = units_table[corresponding_index_name][:]
                index_spacing = np.diff(data_index, prepend=0)
                all_index_spacing_are_the_same = np.unique(index_spacing).size == 1
                if all_index_spacing_are_the_same:
                    start_indices = [0] + list(data_index[:-1])
                    end_indices = list(data_index)
                    values = [data[start_index:end_index] for start_index, end_index in zip(start_indices, end_indices)]

                else:
                    warnings.warn(f"Skipping {property_name} because of unequal shapes across units")
                    continue

            decode_to_string = lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            values = [decode_to_string(val) for val in values]
            self.set_property(property_name, np.asarray(values))


class NwbSortingSegment(BaseSortingSegment):
    def __init__(self, spike_times_data, spike_times_index_data, sampling_frequency: float, t_start: float):
        BaseSortingSegment.__init__(self)
        self.spike_times_data = spike_times_data
        self.spike_times_index_data = spike_times_index_data
        self.spike_times_data = spike_times_data
        self.spike_times_index_data = spike_times_index_data
        self._sampling_frequency = sampling_frequency
        self._t_start = t_start

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        # Extract the spike times for the unit
        unit_index = self.parent_extractor.id_to_index(unit_id)
        if unit_index == 0:
            start_index = 0
        else:
            start_index = self.spike_times_index_data[unit_index - 1]
        end_index = self.spike_times_index_data[unit_index]
        spike_times = self.spike_times_data[start_index:end_index]

        # Transform spike times to frames and subset
        frames = np.round((spike_times - self._t_start) * self._sampling_frequency)

        start_index = 0
        if start_frame is not None:
            start_index = np.searchsorted(frames, start_frame, side="left")

        if end_frame is not None:
            end_index = np.searchsorted(frames, end_frame, side="left")
        else:
            end_index = frames.size

        return frames[start_index:end_index].astype("int64", copy=False)


read_nwb_recording = define_function_from_class(source_class=NwbRecordingExtractor, name="read_nwb_recording")
read_nwb_sorting = define_function_from_class(source_class=NwbSortingExtractor, name="read_nwb_sorting")


def read_nwb(file_path, load_recording=True, load_sorting=False, electrical_series_name=None):
    """Reads NWB file into SpikeInterface extractors.

    Parameters
    ----------
    file_path: str or Path
        Path to NWB file.
    load_recording : bool, default: True
        If True, the recording object is loaded.
    load_sorting : bool, default: False
        If True, the recording object is loaded.
    electrical_series_name: str or None, default: None
        The name of the ElectricalSeries (if multiple ElectricalSeries are present)

    Returns
    -------
    extractors: extractor or tuple
        Single RecordingExtractor/SortingExtractor or tuple with both
        (depending on "load_recording"/"load_sorting") arguments.
    """
    outputs = ()
    if load_recording:
        rec = read_nwb_recording(file_path, electrical_series_name=electrical_series_name)
        outputs = outputs + (rec,)
    if load_sorting:
        sorting = read_nwb_sorting(file_path, electrical_series_name=electrical_series_name)
        outputs = outputs + (sorting,)

    if len(outputs) == 1:
        outputs = outputs[0]

    return outputs
