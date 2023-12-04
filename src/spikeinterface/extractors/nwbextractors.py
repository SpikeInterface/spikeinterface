from __future__ import annotations
from pathlib import Path
from typing import Union, List, Optional, Literal, Dict, BinaryIO

import numpy as np

from spikeinterface import get_global_tmp_folder
from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


def import_lazily():
    "Makes annotations / typing available lazily"
    global NWBFile, ElectricalSeries, NWBHDF5IO
    from pynwb import NWBFile
    from pynwb.ecephys import ElectricalSeries
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

    if electrical_series_name is not None:
        # TODO note that this case does not handle repetitions of the same name
        electrical_series_dict: Dict[str, ElectricalSeries] = {
            item.name: item for item in nwbfile.all_children() if isinstance(item, ElectricalSeries)
        }
        if electrical_series_name not in electrical_series_dict:
            raise ValueError(f"{electrical_series_name} not found in the NWBFile. ")
        electrical_series = electrical_series_dict[electrical_series_name]
    else:
        electrical_series_list: List[ElectricalSeries] = [
            series for series in nwbfile.acquisition.values() if isinstance(series, ElectricalSeries)
        ]
        if len(electrical_series_list) > 1:
            raise ValueError(
                f"More than one acquisition found! You must specify 'electrical_series_name'. \n"
                f"Options in current file are: {[e.name for e in electrical_series_list]}"
            )
        if len(electrical_series_list) == 0:
            raise ValueError("No acquisitions found in the .nwb file.")
        electrical_series = electrical_series_list[0]

    return electrical_series


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

    if stream_mode == "fsspec":
        import fsspec
        import h5py

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

        file = h5py.File(ffspec_file, "r")
        io = NWBHDF5IO(file=file, mode="r", load_namespaces=True)

    elif stream_mode == "ros3":
        import h5py

        assert file_path is not None, "file_path must be specified when using stream_mode='ros3'"

        drivers = h5py.registered_drivers()
        assertion_msg = "ROS3 support not enbabled, use: install -c conda-forge h5py>=3.2 to enable streaming"
        assert "ros3" in drivers, assertion_msg
        io = NWBHDF5IO(path=file_path, mode="r", load_namespaces=True, driver="ros3")

    elif stream_mode == "remfile":
        import remfile
        import h5py

        assert file_path is not None, "file_path must be specified when using stream_mode='remfile'"
        rfile = remfile.File(file_path)
        h5_file = h5py.File(rfile, "r")
        io = NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True)

    elif file_path is not None:
        file_path = str(Path(file_path).absolute())
        io = NWBHDF5IO(path=file_path, mode="r", load_namespaces=True)

    else:
        import h5py

        assert file is not None, "Unexpected, file is None"
        h5_file = h5py.File(file, "r")
        io = NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True)

    nwbfile = io.read()
    return nwbfile


class NwbRecordingExtractor(BaseRecording):
    """Load an NWBFile as a RecordingExtractor.

    Parameters
    ----------
    file_path: str, Path, or None
        Path to NWB file or s3 url (or None if using file instead)
    electrical_series_name: str or None, default: None
        The name of the ElectricalSeries. Used if multiple ElectricalSeries are present.
    file: file-like object or None, default: None
        File-like object to read from (if None, file_path must be specified)
    load_time_vector: bool, default: False
        If True, the time vector is loaded to the recording object.
    samples_for_rate_estimation: int, default: 100000
        The number of timestamp samples to use to estimate the rate.
        Used if "rate" is not specified in the ElectricalSeries.
    stream_mode : "fsspec" | "ros3" | "remfile" | None, default: None
        The streaming mode to use. If None it assumes the file is on the local disk.
    cache: bool, default: False
        If True, the file is cached in the file passed to stream_cache_path
        if False, the file is not cached.
    stream_cache_path: str or Path or None, default: None
        Local path for caching. If None it uses the current working directory (cwd)

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
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"
    name = "nwb"

    def __init__(
        self,
        file_path: str | Path | None = None,  # provide either this or file
        electrical_series_name: str | None = None,
        load_time_vector: bool = False,
        samples_for_rate_estimation: int = 100000,
        cache: bool = False,
        stream_mode: Optional[Literal["fsspec", "ros3", "remfile"]] = None,
        stream_cache_path: str | Path | None = None,
        *,
        file: BinaryIO | None = None,  # file-like - provide either this or file_path
    ):
        try:
            from pynwb import NWBHDF5IO, NWBFile
            from pynwb.ecephys import ElectrodeGroup
        except ImportError:
            raise ImportError(self.installation_mesg)

        if file_path is not None and file is not None:
            raise ValueError("Provide either file_path or file, not both")
        if file_path is None and file is None:
            raise ValueError("Provide either file_path or file")

        self.stream_mode = stream_mode
        self.stream_cache_path = stream_cache_path
        self._electrical_series_name = electrical_series_name

        self.file_path = file_path
        self._nwbfile = read_nwbfile(
            file_path=file_path, file=file, stream_mode=stream_mode, cache=cache, stream_cache_path=stream_cache_path
        )
        electrical_series = retrieve_electrical_series(self._nwbfile, electrical_series_name)
        # The indices in the electrode table corresponding to this electrical series
        electrodes_indices = electrical_series.electrodes.data[:]
        # The table for all the electrodes in the nwbfile
        electrodes_table = self._nwbfile.electrodes

        ###################
        # Extract temporal information TODO: Should be a function
        ###################
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
            assert timestamps is not None, (
                "Could not find rate information as both 'rate' and "
                "'timestamps' are missing from the file. "
                "Use the 'sampling_frequency' argument."
            )
            sampling_frequency = 1.0 / np.median(np.diff(timestamps[:samples_for_rate_estimation]))

        if load_time_vector and timestamps is not None:
            times_kwargs = dict(time_vector=electrical_series.timestamps)
        else:
            times_kwargs = dict(sampling_frequency=sampling_frequency, t_start=t_start)

        # Extractors channel groups must be integers, but Nwb electrodes group_name can be strings
        if "group_name" in electrodes_table.colnames:
            unique_electrode_group_names = list(np.unique(electrodes_table["group_name"][:]))

        # Fill channel properties dictionary from electrodes table
        if "channel_name" in electrodes_table.colnames:
            channel_ids = [
                electrical_series.electrodes["channel_name"][electrodes_index]
                for electrodes_index in electrodes_indices
            ]
        else:
            channel_ids = [electrical_series.electrodes.table.id[x] for x in electrodes_indices]

        dtype = electrical_series.data.dtype
        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        num_frames = int(electrical_series.data.shape[0])
        recording_segment = NwbRecordingSegment(
            nwbfile=self._nwbfile,
            electrical_series_name=electrical_series_name,
            num_frames=num_frames,
            times_kwargs=times_kwargs,
        )
        self.add_recording_segment(recording_segment)

        #################
        # Extract gains and offsets TODO: Should be a function
        #################

        # Channels gains - for RecordingExtractor, these are values to cast traces to uV
        gains = electrical_series.conversion * 1e6
        if electrical_series.channel_conversion is not None:
            gains = electrical_series.conversion * electrical_series.channel_conversion[:] * 1e6

        # Set gains
        self.set_channel_gains(gains)

        # Set offsets
        offset = electrical_series.offset if hasattr(electrical_series, "offset") else 0
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
            zip(channel_ids, electrodes_indices)
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
            zip(channel_ids, electrodes_indices)
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
                    groups = [values] * len(channel_ids)
                else:
                    groups = values
                self.set_channel_groups(groups)
            else:
                self.set_property(property_name, values)

        if stream_mode is None and file_path is not None:
            file_path = str(Path(file_path).resolve())

        if stream_mode == "fsspec" and stream_cache_path is not None:
            stream_cache_path = str(Path(self.stream_cache_path).absolute())

        self.extra_requirements.extend(["pandas", "pynwb", "hdmf"])
        self._electrical_series = electrical_series

        # set serializability bools
        if file is not None:
            # not json serializable if file arg is provided
            self._serializability["json"] = False

        self._kwargs = {
            "file_path": file_path,
            "electrical_series_name": self._electrical_series_name,
            "load_time_vector": load_time_vector,
            "samples_for_rate_estimation": samples_for_rate_estimation,
            "stream_mode": stream_mode,
            "cache": cache,
            "stream_cache_path": stream_cache_path,
            "file": file,
        }


class NwbRecordingSegment(BaseRecordingSegment):
    def __init__(self, nwbfile, electrical_series_name, num_frames, times_kwargs):
        BaseRecordingSegment.__init__(self, **times_kwargs)
        self._nwbfile = nwbfile
        self._electrical_series_name = electrical_series_name
        self.electrical_series = retrieve_electrical_series(self._nwbfile, self._electrical_series_name)
        self._num_samples = num_frames

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

        electrical_series_data = self.electrical_series.data
        if isinstance(channel_indices, slice):
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
        samples_for_rate_estimation: int = 100000,
        stream_mode: str | None = None,
        cache: bool = False,
        stream_cache_path: str | Path | None = None,
    ):
        try:
            from pynwb import NWBHDF5IO, NWBFile
            from pynwb.ecephys import ElectrodeGroup
        except ImportError:
            raise ImportError(self.installation_mesg)

        self.stream_mode = stream_mode
        self.stream_cache_path = stream_cache_path
        self._electrical_series_name = electrical_series_name

        self.file_path = file_path
        self._nwbfile = read_nwbfile(
            file_path=file_path, stream_mode=stream_mode, cache=cache, stream_cache_path=stream_cache_path
        )

        units_ids = list(self._nwbfile.units.id[:])

        timestamps = None
        if sampling_frequency is None:
            # defines the electrical series from where the sorting came from
            # important to know the sampling_frequency
            self.electrical_series = retrieve_electrical_series(self._nwbfile, self._electrical_series_name)
            # get rate
            if self.electrical_series.rate is not None:
                sampling_frequency = self.electrical_series.rate
            else:
                if hasattr(self.electrical_series, "timestamps"):
                    if self.electrical_series.timestamps is not None:
                        timestamps = self.electrical_series.timestamps
                        sampling_frequency = 1 / np.median(np.diff(timestamps[samples_for_rate_estimation]))

        assert sampling_frequency is not None, (
            "Couldn't load sampling frequency. Please provide it with the " "'sampling_frequency' argument"
        )

        BaseSorting.__init__(self, sampling_frequency=sampling_frequency, unit_ids=units_ids)
        sorting_segment = NwbSortingSegment(
            nwbfile=self._nwbfile, sampling_frequency=sampling_frequency, timestamps=timestamps
        )
        self.add_sorting_segment(sorting_segment)

        # Add properties:
        properties = dict()
        import warnings

        for column in list(self._nwbfile.units.colnames):
            if column == "spike_times":
                continue

            # Note that this has a different behavior than self._nwbfile.units[column].data
            property_values = self._nwbfile.units[column][:]

            # Making this explicit because I am not sure this is the best test
            is_raggged_array = isinstance(property_values, list)
            if is_raggged_array:
                all_values_have_equal_shape = np.all([p.shape == property_values[0].shape for p in property_values])
                if all_values_have_equal_shape:
                    properties[column] = property_values
                else:
                    warnings.warn(f"Skipping {column} because of unequal shapes across units")

                continue  # To next property

            # The rest of the properties are added as they come
            properties[column] = property_values

        for prop_name, values in properties.items():
            self.set_property(prop_name, np.array(values))

        if stream_mode is None and file_path is not None:
            file_path = str(Path(file_path).resolve())

        self._kwargs = {
            "file_path": file_path,
            "electrical_series_name": self._electrical_series_name,
            "sampling_frequency": sampling_frequency,
            "samples_for_rate_estimation": samples_for_rate_estimation,
            "cache": cache,
            "stream_mode": stream_mode,
            "stream_cache_path": stream_cache_path,
        }


class NwbSortingSegment(BaseSortingSegment):
    def __init__(self, nwbfile, sampling_frequency, timestamps):
        BaseSortingSegment.__init__(self)
        self._nwbfile = nwbfile
        self._sampling_frequency = sampling_frequency
        self._timestamps = timestamps

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
    ) -> np.ndarray:
        # must be implemented in subclass
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.inf
        spike_times = self._nwbfile.units["spike_times"][list(self._nwbfile.units.id[:]).index(unit_id)][:]

        if self._timestamps is not None:
            frames = np.searchsorted(spike_times, self.timestamps)
        else:
            frames = np.round(spike_times * self._sampling_frequency)
        return frames[(frames >= start_frame) & (frames < end_frame)].astype("int64", copy=False)


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
