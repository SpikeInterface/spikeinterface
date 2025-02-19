from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Literal, Dict, BinaryIO
import warnings

import numpy as np

from spikeinterface import get_global_tmp_folder
from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


def read_file_from_backend(
    *,
    file_path: str | Path | None,
    file: BinaryIO | None = None,
    stream_mode: Literal["ffspec", "remfile"] | None = None,
    cache: bool = False,
    stream_cache_path: str | Path | None = None,
    storage_options: dict | None = None,
):
    """
    Reads a file from a hdf5 or zarr backend.
    """
    if stream_mode == "fsspec":
        import h5py
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

        if _is_hdf5_file(ffspec_file):
            open_file = h5py.File(ffspec_file, "r")
        else:
            raise RuntimeError(f"{file_path} is not a valid HDF5 file!")

    elif stream_mode == "ros3":
        import h5py

        assert file_path is not None, "file_path must be specified when using stream_mode='ros3'"

        drivers = h5py.registered_drivers()
        assertion_msg = "ROS3 support not enbabled, use: install -c conda-forge h5py>=3.2 to enable streaming"
        assert "ros3" in drivers, assertion_msg
        open_file = h5py.File(name=file_path, mode="r", driver="ros3")

    elif stream_mode == "remfile":
        import remfile
        import h5py

        assert file_path is not None, "file_path must be specified when using stream_mode='remfile'"
        rfile = remfile.File(file_path)
        if _is_hdf5_file(rfile):
            open_file = h5py.File(rfile, "r")
        else:
            raise RuntimeError(f"{file_path} is not a valid HDF5 file!")

    elif stream_mode == "zarr":
        import zarr

        open_file = zarr.open(file_path, mode="r", storage_options=storage_options)

    elif file_path is not None:  # local
        file_path = str(Path(file_path).resolve())
        backend = _get_backend_from_local_file(file_path)
        if backend == "zarr":
            import zarr

            open_file = zarr.open(file_path, mode="r")
        else:
            import h5py

            open_file = h5py.File(name=file_path, mode="r")
    else:
        import h5py

        assert file is not None, "Both file_path and file are None"
        open_file = h5py.File(file, "r")

    return open_file


def read_nwbfile(
    *,
    backend: Literal["hdf5", "zarr"],
    file_path: str | Path | None,
    file: BinaryIO | None = None,
    stream_mode: Literal["ffspec", "remfile", "zarr"] | None = None,
    cache: bool = False,
    stream_cache_path: str | Path | None = None,
    storage_options: dict | None = None,
) -> "NWBFile":
    """
    Read an NWB file and return the NWBFile object.

    Parameters
    ----------
    file_path : Path, str or None
        The path to the NWB file. Either provide this or file.
    file : file-like object or None
        The file-like object to read from. Either provide this or file_path.
    stream_mode : "fsspec" | "remfile" | None, default: None
        The streaming mode to use. If None it assumes the file is on the local disk.
    cache : bool, default: False
        If True, the file is cached in the file passed to stream_cache_path
        if False, the file is not cached.
    stream_cache_path : str or None, default: None
        The path to the cache storage, when default to None it uses the a temporary
        folder.
    Returns
    -------
    nwbfile : NWBFile
        The NWBFile object.

    Notes
    -----
    This function can stream data from the "fsspec", and "rem" protocols.


    Examples
    --------
    >>> nwbfile = read_nwbfile(file_path="data.nwb", backend="hdf5", stream_mode="fsspec")
    """

    if file_path is not None and file is not None:
        raise ValueError("Provide either file_path or file, not both")
    if file_path is None and file is None:
        raise ValueError("Provide either file_path or file")

    open_file = read_file_from_backend(
        file_path=file_path,
        file=file,
        stream_mode=stream_mode,
        cache=cache,
        stream_cache_path=stream_cache_path,
        storage_options=storage_options,
    )
    if backend == "hdf5":
        from pynwb import NWBHDF5IO

        io = NWBHDF5IO(file=open_file, mode="r", load_namespaces=True)
    else:
        from hdmf_zarr import NWBZarrIO

        io = NWBZarrIO(path=open_file.store, mode="r", load_namespaces=True)

    nwbfile = io.read()
    return nwbfile


1


def _retrieve_electrical_series_pynwb(
    nwbfile: "NWBFile", electrical_series_path: Optional[str] = None
) -> "ElectricalSeries":
    """
    Get an ElectricalSeries object from an NWBFile.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWBFile object from which to extract the ElectricalSeries.
    electrical_series_path : str, default: None
        The name of the ElectricalSeries to extract. If not specified, it will return the first found ElectricalSeries
        if there's only one; otherwise, it raises an error.

    Returns
    -------
    ElectricalSeries
        The requested ElectricalSeries object.

    Raises
    ------
    ValueError
        If no acquisitions are found in the NWBFile or if multiple acquisitions are found but no electrical_series_path
        is provided.
    AssertionError
        If the specified electrical_series_path is not present in the NWBFile.
    """
    from pynwb.ecephys import ElectricalSeries

    electrical_series_dict: Dict[str, ElectricalSeries] = {}

    for item in nwbfile.all_children():
        if isinstance(item, ElectricalSeries):
            # remove data and skip first "/"
            electrical_series_key = item.data.name.replace("/data", "")[1:]
            electrical_series_dict[electrical_series_key] = item

    if electrical_series_path is not None:
        if electrical_series_path not in electrical_series_dict:
            raise ValueError(f"{electrical_series_path} not found in the NWBFile. ")
        electrical_series = electrical_series_dict[electrical_series_path]
    else:
        electrical_series_list = list(electrical_series_dict.keys())
        if len(electrical_series_list) > 1:
            raise ValueError(
                f"More than one acquisition found! You must specify 'electrical_series_path'. \n"
                f"Options in current file are: {[e for e in electrical_series_list]}"
            )
        if len(electrical_series_list) == 0:
            raise ValueError("No acquisitions found in the .nwb file.")
        electrical_series = electrical_series_dict[electrical_series_list[0]]

    return electrical_series


def _retrieve_unit_table_pynwb(nwbfile: "NWBFile", unit_table_path: Optional[str] = None) -> "Units":
    """
    Get an Units object from an NWBFile.
    Units tables can be either the main unit table (nwbfile.units) or in the processing module.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWBFile object from which to extract the Units.
    unit_table_path : str, default: None
        The path of the Units to extract. If not specified, it will return the first found Units
        if there's only one; otherwise, it raises an error.

    Returns
    -------
    Units
        The requested Units object.

    Raises
    ------
    ValueError
        If no unit tables are found in the NWBFile or if multiple unit tables are found but no unit_table_path
        is provided.
    AssertionError
        If the specified unit_table_path is not present in the NWBFile.
    """
    from pynwb.misc import Units

    unit_table_dict: Dict[str:Units] = {}

    for item in nwbfile.all_children():
        if isinstance(item, Units):
            # retrieve name of "id" column and skip first "/"
            unit_table_key = item.id.data.name.replace("/id", "")[1:]
            unit_table_dict[unit_table_key] = item

    if unit_table_path is not None:
        if unit_table_path not in unit_table_dict:
            raise ValueError(f"{unit_table_path} not found in the NWBFile. ")
        unit_table = unit_table_dict[unit_table_path]
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


def _is_hdf5_file(filename_or_file):
    if isinstance(filename_or_file, (str, Path)):
        import h5py

        filename = str(filename_or_file)
        is_hdf5 = h5py.h5f.is_hdf5(filename.encode("utf-8"))
    else:
        file_signature = filename_or_file.read(8)
        # Source of the magic number https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html
        is_hdf5 = file_signature == b"\x89HDF\r\n\x1a\n"

    return is_hdf5


def _get_backend_from_local_file(file_path: str | Path) -> str:
    """
    Returns the file backend from a file path ("hdf5", "zarr")

    Parameters
    ----------
    file_path : str or Path
        The path to the file.

    Returns
    -------
    backend : str
        The file backend ("hdf5", "zarr")
    """
    file_path = Path(file_path)
    if file_path.is_file():
        if _is_hdf5_file(file_path):
            backend = "hdf5"
        else:
            raise RuntimeError(f"{file_path} is not a valid HDF5 file!")
    elif file_path.is_dir():
        try:
            import zarr

            with zarr.open(file_path, "r") as f:
                backend = "zarr"
        except:
            raise RuntimeError(f"{file_path} is not a valid Zarr folder!")
    else:
        raise RuntimeError(f"File {file_path} is not an existing file or folder!")
    return backend


def _find_neurodata_type_from_backend(group, path="", result=None, neurodata_type="ElectricalSeries", backend="hdf5"):
    """
    Recursively searches for groups with the specified neurodata_type hdf5 or zarr object,
    and returns a list with their paths.
    """
    if backend == "hdf5":
        import h5py

        group_class = h5py.Group
    else:
        import zarr

        group_class = zarr.Group

    if result is None:
        result = []

    for neurodata_name, value in group.items():
        # Check if it's a group and if it has the neurodata_type
        if isinstance(value, group_class):
            current_path = f"{path}/{neurodata_name}" if path else neurodata_name
            if value.attrs.get("neurodata_type") == neurodata_type:
                result.append(current_path)
            _find_neurodata_type_from_backend(
                value, current_path, result, neurodata_type, backend
            )  # Recursive call for sub-groups
    return result


def _fetch_time_info_pynwb(electrical_series, samples_for_rate_estimation, load_time_vector=False):
    """
    Extracts the sampling frequency and the time vector from an ElectricalSeries object.
    """
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


def _retrieve_electrodes_indices_from_electrical_series_backend(open_file, electrical_series, backend="hdf5"):
    """
    Retrieves the indices of the electrodes from the electrical series.
    For the Zarr backend, the electrodes are stored in the electrical_series.attrs["zarr_link"].
    """
    if "electrodes" not in electrical_series:
        if backend == "zarr":
            import zarr

            # links must be resolved
            zarr_links = electrical_series.attrs["zarr_link"]
            electrodes_path = None
            for zarr_link in zarr_links:
                if zarr_link["name"] == "electrodes":
                    electrodes_path = zarr_link["path"]
            assert electrodes_path is not None, "electrodes must be present in the electrical series"
            electrodes_indices = open_file[electrodes_path][:]
        else:
            raise ValueError("electrodes must be present in the electrical series")
    else:
        electrodes_indices = electrical_series["electrodes"][:]
    return electrodes_indices


class _BaseNWBExtractor:
    "A class for common methods for NWB extractors."

    def _close_hdf5_file(self):
        has_hdf5_backend = hasattr(self, "_file")
        if has_hdf5_backend:
            import h5py

            main_file_id = self._file.id
            open_object_ids_main = h5py.h5f.get_obj_ids(main_file_id, types=h5py.h5f.OBJ_ALL)
            for object_id in open_object_ids_main:
                object_name = h5py.h5i.get_name(object_id).decode("utf-8")
                try:
                    object_id.close()
                except:
                    import warnings

                    warnings.warn(f"Error closing object {object_name}")

    def __del__(self):
        # backend mode
        if hasattr(self, "_file"):
            if hasattr(self._file, "store"):
                self._file.store.close()
            else:
                self._close_hdf5_file()
        # pynwb mode
        elif hasattr(self, "_nwbfile"):
            io = self._nwbfile.get_read_io()
            if io is not None:
                io.close()


class NwbRecordingExtractor(BaseRecording, _BaseNWBExtractor):
    """Load an NWBFile as a RecordingExtractor.

    Parameters
    ----------
    file_path : str, Path, or None
        Path to the NWB file or an s3 URL. Use this parameter to specify the file location
        if not using the `file` parameter.
    electrical_series_name : str or None, default: None
        Deprecated, use `electrical_series_path` instead.
    electrical_series_path : str or None, default: None
        The name of the ElectricalSeries object within the NWB file. This parameter is crucial
        when the NWB file contains multiple ElectricalSeries objects. It helps in identifying
        which specific series to extract data from. If there is only one ElectricalSeries and
        this parameter is not set, that unique series will be used by default.
        If multiple ElectricalSeries are present and this parameter is not set, an error is raised.
        The `electrical_series_path` corresponds to the path within the NWB file, e.g.,
        'acquisition/MyElectricalSeries`.
    load_time_vector : bool, default: False
        If set to True, the time vector is also loaded into the recording object. Useful for
        cases where precise timing information is required.
    samples_for_rate_estimation : int, default: 1000
        The number of timestamp samples used for estimating the sampling rate. This is relevant
        when the 'rate' attribute is not available in the ElectricalSeries.
    stream_mode : "fsspec" | "remfile" | "zarr" | None, default: None
        Determines the streaming mode for reading the file. Use this for optimized reading from
        different sources, such as local disk or remote servers.
    load_channel_properties : bool, default: True
        If True, all the channel properties are loaded from the NWB file and stored as properties.
        For streaming purposes, it can be useful to set this to False to speed up streaming.
    file : file-like object or None, default: None
        A file-like object representing the NWB file. Use this parameter if you have an in-memory
        representation of the NWB file instead of a file path.
    cache : bool, default: False
        Indicates whether to cache the file locally when using streaming. Caching can improve performance for
        remote files.
    stream_cache_path : str, Path, or None, default: None
        Specifies the local path for caching the file. Relevant only if `cache` is True.
    storage_options : dict | None = None,
        These are the additional kwargs (e.g. AWS credentials) that are passed to the zarr.open convenience function.
        This is only used on the "zarr" stream_mode.
    use_pynwb : bool, default: False
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
    >>> dandiset_id = "001054"
    >>> filepath = "sub-Dory/sub-Dory_ses-2020-09-14-004_ecephys.nwb"
    >>> with DandiAPIClient() as client:
    >>>     asset = client.get_dandiset(dandiset_id).get_asset_by_path(filepath)
    >>>     s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    >>>
    >>> rec = NwbRecordingExtractor(s3_url, stream_mode="remfile")
    """

    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(
        self,
        file_path: str | Path | None = None,  # provide either this or file
        electrical_series_name: str | None = None,  # deprecated
        load_time_vector: bool = False,
        samples_for_rate_estimation: int = 1_000,
        stream_mode: Optional[Literal["fsspec", "remfile", "zarr"]] = None,
        stream_cache_path: str | Path | None = None,
        electrical_series_path: str | None = None,
        load_channel_properties: bool = True,
        *,
        file: BinaryIO | None = None,  # file-like - provide either this or file_path
        cache: bool = False,
        storage_options: dict | None = None,
        use_pynwb: bool = False,
    ):

        if stream_mode == "ros3":
            warnings.warn(
                "The 'ros3' stream_mode is deprecated and will be removed in version 0.103.0. "
                "Use 'fsspec' stream_mode instead.",
                DeprecationWarning,
            )

        if file_path is not None and file is not None:
            raise ValueError("Provide either file_path or file, not both")
        if file_path is None and file is None:
            raise ValueError("Provide either file_path or file")

        if electrical_series_name is not None:
            warning_msg = (
                "The `electrical_series_name` parameter is deprecated and will be removed in version 0.101.0.\n"
                "Use `electrical_series_path` instead."
            )
            if electrical_series_path is None:
                warning_msg += f"\nSetting `electrical_series_path` to 'acquisition/{electrical_series_name}'."
                electrical_series_path = f"acquisition/{electrical_series_name}"
            else:
                warning_msg += f"\nIgnoring `electrical_series_name` and using the provided `electrical_series_path`."
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)

        self.file_path = file_path
        self.stream_mode = stream_mode
        self.stream_cache_path = stream_cache_path
        self.storage_options = storage_options
        self.electrical_series_path = electrical_series_path

        if self.stream_mode is None and file is None:
            self.backend = _get_backend_from_local_file(file_path)
        else:
            if self.stream_mode == "zarr":
                self.backend = "zarr"
            else:
                self.backend = "hdf5"

        # extract info
        if use_pynwb:
            try:
                import pynwb
            except ImportError:
                raise ImportError(self.installation_mesg)

            (
                channel_ids,
                sampling_frequency,
                dtype,
                segment_data,
                times_kwargs,
            ) = self._fetch_recording_segment_info_pynwb(file, cache, load_time_vector, samples_for_rate_estimation)
        else:
            (
                channel_ids,
                sampling_frequency,
                dtype,
                segment_data,
                times_kwargs,
            ) = self._fetch_recording_segment_info_backend(file, cache, load_time_vector, samples_for_rate_estimation)

        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        recording_segment = NwbRecordingSegment(
            electrical_series_data=segment_data,
            times_kwargs=times_kwargs,
        )
        self.add_recording_segment(recording_segment)

        # fetch and add main recording properties
        if use_pynwb:
            gains, offsets, locations, groups = self._fetch_main_properties_pynwb()
        else:
            gains, offsets, locations, groups = self._fetch_main_properties_backend()

        self.set_channel_gains(gains)
        self.set_channel_offsets(offsets)
        if locations is not None:
            self.set_channel_locations(locations)
        if groups is not None:
            self.set_channel_groups(groups)

        # fetch and add additional recording properties
        if load_channel_properties:
            if use_pynwb:
                electrodes_table = self._nwbfile.electrodes
                electrodes_indices = self.electrical_series.electrodes.data[:]
                columns = electrodes_table.colnames
            else:
                electrodes_table = self._file["/general/extracellular_ephys/electrodes"]
                electrodes_indices = _retrieve_electrodes_indices_from_electrical_series_backend(
                    self._file, self.electrical_series, self.backend
                )
                columns = electrodes_table.attrs["colnames"]
            properties = self._fetch_other_properties(electrodes_table, electrodes_indices, columns)

            for property_name, property_values in properties.items():
                values = [x.decode("utf-8") if isinstance(x, bytes) else x for x in property_values]
                self.set_property(property_name, values)

        if stream_mode is None and file_path is not None:
            file_path = str(Path(file_path).resolve())

        if stream_mode == "fsspec" and stream_cache_path is not None:
            stream_cache_path = str(Path(self.stream_cache_path).absolute())

        # set serializability bools
        if file is not None:
            # not json serializable if file arg is provided
            self._serializability["json"] = False

        if storage_options is not None and stream_mode == "zarr":
            warnings.warn(
                "The `storage_options` parameter will not be propagated to JSON or pickle files for security reasons, "
                "so the extractor will not be JSON/pickle serializable. Only in-memory mode will be available."
            )
            # not serializable if storage_options is provided
            self._serializability["json"] = False
            self._serializability["pickle"] = False

        self._kwargs = {
            "file_path": file_path,
            "electrical_series_path": self.electrical_series_path,
            "load_time_vector": load_time_vector,
            "samples_for_rate_estimation": samples_for_rate_estimation,
            "stream_mode": stream_mode,
            "load_channel_properties": load_channel_properties,
            "storage_options": storage_options,
            "cache": cache,
            "stream_cache_path": stream_cache_path,
            "file": file,
        }

        # Set extra requirements for the extractor, so they can be installed when using docker
        if use_pynwb:
            self.extra_requirements.append("pynwb")
        else:
            if self.backend == "hdf5":
                self.extra_requirements.append("h5py")
            if self.backend == "zarr":
                self.extra_requirements.append("zarr")
        if self.stream_mode == "fsspec":
            self.extra_requirements.append("fsspec")
        if self.stream_mode == "remfile":
            self.extra_requirements.append("remfile")

    def _fetch_recording_segment_info_pynwb(self, file, cache, load_time_vector, samples_for_rate_estimation):
        self._nwbfile = read_nwbfile(
            backend=self.backend,
            file_path=self.file_path,
            file=file,
            stream_mode=self.stream_mode,
            cache=cache,
            stream_cache_path=self.stream_cache_path,
        )
        electrical_series = _retrieve_electrical_series_pynwb(self._nwbfile, self.electrical_series_path)
        # The indices in the electrode table corresponding to this electrical series
        electrodes_indices = electrical_series.electrodes.data[:]
        # The table for all the electrodes in the nwbfile
        electrodes_table = self._nwbfile.electrodes

        sampling_frequency, times_kwargs = _fetch_time_info_pynwb(
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

        return channel_ids, sampling_frequency, dtype, electrical_series_data, times_kwargs

    def _fetch_recording_segment_info_backend(self, file, cache, load_time_vector, samples_for_rate_estimation):
        open_file = read_file_from_backend(
            file_path=self.file_path,
            file=file,
            stream_mode=self.stream_mode,
            cache=cache,
            stream_cache_path=self.stream_cache_path,
        )

        # If the electrical_series_path is not given, `_find_neurodata_type_from_backend` will be called
        # And returns a list with the electrical_series_paths available in the file.
        # If there is only one electrical series, the electrical_series_path is set to the name of the series,
        # otherwise an error is raised.
        if self.electrical_series_path is None:
            available_electrical_series = _find_neurodata_type_from_backend(
                open_file, neurodata_type="ElectricalSeries", backend=self.backend
            )
            # if electrical_series_path is None:
            if len(available_electrical_series) == 1:
                self.electrical_series_path = available_electrical_series[0]
            else:
                raise ValueError(
                    "Multiple ElectricalSeries found in the file. "
                    "Please specify the 'electrical_series_path' argument:"
                    f"Available options are: {available_electrical_series}."
                )

        # Open the electrical series. In case of failure, raise an error with the available options.
        try:
            electrical_series = open_file[self.electrical_series_path]
        except KeyError:
            available_electrical_series = _find_neurodata_type_from_backend(
                open_file, neurodata_type="ElectricalSeries", backend=self.backend
            )
            raise ValueError(
                f"{self.electrical_series_path} not found in the NWB file!"
                f"Available options are: {available_electrical_series}."
            )
        electrodes_indices = _retrieve_electrodes_indices_from_electrical_series_backend(
            open_file, electrical_series, self.backend
        )
        # The table for all the electrodes in the nwbfile
        electrodes_table = open_file["/general/extracellular_ephys/electrodes"]
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
            times_kwargs = dict(time_vector=electrical_series["timestamps"])
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
        self._file = open_file

        return channel_ids, sampling_frequency, dtype, electrical_series_data, times_kwargs

    def _fetch_locations_and_groups(self, electrodes_table, electrodes_indices):
        # Channel locations
        locations = None
        if "rel_x" in electrodes_table:
            if "rel_y" in electrodes_table:
                ndim = 3 if "rel_z" in electrodes_table else 2
                locations = np.zeros((self.get_num_channels(), ndim), dtype=float)
                locations[:, 0] = electrodes_table["rel_x"][electrodes_indices]
                locations[:, 1] = electrodes_table["rel_y"][electrodes_indices]
                if "rel_z" in electrodes_table:
                    locations[:, 2] = electrodes_table["rel_z"][electrodes_indices]

        # Channel groups
        groups = None
        if "group_name" in electrodes_table:
            groups = electrodes_table["group_name"][electrodes_indices][:]
        if groups is not None:
            groups = np.array([x.decode("utf-8") if isinstance(x, bytes) else x for x in groups])
        return locations, groups

    def _fetch_other_properties(self, electrodes_table, electrodes_indices, columns):
        #########
        # Extract and re-name properties from nwbfile TODO: Should be a function
        ########
        properties = dict()
        properties_to_skip = [
            "id",
            "rel_x",
            "rel_y",
            "rel_z",
            "group",
            "group_name",
            "channel_name",
            "offset",
        ]
        rename_properties = dict(location="brain_area")

        for column in columns:
            if column in properties_to_skip:
                continue
            else:
                column_name = rename_properties.get(column, column)
                properties[column_name] = electrodes_table[column][electrodes_indices]

        return properties

    def _fetch_main_properties_pynwb(self):
        """
        Fetches the main properties from the NWBFile and stores them in the RecordingExtractor, including:

        - gains
        - offsets
        - locations
        - groups
        """
        electrodes_indices = self.electrical_series.electrodes.data[:]
        electrodes_table = self._nwbfile.electrodes

        # Channels gains - for RecordingExtractor, these are values to cast traces to uV
        gains = self.electrical_series.conversion * 1e6
        if self.electrical_series.channel_conversion is not None:
            gains = self.electrical_series.conversion * self.electrical_series.channel_conversion[:] * 1e6

        # Channel offsets
        offset = self.electrical_series.offset if hasattr(self.electrical_series, "offset") else 0
        if offset == 0 and "offset" in electrodes_table:
            offset = electrodes_table["offset"].data[electrodes_indices]
        offsets = offset * 1e6

        locations, groups = self._fetch_locations_and_groups(electrodes_table, electrodes_indices)

        return gains, offsets, locations, groups

    def _fetch_main_properties_backend(self):
        """
        Fetches the main properties from the NWBFile and stores them in the RecordingExtractor, including:

        - gains
        - offsets
        - locations
        - groups
        """
        electrodes_indices = _retrieve_electrodes_indices_from_electrical_series_backend(
            self._file, self.electrical_series, self.backend
        )
        electrodes_table = self._file["/general/extracellular_ephys/electrodes"]

        # Channels gains - for RecordingExtractor, these are values to cast traces to uV
        data_attributes = self.electrical_series["data"].attrs
        electrical_series_conversion = data_attributes["conversion"]
        gains = electrical_series_conversion * 1e6
        channel_conversion = self.electrical_series.get("channel_conversion", None)
        if channel_conversion:
            gains *= self.electrical_series["channel_conversion"][:]

        # Channel offsets
        offset = data_attributes["offset"] if "offset" in data_attributes else 0
        if offset == 0 and "offset" in electrodes_table:
            offset = electrodes_table["offset"][electrodes_indices]
        offsets = offset * 1e6

        # Channel locations and groups
        locations, groups = self._fetch_locations_and_groups(electrodes_table, electrodes_indices)

        return gains, offsets, locations, groups

    @staticmethod
    def fetch_available_electrical_series_paths(
        file_path: str | Path,
        stream_mode: Optional[Literal["fsspec", "remfile", "zarr"]] = None,
        storage_options: dict | None = None,
    ) -> list[str]:
        """
        Retrieves the paths to all ElectricalSeries objects within a neurodata file.

        Parameters
        ----------
        file_path : str | Path
            The path to the neurodata file to be analyzed.
        stream_mode : "fsspec" | "remfile" | "zarr" | None, optional
            Determines the streaming mode for reading the file. Use this for optimized reading from
            different sources, such as local disk or remote servers.
        storage_options : dict | None = None,
            These are the additional kwargs (e.g. AWS credentials) that are passed to the zarr.open convenience function.
            This is only used on the "zarr" stream_mode.
        Returns
        -------
        list of str
            A list of paths to all ElectricalSeries objects found in the file.


        Notes
        -----
        The paths are returned as strings, and can be used to load the desired ElectricalSeries object.
        Examples of paths are:
            - "acquisition/ElectricalSeries1"
            - "acquisition/ElectricalSeries2"
            - "processing/ecephys/LFP/ElectricalSeries1"
            - "processing/my_custom_module/MyContainer/ElectricalSeries2"
        """

        if stream_mode is None:
            backend = _get_backend_from_local_file(file_path)
        else:
            if stream_mode == "zarr":
                backend = "zarr"
            else:
                backend = "hdf5"

        file_handle = read_file_from_backend(
            file_path=file_path,
            stream_mode=stream_mode,
            storage_options=storage_options,
        )

        electrical_series_paths = _find_neurodata_type_from_backend(
            file_handle,
            neurodata_type="ElectricalSeries",
            backend=backend,
        )
        return electrical_series_paths


class NwbRecordingSegment(BaseRecordingSegment):
    def __init__(self, electrical_series_data, times_kwargs):
        BaseRecordingSegment.__init__(self, **times_kwargs)
        self.electrical_series_data = electrical_series_data
        self._num_samples = self.electrical_series_data.shape[0]

    def get_num_samples(self):
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex : Number of samples in the signal block
        """
        return self._num_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
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


class NwbSortingExtractor(BaseSorting, _BaseNWBExtractor):
    """Load an NWBFile as a SortingExtractor.
    Parameters
    ----------
    file_path : str or Path
        Path to NWB file.
    electrical_series_path : str or None, default: None
        The name of the ElectricalSeries (if multiple ElectricalSeries are present).
    sampling_frequency : float or None, default: None
        The sampling frequency in Hz (required if no ElectricalSeries is available).
    unit_table_path : str or None, default: "units"
        The path of the unit table in the NWB file.
    samples_for_rate_estimation : int, default: 100000
        The number of timestamp samples to use to estimate the rate.
        Used if "rate" is not specified in the ElectricalSeries.
    stream_mode : "fsspec" | "remfile" | "zarr" | None, default: None
        The streaming mode to use. If None it assumes the file is on the local disk.
    stream_cache_path : str or Path or None, default: None
        Local path for caching. If None it uses the system temporary directory.
    load_unit_properties : bool, default: True
        If True, all the unit properties are loaded from the NWB file and stored as properties.
    t_start : float or None, default: None
        This is the time at which the corresponding ElectricalSeries start. NWB stores its spikes as times
        and the `t_start` is used to convert the times to seconds. Concrently, the returned frames are computed as:

        `frames = (times - t_start) * sampling_frequency`.

        As SpikeInterface always considers the first frame to be at the beginning of the recording independently
        of the `t_start`.

        When a `t_start` is not provided it will be inferred from the corresponding ElectricalSeries with name equal
        to `electrical_series_path`. The `t_start` then will be either the `ElectricalSeries.starting_time` or the
        first timestamp in the `ElectricalSeries.timestamps`.
    cache : bool, default: False
        If True, the file is cached in the file passed to stream_cache_path
        if False, the file is not cached.
    storage_options : dict | None = None,
        These are the additional kwargs (e.g. AWS credentials) that are passed to the zarr.open convenience function.
        This is only used on the "zarr" stream_mode.
    use_pynwb : bool, default: False
        Uses the pynwb library to read the NWB file. Setting this to False, the default, uses h5py
        to read the file. Using h5py can improve performance by bypassing some of the PyNWB validations.

    Returns
    -------
    sorting : NwbSortingExtractor
        The sorting extractor for the NWB file.
    """

    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(
        self,
        file_path: str | Path,
        electrical_series_path: str | None = None,
        sampling_frequency: float | None = None,
        samples_for_rate_estimation: int = 1_000,
        stream_mode: str | None = None,
        stream_cache_path: str | Path | None = None,
        load_unit_properties: bool = True,
        unit_table_path: str = "units",
        *,
        t_start: float | None = None,
        cache: bool = False,
        storage_options: dict | None = None,
        use_pynwb: bool = False,
    ):

        if stream_mode == "ros3":
            warnings.warn(
                "The 'ros3' stream_mode is deprecated and will be removed in version 0.103.0. "
                "Use 'fsspec' stream_mode instead.",
                DeprecationWarning,
            )

        self.stream_mode = stream_mode
        self.stream_cache_path = stream_cache_path
        self.electrical_series_path = electrical_series_path
        self.file_path = file_path
        self.t_start = t_start
        self.provided_or_electrical_series_sampling_frequency = sampling_frequency
        self.storage_options = storage_options
        self.units_table = None

        if self.stream_mode is None:
            self.backend = _get_backend_from_local_file(file_path)
        else:
            if self.stream_mode == "zarr":
                self.backend = "zarr"
            else:
                self.backend = "hdf5"

        if use_pynwb:
            try:
                import pynwb
            except ImportError:
                raise ImportError(self.installation_mesg)

            unit_ids, spike_times_data, spike_times_index_data = self._fetch_sorting_segment_info_pynwb(
                unit_table_path=unit_table_path, samples_for_rate_estimation=samples_for_rate_estimation, cache=cache
            )
        else:
            unit_ids, spike_times_data, spike_times_index_data = self._fetch_sorting_segment_info_backend(
                unit_table_path=unit_table_path, samples_for_rate_estimation=samples_for_rate_estimation, cache=cache
            )

        BaseSorting.__init__(
            self, sampling_frequency=self.provided_or_electrical_series_sampling_frequency, unit_ids=unit_ids
        )

        sorting_segment = NwbSortingSegment(
            spike_times_data=spike_times_data,
            spike_times_index_data=spike_times_index_data,
            sampling_frequency=self.sampling_frequency,
            t_start=self.t_start,
        )
        self.add_sorting_segment(sorting_segment)

        # fetch and add sorting properties
        if load_unit_properties:
            if use_pynwb:
                columns = [c.name for c in self.units_table.columns]
                self.extra_requirements.append("pynwb")
            else:
                columns = list(self.units_table.keys())
                self.extra_requirements.append("h5py")
            properties = self._fetch_properties(columns)
            for property_name, property_values in properties.items():
                values = [x.decode("utf-8") if isinstance(x, bytes) else x for x in property_values]
                self.set_property(property_name, values)
        if stream_mode is not None:
            self.extra_requirements.append(stream_mode)

        if stream_mode is None and file_path is not None:
            file_path = str(Path(file_path).resolve())

        if storage_options is not None and stream_mode == "zarr":
            warnings.warn(
                "The `storage_options` parameter will not be propagated to JSON or pickle files for security reasons, "
                "so the extractor will not be JSON/pickle serializable. Only in-memory mode will be available."
            )
            # not serializable if storage_options is provided
            self._serializability["json"] = False
            self._serializability["pickle"] = False

        self._kwargs = {
            "file_path": file_path,
            "electrical_series_path": self.electrical_series_path,
            "sampling_frequency": sampling_frequency,
            "samples_for_rate_estimation": samples_for_rate_estimation,
            "cache": cache,
            "stream_mode": stream_mode,
            "stream_cache_path": stream_cache_path,
            "storage_options": storage_options,
            "load_unit_properties": load_unit_properties,
            "t_start": self.t_start,
        }

    def _fetch_sorting_segment_info_pynwb(
        self, unit_table_path: str = None, samples_for_rate_estimation: int = 1000, cache: bool = False
    ):
        self._nwbfile = read_nwbfile(
            backend=self.backend,
            file_path=self.file_path,
            stream_mode=self.stream_mode,
            cache=cache,
            stream_cache_path=self.stream_cache_path,
            storage_options=self.storage_options,
        )

        timestamps = None
        if self.provided_or_electrical_series_sampling_frequency is None:
            # defines the electrical series from where the sorting came from
            # important to know the sampling_frequency
            self.electrical_series = _retrieve_electrical_series_pynwb(self._nwbfile, self.electrical_series_path)
            # get rate
            if self.electrical_series.rate is not None:
                self.provided_or_electrical_series_sampling_frequency = self.electrical_series.rate
                self.t_start = self.electrical_series.starting_time
            else:
                if hasattr(self.electrical_series, "timestamps"):
                    if self.electrical_series.timestamps is not None:
                        timestamps = self.electrical_series.timestamps
                        self.provided_or_electrical_series_sampling_frequency = 1 / np.median(
                            np.diff(timestamps[:samples_for_rate_estimation])
                        )
                        self.t_start = timestamps[0]
        assert (
            self.provided_or_electrical_series_sampling_frequency is not None
        ), "Couldn't load sampling frequency. Please provide it with the 'sampling_frequency' argument"
        assert (
            self.t_start is not None
        ), "Couldn't load a starting time for the sorting. Please provide it with the 't_start' argument"
        if unit_table_path == "units":
            units_table = self._nwbfile.units
        else:
            units_table = _retrieve_unit_table_pynwb(self._nwbfile, unit_table_path=unit_table_path)

        name_to_column_data = {c.name: c for c in units_table.columns}
        spike_times_data = name_to_column_data.pop("spike_times").data
        spike_times_index_data = name_to_column_data.pop("spike_times_index").data

        units_ids = name_to_column_data.pop("unit_name", None)
        if units_ids is None:
            units_ids = units_table["id"].data

        # need this for later
        self.units_table = units_table

        return units_ids, spike_times_data, spike_times_index_data

    def _fetch_sorting_segment_info_backend(
        self, unit_table_path: str = None, samples_for_rate_estimation: int = 1000, cache: bool = False
    ):
        open_file = read_file_from_backend(
            file_path=self.file_path,
            stream_mode=self.stream_mode,
            cache=cache,
            stream_cache_path=self.stream_cache_path,
            storage_options=self.storage_options,
        )

        timestamps = None

        if self.provided_or_electrical_series_sampling_frequency is None or self.t_start is None:
            # defines the electrical series from where the sorting came from
            # important to know the sampling_frequency
            available_electrical_series = _find_neurodata_type_from_backend(
                open_file, neurodata_type="ElectricalSeries", backend=self.backend
            )
            if self.electrical_series_path is None:
                if len(available_electrical_series) == 1:
                    self.electrical_series_path = available_electrical_series[0]
                else:
                    raise ValueError(
                        "Multiple ElectricalSeries found in the file. "
                        "Please specify the 'electrical_series_path' argument:"
                        f"Available options are: {available_electrical_series}."
                    )
            else:
                if self.electrical_series_path not in available_electrical_series:
                    raise ValueError(
                        f"'{self.electrical_series_path}' not found in the file. "
                        f"Available options are: {available_electrical_series}"
                    )
            electrical_series = open_file[self.electrical_series_path]

            # Get sampling frequency
            if "starting_time" in electrical_series.keys():
                self.t_start = electrical_series["starting_time"][()]
                self.provided_or_electrical_series_sampling_frequency = electrical_series["starting_time"].attrs["rate"]
            elif "timestamps" in electrical_series.keys():
                timestamps = electrical_series["timestamps"][:]
                self.t_start = timestamps[0]
                self.provided_or_electrical_series_sampling_frequency = 1.0 / np.median(
                    np.diff(timestamps[:samples_for_rate_estimation])
                )

        assert (
            self.provided_or_electrical_series_sampling_frequency is not None
        ), "Couldn't load sampling frequency. Please provide it with the 'sampling_frequency' argument"
        assert (
            self.t_start is not None
        ), "Couldn't load a starting time for the sorting. Please provide it with the 't_start' argument"

        if unit_table_path is None:
            available_unit_table_paths = _find_neurodata_type_from_backend(
                open_file, neurodata_type="Units", backend=self.backend
            )
            if len(available_unit_table_paths) == 1:
                unit_table_path = available_unit_table_paths[0]
            else:
                raise ValueError(
                    "Multiple Units tables found in the file. "
                    "Please specify the 'unit_table_path' argument:"
                    f"Available options are: {available_unit_table_paths}."
                )
        # Try to open the unit table. If it fails, raise an error with the available options.
        try:
            units_table = open_file[unit_table_path]
        except KeyError:
            available_unit_table_paths = _find_neurodata_type_from_backend(
                open_file, neurodata_type="Units", backend=self.backend
            )
            raise ValueError(
                f"{unit_table_path} not found in the NWB file!" f"Available options are: {available_unit_table_paths}."
            )
        self.units_table_location = unit_table_path
        units_table = open_file[self.units_table_location]

        spike_times_data = units_table["spike_times"]
        spike_times_index_data = units_table["spike_times_index"]

        if "unit_name" in units_table:
            unit_ids = units_table["unit_name"]
        else:
            unit_ids = units_table["id"]

        decode_to_string = lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        unit_ids = [decode_to_string(id) for id in unit_ids]

        # need this for later
        self.units_table = units_table

        return unit_ids, spike_times_data, spike_times_index_data

    def _fetch_properties(self, columns):
        units_table = self.units_table

        properties_to_skip = ["spike_times", "spike_times_index", "unit_name", "id"]
        index_columns = [name for name in columns if name.endswith("_index")]
        nested_ragged_array_properties = [name for name in columns if f"{name}_index_index" in columns]

        # Filter those properties that are nested ragged arrays
        skip_properties = properties_to_skip + index_columns + nested_ragged_array_properties
        properties_to_add = [name for name in columns if name not in skip_properties]

        properties = dict()
        for property_name in properties_to_add:
            data = units_table[property_name][:]
            corresponding_index_name = f"{property_name}_index"
            not_ragged_array = corresponding_index_name not in columns
            if not_ragged_array:
                values = data[:]
            else:  # TODO if we want we could make this recursive to handle nested ragged arrays
                data_index = units_table[corresponding_index_name]
                if hasattr(data_index, "data"):
                    # for pynwb we need to get the data from the data attribute
                    data_index = data_index.data[:]
                else:
                    data_index = data_index[:]
                index_spacing = np.diff(data_index, prepend=0)
                all_index_spacing_are_the_same = np.unique(index_spacing).size == 1
                if all_index_spacing_are_the_same:
                    if hasattr(units_table[corresponding_index_name], "data"):
                        # ragged array indexing is handled by pynwb
                        values = data
                    else:
                        # ravel array based on data_index
                        start_indices = [0] + list(data_index[:-1])
                        end_indices = list(data_index)
                        values = [
                            data[start_index:end_index] for start_index, end_index in zip(start_indices, end_indices)
                        ]
                else:
                    warnings.warn(f"Skipping {property_name} because of unequal shapes across units")
                    continue
            properties[property_name] = values

        return properties


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


def _find_timeseries_from_backend(group, path="", result=None, backend="hdf5"):
    """
    Recursively searches for groups with TimeSeries neurodata_type in hdf5 or zarr object,
    and returns a list with their paths.
    """
    if backend == "hdf5":
        import h5py

        group_class = h5py.Group
    else:
        import zarr

        group_class = zarr.Group

    if result is None:
        result = []

    for name, value in group.items():
        if isinstance(value, group_class):
            current_path = f"{path}/{name}" if path else name
            if value.attrs.get("neurodata_type") == "TimeSeries":
                result.append(current_path)
            _find_timeseries_from_backend(value, current_path, result, backend)
    return result


class NwbTimeSeriesExtractor(BaseRecording, _BaseNWBExtractor):
    """Load a TimeSeries from an NWBFile as a RecordingExtractor.

    Parameters
    ----------
    file_path : str | Path | None
        Path to NWB file or an s3 URL. Use this parameter to specify the file location
        if not using the `file` parameter.
    timeseries_path : str | None
        The path to the TimeSeries object within the NWB file. This parameter is required
        when the NWB file contains multiple TimeSeries objects. The path corresponds to
        the location within the NWB file hierarchy, e.g. 'acquisition/MyTimeSeries'.
    load_time_vector : bool, default: False
        If True, the time vector is loaded into the recording object. Useful when
        precise timing information is needed.
    samples_for_rate_estimation : int, default: 1000
        The number of timestamps used for estimating the sampling rate when
        timestamps are used instead of a fixed rate.
    stream_mode : Literal["fsspec", "remfile", "zarr"] | None, default: None
        Determines the streaming mode for reading the file.
    file : BinaryIO | None, default: None
        A file-like object representing the NWB file. Use this parameter if you have
        an in-memory representation of the NWB file instead of a file path given by `file_path`.
    cache : bool, default: False
        If True, the file is cached locally when using streaming.
    stream_cache_path : str | Path | None, default: None
        Local path for caching. Only used if `cache` is True.
    storage_options : dict | None, default: None
        Additional kwargs (e.g. AWS credentials) passed to zarr.open. Only used with
        "zarr" stream_mode.
    use_pynwb : bool, default: False
        If True, uses pynwb library to read the NWB file. Default False uses h5py/zarr
        directly for better performance.

    Returns
    -------
    recording : NwbTimeSeriesExtractor
        A recording extractor containing the TimeSeries data.
    """

    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(
        self,
        file_path: str | Path | None = None,
        timeseries_path: str | None = None,
        load_time_vector: bool = False,
        samples_for_rate_estimation: int = 1_000,
        stream_mode: Optional[Literal["fsspec", "remfile", "zarr"]] = None,
        stream_cache_path: str | Path | None = None,
        *,
        file: BinaryIO | None = None,
        cache: bool = False,
        storage_options: dict | None = None,
        use_pynwb: bool = False,
    ):
        if file_path is not None and file is not None:
            raise ValueError("Provide either file_path or file, not both")
        if file_path is None and file is None:
            raise ValueError("Provide either file_path or file")

        self.file_path = file_path
        self.stream_mode = stream_mode
        self.stream_cache_path = stream_cache_path
        self.storage_options = storage_options
        self.timeseries_path = timeseries_path

        if self.stream_mode is None and file is None:
            self.backend = _get_backend_from_local_file(file_path)
        else:
            self.backend = "zarr" if self.stream_mode == "zarr" else "hdf5"

        if use_pynwb:
            try:
                import pynwb
            except ImportError:
                raise ImportError(self.installation_mesg)

            channel_ids, sampling_frequency, dtype, segment_data, times_kwargs = self._fetch_recording_segment_info(
                file, cache, load_time_vector, samples_for_rate_estimation
            )
        else:
            channel_ids, sampling_frequency, dtype, segment_data, times_kwargs = (
                self._fetch_recording_segment_info_backend(file, cache, load_time_vector, samples_for_rate_estimation)
            )

        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        recording_segment = NwbTimeSeriesSegment(
            timeseries_data=segment_data,
            times_kwargs=times_kwargs,
        )
        self.add_recording_segment(recording_segment)

        if storage_options is not None and stream_mode == "zarr":
            warnings.warn(
                "The `storage_options` parameter will not be propagated to JSON or pickle files "
                "for security reasons, so this extractor will not be JSON/pickle serializable."
            )
            self._serializability["json"] = False
            self._serializability["pickle"] = False

        self._kwargs = {
            "file_path": file_path,
            "timeseries_path": self.timeseries_path,
            "load_time_vector": load_time_vector,
            "samples_for_rate_estimation": samples_for_rate_estimation,
            "stream_mode": stream_mode,
            "storage_options": storage_options,
            "cache": cache,
            "stream_cache_path": stream_cache_path,
            "file": file,
        }

        if use_pynwb:
            self.extra_requirements.append("pynwb")
        else:
            if self.backend == "hdf5":
                self.extra_requirements.append("h5py")
            if self.backend == "zarr":
                self.extra_requirements.append("zarr")

        if self.stream_mode == "fsspec":
            self.extra_requirements.append("fsspec")
        elif self.stream_mode == "remfile":
            self.extra_requirements.append("remfile")

    def _fetch_recording_segment_info(self, file, cache, load_time_vector, samples_for_rate_estimation):
        self._nwbfile = read_nwbfile(
            backend=self.backend,
            file_path=self.file_path,
            file=file,
            stream_mode=self.stream_mode,
            cache=cache,
            stream_cache_path=self.stream_cache_path,
            storage_options=self.storage_options,
        )

        from pynwb.base import TimeSeries

        time_series_dict: dict[str, TimeSeries] = {}

        for item in self._nwbfile.all_children():
            if isinstance(item, TimeSeries):
                time_series_dict[item.data.name.replace("/data", "")[1:]] = item

        if self.timeseries_path is not None:
            if self.timeseries_path not in time_series_dict:
                raise ValueError(f"TimeSeries {self.timeseries_path} not found in file")

        else:
            if len(time_series_dict) == 1:
                self.timeseries_path = list(time_series_dict.keys())[0]
            else:
                raise ValueError(
                    f"Multiple TimeSeries found! Specify 'timeseries_path'. Options: {list(time_series_dict.keys())}"
                )

        timeseries = time_series_dict[self.timeseries_path]

        # Get sampling frequency and timing info
        if hasattr(timeseries, "rate") and timeseries.rate is not None:
            sampling_frequency = timeseries.rate
            t_start = timeseries.starting_time if hasattr(timeseries, "starting_time") else 0
            timestamps = None
        elif hasattr(timeseries, "timestamps"):
            timestamps = timeseries.timestamps
            sampling_frequency = 1.0 / np.median(np.diff(timestamps[:samples_for_rate_estimation]))
            t_start = timestamps[0]

        if load_time_vector and timestamps is not None:
            times_kwargs = dict(time_vector=timestamps)
        else:
            times_kwargs = dict(sampling_frequency=sampling_frequency, t_start=t_start)

        # Create channel IDs based on data shape
        data = timeseries.data
        if data.ndim == 1:
            num_channels = 1
        else:
            num_channels = data.shape[1]
        channel_ids = np.arange(num_channels)
        dtype = data.dtype

        return channel_ids, sampling_frequency, dtype, data, times_kwargs

    def _fetch_recording_segment_info_backend(self, file, cache, load_time_vector, samples_for_rate_estimation):
        open_file = read_file_from_backend(
            file_path=self.file_path,
            file=file,
            stream_mode=self.stream_mode,
            cache=cache,
            stream_cache_path=self.stream_cache_path,
            storage_options=self.storage_options,
        )

        # If timeseries_path not provided, find all TimeSeries objects
        if self.timeseries_path is None:
            available_timeseries = _find_timeseries_from_backend(open_file, backend=self.backend)
            if len(available_timeseries) == 1:
                self.timeseries_path = available_timeseries[0]
            else:
                raise ValueError(
                    f"Multiple TimeSeries found! Specify 'timeseries_path'. Options: {available_timeseries}"
                )

        # Get TimeSeries object
        try:
            timeseries = open_file[self.timeseries_path]
        except KeyError:
            available_timeseries = _find_timeseries_from_backend(open_file, backend=self.backend)
            raise ValueError(f"{self.timeseries_path} not found! Available options: {available_timeseries}")

        # Get timing information
        if "starting_time" in timeseries:
            t_start = timeseries["starting_time"][()]
            sampling_frequency = timeseries["starting_time"].attrs["rate"]
            timestamps = None
        elif "timestamps" in timeseries:
            timestamps = timeseries["timestamps"][:]
            sampling_frequency = 1.0 / np.median(np.diff(timestamps[:samples_for_rate_estimation]))
            t_start = timestamps[0]
        else:
            raise ValueError("TimeSeries must have either starting_time or timestamps")

        if load_time_vector and timestamps is not None:
            times_kwargs = dict(time_vector=timestamps)
        else:
            times_kwargs = dict(sampling_frequency=sampling_frequency, t_start=t_start)

        # Create channel IDs based on data shape
        data = timeseries["data"]
        if data.ndim == 1:
            num_channels = 1
        else:
            num_channels = data.shape[1]
        channel_ids = np.arange(num_channels)
        dtype = data.dtype

        # Store for later use
        self.timeseries = timeseries
        self._file = open_file

        return channel_ids, sampling_frequency, dtype, data, times_kwargs

    @staticmethod
    def fetch_available_timeseries_paths(
        file_path: str | Path,
        stream_mode: Optional[Literal["fsspec", "remfile", "zarr"]] = None,
        storage_options: dict | None = None,
    ) -> list[str]:
        """
        Get paths to all TimeSeries objects in a neurodata file.

        Parameters
        ----------
        file_path : str | Path
            Path to the NWB file.
        stream_mode : str | None
            Streaming mode for reading remote files.
        storage_options : dict | None
            Additional options for zarr storage.

        Returns
        -------
        list[str]
            List of paths to TimeSeries objects.
        """
        if stream_mode is None:
            backend = _get_backend_from_local_file(file_path)
        else:
            backend = "zarr" if stream_mode == "zarr" else "hdf5"

        file_handle = read_file_from_backend(
            file_path=file_path,
            stream_mode=stream_mode,
            storage_options=storage_options,
        )

        timeseries_paths = _find_timeseries_from_backend(
            file_handle,
            backend=backend,
        )
        return timeseries_paths


class NwbTimeSeriesSegment(BaseRecordingSegment):
    """Segment class for NwbTimeSeriesExtractor."""

    def __init__(self, timeseries_data, times_kwargs):
        BaseRecordingSegment.__init__(self, **times_kwargs)
        self.timeseries_data = timeseries_data
        self._num_samples = self.timeseries_data.shape[0]

    def get_num_samples(self):
        """Returns the number of samples in this signal block."""
        return self._num_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        """
        Extract traces from the TimeSeries between start_frame and end_frame for specified channels.

        Parameters
        ----------
        start_frame : int
            Start frame of the slice to extract.
        end_frame : int
            End frame of the slice to extract.
        channel_indices : array-like
            Channel indices to extract.

        Returns
        -------
        traces : np.ndarray
            Extracted traces of shape (num_frames, num_channels)
        """
        if self.timeseries_data.ndim == 1:
            traces = self.timeseries_data[start_frame:end_frame][:, np.newaxis]
        elif isinstance(channel_indices, slice):
            traces = self.timeseries_data[start_frame:end_frame, channel_indices]
        else:
            # channel_indices is np.ndarray
            if np.array(channel_indices).size > 1 and np.any(np.diff(channel_indices) < 0):
                # get around h5py constraint that it does not allow datasets
                # to be indexed out of order
                sorted_channel_indices = np.sort(channel_indices)
                resorted_indices = np.array([list(sorted_channel_indices).index(ch) for ch in channel_indices])
                recordings = self.timeseries_data[start_frame:end_frame, sorted_channel_indices]
                traces = recordings[:, resorted_indices]
            else:
                traces = self.timeseries_data[start_frame:end_frame, channel_indices]

        return traces


# Create the reading function


read_nwb_recording = define_function_from_class(source_class=NwbRecordingExtractor, name="read_nwb_recording")
read_nwb_sorting = define_function_from_class(source_class=NwbSortingExtractor, name="read_nwb_sorting")
read_nwb_timeseries = define_function_from_class(source_class=NwbTimeSeriesExtractor, name="read_nwb_timeseries")


def read_nwb(file_path, load_recording=True, load_sorting=False, electrical_series_path=None):
    """Reads NWB file into SpikeInterface extractors.

    Parameters
    ----------
    file_path : str or Path
        Path to NWB file.
    load_recording : bool, default: True
        If True, the recording object is loaded.
    load_sorting : bool, default: False
        If True, the recording object is loaded.
    electrical_series_path : str or None, default: None
        The name of the ElectricalSeries (if multiple ElectricalSeries are present)

    Returns
    -------
    extractors : extractor or tuple
        Single RecordingExtractor/SortingExtractor or tuple with both
        (depending on "load_recording"/"load_sorting") arguments.
    """
    outputs = ()
    if load_recording:
        rec = read_nwb_recording(file_path, electrical_series_path=electrical_series_path)
        outputs = outputs + (rec,)
    if load_sorting:
        sorting = read_nwb_sorting(file_path, electrical_series_path=electrical_series_path)
        outputs = outputs + (sorting,)

    if len(outputs) == 1:
        outputs = outputs[0]

    return outputs
