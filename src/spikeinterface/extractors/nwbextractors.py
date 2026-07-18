from pathlib import Path
from typing import List, Optional, Literal, Dict, BinaryIO
import sys
import warnings
import importlib.util

import numpy as np

from spikeinterface import get_global_tmp_folder
from spikeinterface.core import (
    BaseRecording,
    BaseRecordingSegment,
    BaseSorting,
    BaseSortingSegment,
    SortingAnalyzer,
    get_default_analyzer_extension_params,
)
from spikeinterface.core.base import minimum_spike_dtype
from spikeinterface.core.core_tools import define_function_from_class

if importlib.util.find_spec("pynwb") is not None:
    HAVE_PYNWB = True
else:
    HAVE_PYNWB = False


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


class _NWBReader:
    """Read an NWB recording (its ElectricalSeries and electrodes table) for one storage format.

    SpikeInterface can read an NWB file three ways; the choice is stored in ``self.reading_method``
    (``"use_pynwb"`` / ``"use_hdf5"`` / ``"use_zarr"``), and every method switches on that one
    variable, so the extractor talks to one object and never branches on format itself.

    ``load`` opens the file handle, locates the ElectricalSeries, and binds ``self.series``,
    ``self.electrodes_table`` and ``self.electrodes_indices``; ``close`` releases the handle.
    """

    def __init__(self, *, reading_method, electrical_series_path=None, unit_table_path=None, timeseries_path=None):
        assert reading_method in ("use_pynwb", "use_hdf5", "use_zarr"), f"Unknown reading_method {reading_method}"
        self.reading_method = reading_method
        self.electrical_series_path = electrical_series_path
        self.unit_table_path = unit_table_path
        self.timeseries_path = timeseries_path
        self.file = None
        self.nwbfile = None
        self.series = None
        self.electrodes_table = None
        self.electrodes_indices = None
        self.units_table = None

    @staticmethod
    def _storage_backend(file_path=None, file=None, stream_mode=None):
        # Whether the bytes are hdf5 or zarr; needed to pick pynwb's IO and to scan neurodata types.
        if stream_mode is None and file is None:
            return _get_backend_from_local_file(file_path)
        return "zarr" if stream_mode == "zarr" else "hdf5"

    def _open_handle(self, *, file_path, file, stream_mode, cache, stream_cache_path, storage_options=None):
        # Open the file handle: a pynwb NWBFile, or a raw h5py.File / zarr group.
        if self.reading_method == "use_pynwb":
            self.nwbfile = read_nwbfile(
                backend=self._storage_backend(file_path, file, stream_mode),
                file_path=file_path,
                file=file,
                stream_mode=stream_mode,
                cache=cache,
                stream_cache_path=stream_cache_path,
                storage_options=storage_options,
            )
        else:
            self.file = read_file_from_backend(
                file_path=file_path,
                file=file,
                stream_mode=stream_mode,
                cache=cache,
                stream_cache_path=stream_cache_path,
                storage_options=storage_options,
            )

    def _read_column(self, table, name, indices=None):
        # Materialize a table column to numpy (h5py only fancy-indexes increasing, GH-4619), optionally
        # reorder to `indices`, and decode HDF5 byte-strings once on behalf of every caller.
        column = table[name]
        # A DynamicTableRegion column (e.g. IBL's `max_electrode`) holds a per-row index into another
        # table. On the pynwb path `column[:]` resolves the region into a DataFrame of the referenced
        # rows, so read the raw per-row indices (`column.data`) instead. The raw hdf5/zarr dataset is
        # already those indices, so nothing is unwrapped there. This keeps every backend returning the
        # same plain array. The pynwb-only import is safe here: this branch runs only when
        # reading_method is "use_pynwb", where pynwb (and hdmf) are present.
        if self.reading_method == "use_pynwb":
            from hdmf.common import DynamicTableRegion

            if isinstance(column, DynamicTableRegion):
                column = column.data
        values = np.asarray(column[:])
        if indices is not None:
            values = values[indices]
        if values.dtype.kind in ("S", "O"):
            values = np.array([v.decode("utf-8") if isinstance(v, bytes) else v for v in values])
        return values

    def _read_table_property(self, table, name, columns, indices=None):
        # Load one DynamicTable column as a per-row property, or return None if it cannot be one.
        # Scalar columns are read directly (and optionally reordered to `indices`). Ragged columns
        # (those with a `<name>_index`) are decided from their index alone.
        #
        # Performance, read the (tiny) index before the (potentially huge) data. Whether a ragged
        # column can be a per-row property is fully determined by its index: unequal per-row lengths
        # cannot stack into a per-row array, and a ragged column cannot be reordered to a channel
        # subset. So we inspect the index first and, when the column is unusable, return None WITHOUT
        # ever reading its data. This matters because ragged columns are frequently per-spike, e.g. a
        # Units table's spike amplitudes or spike depths, which are as long as the whole spike vector
        # (tens of millions of values, hundreds of MB), while the index is only `num_rows` values. The
        # old code read the full data and then discarded it after finding the shapes unequal. On local
        # disk that waste is nearly invisible; over a network file (remfile / streaming from DANDI) it
        # is severe, because each of those columns becomes many range requests for bytes we throw away.
        # For a typical IBL session this avoided reading ~330 MB (two ~20M-long per-spike columns) that
        # were only ever skipped. Do not reorder this to read `data` before checking the index.
        index_name = f"{name}_index"
        if index_name not in columns:
            return self._read_column(table, name, indices)

        index = table[index_name]
        index = index.data[:] if hasattr(index, "data") else index[:]
        if indices is not None or np.unique(np.diff(index, prepend=0)).size != 1:
            return None
        data = table[name]
        if hasattr(data, "data"):  # pynwb resolves the ragged structure when sliced
            return data[:]
        data = np.asarray(data[:])
        starts = np.concatenate([[0], index[:-1]])
        return [data[start:end] for start, end in zip(starts, index)]

    def close(self):
        # Release the open handle (raw hdf5 / zarr store, or the pynwb read IO).
        if self.file is not None:
            if hasattr(self.file, "store"):  # zarr
                self.file.store.close()
            else:  # hdf5: close every object still open on the file id
                import h5py

                for object_id in h5py.h5f.get_obj_ids(self.file.id, types=h5py.h5f.OBJ_ALL):
                    try:
                        object_id.close()
                    except Exception:
                        warnings.warn(f"Error closing object {h5py.h5i.get_name(object_id).decode('utf-8')}")
        elif self.nwbfile is not None:
            io = self.nwbfile.get_read_io()
            if io is not None:
                io.close()

    def __del__(self):
        # Release the file handle on garbage collection.
        if getattr(sys, "meta_path", None) is None:  # avoid import errors during interpreter shutdown
            return
        self.close()

    @staticmethod
    def available_electrical_series(file_path, stream_mode=None, storage_options=None):
        """Paths of every ElectricalSeries in the file, read directly (without pynwb)."""
        backend = _NWBReader._storage_backend(file_path, stream_mode=stream_mode)
        file_handle = read_file_from_backend(
            file_path=file_path, stream_mode=stream_mode, storage_options=storage_options
        )
        return _find_neurodata_type_from_backend(file_handle, neurodata_type="ElectricalSeries", backend=backend)

    def load_recording(
        self, *, file_path=None, file=None, stream_mode=None, cache=False, stream_cache_path=None, storage_options=None
    ):
        # Open the handle and bind the ElectricalSeries + electrodes table.
        self._open_handle(
            file_path=file_path,
            file=file,
            stream_mode=stream_mode,
            cache=cache,
            stream_cache_path=stream_cache_path,
            storage_options=storage_options,
        )
        if self.reading_method == "use_pynwb":
            self.series = _retrieve_electrical_series_pynwb(self.nwbfile, self.electrical_series_path)
            self.electrodes_indices = self.series.electrodes.data[:]
            self.electrodes_table = self.nwbfile.electrodes
        else:
            backend = "zarr" if self.reading_method == "use_zarr" else "hdf5"
            self.series = self._locate_electrical_series(backend)
            self.electrodes_indices = _retrieve_electrodes_indices_from_electrical_series_backend(
                self.file, self.series, backend
            )
            self.electrodes_table = self.file["/general/extracellular_ephys/electrodes"]

    def load_units(
        self, *, file_path=None, file=None, stream_mode=None, cache=False, stream_cache_path=None, storage_options=None
    ):
        # Open the handle and bind the Units table.
        self._open_handle(
            file_path=file_path,
            file=file,
            stream_mode=stream_mode,
            cache=cache,
            stream_cache_path=stream_cache_path,
            storage_options=storage_options,
        )
        if self.reading_method == "use_pynwb":
            if self.unit_table_path == "units":
                self.units_table = self.nwbfile.units
            else:
                self.units_table = _retrieve_unit_table_pynwb(self.nwbfile, unit_table_path=self.unit_table_path)
        else:
            backend = "zarr" if self.reading_method == "use_zarr" else "hdf5"
            self.units_table = self._locate_units_table(backend)

    def _locate_units_table(self, backend):
        # Resolve unit_table_path (auto-discovering it when the file has exactly one Units table)
        # and return the table handle, raising a helpful error that lists the options on failure.
        if self.unit_table_path is None:
            # Fast path: the primary NWB Units table lives at the canonical top-level "units". Check it
            # directly instead of walking the whole HDF5/zarr tree, which over a stream costs one network
            # round-trip per object (hundreds on a rich file, dominating the extractor init). Only fall
            # back to a full search when "units" is absent (a file that stores units in a non-standard
            # location). When "units" is present it is used as-is, so a file with several Units tables is no
            # longer flagged here; pass `unit_table_path` explicitly to select a non-canonical one.
            if "units" in self.file and self.file["units"].attrs.get("neurodata_type") == "Units":
                self.unit_table_path = "units"
            else:
                available = _find_neurodata_type_from_backend(self.file, neurodata_type="Units", backend=backend)
                if len(available) != 1:
                    raise ValueError(
                        "Multiple Units tables found in the file. "
                        "Please specify the 'unit_table_path' argument:"
                        f"Available options are: {available}."
                    )
                self.unit_table_path = available[0]
        try:
            return self.file[self.unit_table_path]
        except KeyError:
            available = _find_neurodata_type_from_backend(self.file, neurodata_type="Units", backend=backend)
            raise ValueError(
                f"{self.unit_table_path} not found in the NWB file!" f"Available options are: {available}."
            )

    @staticmethod
    def available_units_tables(file_path, stream_mode=None, storage_options=None):
        """Paths of every Units table in the file, read directly (without pynwb)."""
        backend = _NWBReader._storage_backend(file_path, stream_mode=stream_mode)
        file_handle = read_file_from_backend(
            file_path=file_path, stream_mode=stream_mode, storage_options=storage_options
        )
        return _find_neurodata_type_from_backend(file_handle, neurodata_type="Units", backend=backend)

    @property
    def units_column_names(self):
        if self.reading_method == "use_pynwb":
            return [column.name for column in self.units_table.columns]
        return list(self.units_table.keys())

    def unit_ids(self):
        if "unit_name" in self.units_column_names:
            return list(self._read_column(self.units_table, "unit_name"))
        if self.reading_method == "use_pynwb":
            return list(np.asarray(self.units_table.id[:]))
        return list(np.asarray(self.units_table["id"][:]))

    def spike_times(self):
        if self.reading_method == "use_pynwb":
            return {column.name: column for column in self.units_table.columns}["spike_times"].data
        return self.units_table["spike_times"]

    def spike_times_index(self):
        if self.reading_method == "use_pynwb":
            return {column.name: column for column in self.units_table.columns}["spike_times_index"].data
        return self.units_table["spike_times_index"]

    def _sampling_frequency_from_units_metadata(self):
        # The Units.resolution attribute documents the precision of the spike times, typically
        # 1 / sampling_rate (in seconds). When present it lets a units-only file (one with no
        # ElectricalSeries) report a sampling frequency. It is spike-native, so it is preferred over
        # the ElectricalSeries rate, which may belong to a series the sorting was not computed from.
        # pynwb exposes it as Units.resolution; the raw hdf5 / zarr layout stores it as an attribute
        # of the spike_times dataset.
        if self.reading_method == "use_pynwb":
            resolution = getattr(self.units_table, "resolution", None)
        else:
            resolution = self.units_table["spike_times"].attrs.get("resolution", None)
        if resolution is None:
            return None
        resolution = float(resolution)
        if not np.isfinite(resolution) or resolution <= 0:
            return None
        return 1.0 / resolution

    def _has_electrical_series(self):
        # Whether the file contains any ElectricalSeries at all. Used to decide, for a sorting, whether
        # there is a recording to align to: if there is none, t_start has no origin and defaults to 0.
        if self.reading_method == "use_pynwb":
            from pynwb.ecephys import ElectricalSeries

            electrical_series = [item for item in self.nwbfile.all_children() if isinstance(item, ElectricalSeries)]
        else:
            backend = "zarr" if self.reading_method == "use_zarr" else "hdf5"
            electrical_series = _find_neurodata_type_from_backend(
                self.file, neurodata_type="ElectricalSeries", backend=backend
            )
        file_has_electrical_series = len(electrical_series) > 0
        return file_has_electrical_series

    def _fetch_unit_properties(self):
        # Return the unit-table columns as a name -> per-unit values mapping in a backend-independent
        # format. Skips id / spike-time columns, index columns, nested ragged arrays, and ragged
        # columns whose per-unit shapes are unequal. The last are detected from the (small) index
        # alone, so their (potentially large, e.g. per-spike) data is never read.
        columns = self.units_column_names

        properties_to_skip = ["spike_times", "spike_times_index", "unit_name", "id"]
        index_columns = [name for name in columns if name.endswith("_index")]
        nested_ragged_array_properties = [name for name in columns if f"{name}_index_index" in columns]
        skip_properties = properties_to_skip + index_columns + nested_ragged_array_properties
        properties_to_add = [name for name in columns if name not in skip_properties]

        properties = dict()
        for property_name in properties_to_add:
            values = self._read_table_property(self.units_table, property_name, columns)
            # A None result is a ragged column with a variable length per unit (typically per-spike
            # data such as spike amplitudes): not a per-unit property, so it is skipped and its data is
            # never read. No warning is emitted, this is expected for normal files and there is nothing
            # for the user to act on.
            if values is not None:
                properties[property_name] = values

        return properties

    def fetch_time_info_from_electrical_series(self, samples_for_rate_estimation):
        # Locate the ElectricalSeries the sorting came from and read its rate, t_start, and timestamps
        # (timestamps is None for a rate-based series, else the per-sample time vector).
        if self.reading_method == "use_pynwb":
            self.series = _retrieve_electrical_series_pynwb(self.nwbfile, self.electrical_series_path)
        else:
            backend = "zarr" if self.reading_method == "use_zarr" else "hdf5"
            self.series = self._locate_electrical_series(backend)
        sampling_frequency, t_start, timestamps = self.time_info(samples_for_rate_estimation)
        return sampling_frequency, t_start, timestamps

    # --- generic TimeSeries (time-series recording) --------------------------------------------
    def load_timeseries(
        self, *, file_path=None, file=None, stream_mode=None, cache=False, stream_cache_path=None, storage_options=None
    ):
        # Open the handle and bind a generic TimeSeries as self.series (no electrodes table).
        self._open_handle(
            file_path=file_path,
            file=file,
            stream_mode=stream_mode,
            cache=cache,
            stream_cache_path=stream_cache_path,
            storage_options=storage_options,
        )
        if self.reading_method == "use_pynwb":
            self.series = self._retrieve_timeseries_pynwb()
        else:
            backend = "zarr" if self.reading_method == "use_zarr" else "hdf5"
            self.series = self._locate_timeseries(backend)

    def _retrieve_timeseries_pynwb(self):
        from pynwb.base import TimeSeries

        time_series_dict = {}
        for item in self.nwbfile.all_children():
            if isinstance(item, TimeSeries):
                time_series_dict[item.data.name.replace("/data", "")[1:]] = item
        if self.timeseries_path is not None:
            if self.timeseries_path not in time_series_dict:
                raise ValueError(f"TimeSeries {self.timeseries_path} not found in file")
        elif len(time_series_dict) == 1:
            self.timeseries_path = list(time_series_dict.keys())[0]
        else:
            raise ValueError(
                f"Multiple TimeSeries found! Specify 'timeseries_path'. Options: {list(time_series_dict.keys())}"
            )
        return time_series_dict[self.timeseries_path]

    def _locate_timeseries(self, backend):
        if self.timeseries_path is None:
            available = _find_timeseries_from_backend(self.file, backend=backend)
            if len(available) != 1:
                raise ValueError(f"Multiple TimeSeries found! Specify 'timeseries_path'. Options: {available}")
            self.timeseries_path = available[0]
        try:
            return self.file[self.timeseries_path]
        except KeyError:
            available = _find_timeseries_from_backend(self.file, backend=backend)
            raise ValueError(f"{self.timeseries_path} not found! Available options: {available}")

    @staticmethod
    def available_timeseries(file_path, stream_mode=None, storage_options=None):
        """Paths of every TimeSeries in the file, read directly (without pynwb)."""
        backend = _NWBReader._storage_backend(file_path, stream_mode=stream_mode)
        file_handle = read_file_from_backend(
            file_path=file_path, stream_mode=stream_mode, storage_options=storage_options
        )
        return _find_timeseries_from_backend(file_handle, backend=backend)

    def _locate_electrical_series(self, backend):
        # Resolve electrical_series_path (auto-discovering it when the file has exactly one series)
        # and return the series handle, raising a helpful error that lists the options on failure.
        if self.electrical_series_path is None:
            available = _find_neurodata_type_from_backend(self.file, neurodata_type="ElectricalSeries", backend=backend)
            if len(available) != 1:
                raise ValueError(
                    "Multiple ElectricalSeries found in the file. "
                    "Please specify the 'electrical_series_path' argument:"
                    f"Available options are: {available}."
                )
            self.electrical_series_path = available[0]
        try:
            return self.file[self.electrical_series_path]
        except KeyError:
            available = _find_neurodata_type_from_backend(self.file, neurodata_type="ElectricalSeries", backend=backend)
            raise ValueError(
                f"{self.electrical_series_path} not found in the NWB file!" f"Available options are: {available}."
            )

    # --- electrodes table ----------------------------------------------------------------------
    @property
    def column_names(self):
        if self.reading_method == "use_pynwb":
            return list(self.electrodes_table.colnames)
        elif self.reading_method == "use_hdf5":
            return list(self.electrodes_table.attrs["colnames"])
        elif self.reading_method == "use_zarr":
            return list(self.electrodes_table.attrs["colnames"])

    def read_electrode_property(self, name):
        # An electrode property is one electrodes-table column, reordered to this series' channels.
        # Ragged electrode columns cannot be per-channel scalars, so they are skipped (None returned)
        # without their data being read.
        return self._read_table_property(self.electrodes_table, name, self.column_names, self.electrodes_indices)

    def _read_ids(self):
        if self.reading_method == "use_pynwb":
            ids = self.electrodes_table.id
        elif self.reading_method == "use_hdf5":
            ids = self.electrodes_table["id"]
        elif self.reading_method == "use_zarr":
            ids = self.electrodes_table["id"]
        return np.asarray(ids[:])[self.electrodes_indices]

    def channel_ids(self):
        if "channel_name" in self.column_names:
            return list(self.read_electrode_property("channel_name"))
        return list(self._read_ids())

    # --- electrical series ---------------------------------------------------------------------
    def data(self):
        if self.reading_method == "use_pynwb":
            return self.series.data
        elif self.reading_method == "use_hdf5":
            return self.series["data"]
        elif self.reading_method == "use_zarr":
            return self.series["data"]

    def dtype(self):
        return self.data().dtype

    def time_info(self, samples_for_rate_estimation):
        # Return the NWB time facts; the extractor decides how to turn them into times_kwargs.
        if self.reading_method == "use_pynwb":
            series = self.series
            sampling_frequency = series.rate if hasattr(series, "rate") else None
            t_start = series.starting_time if hasattr(series, "starting_time") else None
            timestamps = series.timestamps if getattr(series, "timestamps", None) is not None else None
            if timestamps is not None:
                t_start = timestamps[0]
            if sampling_frequency is None:
                sampling_frequency = 1.0 / np.median(np.diff(timestamps[:samples_for_rate_estimation]))
            return sampling_frequency, t_start, timestamps
        elif self.reading_method == "use_hdf5":
            series = self.series
            if "starting_time" in series.keys():
                t_start = series["starting_time"][()]
                sampling_frequency = series["starting_time"].attrs["rate"]
                timestamps = None
            elif "timestamps" in series.keys():
                timestamps = series["timestamps"]
                t_start = timestamps[0]
                sampling_frequency = 1.0 / np.median(np.diff(timestamps[:samples_for_rate_estimation]))
            else:
                raise ValueError("TimeSeries must have either starting_time or timestamps")
            return sampling_frequency, t_start, timestamps
        elif self.reading_method == "use_zarr":
            series = self.series
            if "starting_time" in series.keys():
                t_start = series["starting_time"][()]
                sampling_frequency = series["starting_time"].attrs["rate"]
                timestamps = None
            elif "timestamps" in series.keys():
                timestamps = series["timestamps"]
                t_start = timestamps[0]
                sampling_frequency = 1.0 / np.median(np.diff(timestamps[:samples_for_rate_estimation]))
            else:
                raise ValueError("TimeSeries must have either starting_time or timestamps")
            return sampling_frequency, t_start, timestamps

    def _conversion(self):
        if self.reading_method == "use_pynwb":
            return self.series.conversion
        elif self.reading_method == "use_hdf5":
            return self.series["data"].attrs["conversion"]
        elif self.reading_method == "use_zarr":
            return self.series["data"].attrs["conversion"]

    def _channel_conversion(self):
        if self.reading_method == "use_pynwb":
            channel_conversion = self.series.channel_conversion
            return channel_conversion[:] if channel_conversion is not None else None
        elif self.reading_method == "use_hdf5":
            if self.series.get("channel_conversion", None) is not None:
                return self.series["channel_conversion"][:]
            return None
        elif self.reading_method == "use_zarr":
            if self.series.get("channel_conversion", None) is not None:
                return self.series["channel_conversion"][:]
            return None

    def _series_offset(self):
        if self.reading_method == "use_pynwb":
            return self.series.offset if hasattr(self.series, "offset") else 0
        elif self.reading_method == "use_hdf5":
            data_attributes = self.series["data"].attrs
            return data_attributes["offset"] if "offset" in data_attributes else 0
        elif self.reading_method == "use_zarr":
            data_attributes = self.series["data"].attrs
            return data_attributes["offset"] if "offset" in data_attributes else 0

    # --- recording ingredients (still volts / NWB names; the extractor applies uV + SI naming) ----
    def gain_to_volts(self):
        gain = self._conversion()
        channel_conversion = self._channel_conversion()
        if channel_conversion is not None:
            gain = gain * channel_conversion
        return gain

    def offset_to_volts(self):
        # NWB stores the offset on the series, or (fallback) per channel in the electrodes table.
        offset = self._series_offset()
        if offset == 0 and "offset" in self.column_names:
            offset = self.read_electrode_property("offset")
        return offset

    def locations(self):
        if not ("rel_x" in self.column_names and "rel_y" in self.column_names):
            return None
        ndim = 3 if "rel_z" in self.column_names else 2
        locations = np.zeros((len(self.electrodes_indices), ndim), dtype=float)
        locations[:, 0] = self.read_electrode_property("rel_x")
        locations[:, 1] = self.read_electrode_property("rel_y")
        if "rel_z" in self.column_names:
            locations[:, 2] = self.read_electrode_property("rel_z")
        return locations

    def groups(self):
        if "group_name" not in self.column_names:
            return None
        return self.read_electrode_property("group_name")


class NwbRecordingExtractor(BaseRecording):
    """Load an NWBFile as a RecordingExtractor.

    Parameters
    ----------
    file_path : str, Path, or None
        Path to the NWB file or an s3 URL. Use this parameter to specify the file location
        if not using the `file` parameter.
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

        if file_path is not None and file is not None:
            raise ValueError("Provide either file_path or file, not both")
        if file_path is None and file is None:
            raise ValueError("Provide either file_path or file")

        self.file_path = file_path
        self.stream_mode = stream_mode
        self.stream_cache_path = stream_cache_path
        self.storage_options = storage_options
        self.electrical_series_path = electrical_series_path

        # extract info
        if use_pynwb and not HAVE_PYNWB:
            raise ImportError(self.installation_mesg)

        if use_pynwb:
            reading_method = "use_pynwb"  # the reader detects hdf5 vs zarr itself for pynwb
        else:
            reading_method = f"use_{_NWBReader._storage_backend(file_path, file, self.stream_mode)}"

        self._reader = _NWBReader(
            reading_method=reading_method,
            electrical_series_path=self.electrical_series_path,
        )
        self._reader.load_recording(
            file_path=self.file_path,
            file=file,
            stream_mode=self.stream_mode,
            cache=cache,
            stream_cache_path=self.stream_cache_path,
            storage_options=self.storage_options,
        )
        self.electrical_series_path = self._reader.electrical_series_path

        channel_ids = self._reader.channel_ids()
        sampling_frequency, t_start, timestamps = self._reader.time_info(samples_for_rate_estimation)
        if load_time_vector and timestamps is not None:
            times_kwargs = dict(time_vector=timestamps)
        else:
            times_kwargs = dict(sampling_frequency=sampling_frequency, t_start=t_start)
        segment_data = self._reader.data()
        dtype = self._reader.dtype()

        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        recording_segment = NwbRecordingSegment(
            electrical_series_data=segment_data,
            times_kwargs=times_kwargs,
        )
        self.add_recording_segment(recording_segment)

        # fetch and add main recording properties (gains/offsets are cast to uV here)
        gains_to_uV = self._reader.gain_to_volts() * 1e6
        offsets_to_uV = self._reader.offset_to_volts() * 1e6
        self.set_channel_gains(gains_to_uV)
        self.set_channel_offsets(offsets_to_uV)
        locations = self._reader.locations()
        if locations is not None:
            self.set_channel_locations(locations)
        groups = self._reader.groups()
        if groups is not None:
            self.set_channel_groups(groups)

        # Every other electrodes-table column becomes a generic channel property. The columns below
        # are skipped because they were already mapped to core recording fields above (channel ids,
        # locations, groups, offsets); "location" is exposed but renamed to SpikeInterface's brain_area.
        if load_channel_properties:
            columns_mapped_to_core_fields = [
                "id",
                "rel_x",
                "rel_y",
                "rel_z",
                "group",
                "group_name",
                "channel_name",
                "offset",
            ]
            for column in self._reader.column_names:
                if column in columns_mapped_to_core_fields:
                    continue
                values = self._reader.read_electrode_property(column)
                if values is None:  # ragged electrode column, not a per-channel property
                    continue
                property_name = "brain_area" if column == "location" else column
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
        if self._reader.reading_method == "use_pynwb":
            self.extra_requirements.append("pynwb")
        elif self._reader.reading_method == "use_hdf5":
            self.extra_requirements.append("h5py")
        elif self._reader.reading_method == "use_zarr":
            self.extra_requirements.append("zarr")
        if self.stream_mode == "fsspec":
            self.extra_requirements.append("fsspec")
        if self.stream_mode == "remfile":
            self.extra_requirements.append("remfile")

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

        return _NWBReader.available_electrical_series(
            file_path, stream_mode=stream_mode, storage_options=storage_options
        )


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


class NwbSortingExtractor(BaseSorting):
    """Load an NWBFile as a SortingExtractor.

    Parameters
    ----------
    file_path : str or Path
        Path to NWB file.
    electrical_series_path : str or None, default: None
        Path of the ElectricalSeries to read the time base (`sampling_frequency` and `t_start`) from.
        When given, it is the sole source of the time base and cannot be combined with an explicit
        `sampling_frequency` or `t_start` (passing both raises). When omitted, the time base must be
        provided directly (see `t_start` / `sampling_frequency`); no ElectricalSeries is auto-selected.
        See Notes.
    sampling_frequency : float or None, default: None
        The sampling frequency in Hz. If None, it is read from the named ElectricalSeries, or (when no
        ElectricalSeries is named) from the Units table ``resolution`` attribute. See Notes.
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
        Time (in seconds, on the NWB session clock) of the recording's first sample. NWB stores spikes
        as times; frames are computed as ``frames = (times - t_start) * sampling_frequency``. The first
        frame is always the start of the recording, independent of `t_start`. If None: it is read from
        the named ElectricalSeries; else, when the file contains an ElectricalSeries that was not named,
        it must be provided (an error is raised otherwise); else, when the file has no ElectricalSeries,
        it defaults to 0. See Notes.
    cache : bool, default: False
        If True, the file is cached in the file passed to stream_cache_path
        if False, the file is not cached.
    storage_options : dict | None, default: None
        These are the additional kwargs (e.g. AWS credentials) that are passed to the zarr.open convenience function.
        This is only used on the "zarr" stream_mode.
    use_pynwb : bool, default: False
        Uses the pynwb library to read the NWB file. Setting this to False, the default, uses h5py
        to read the file. Using h5py can improve performance by bypassing some of the PyNWB validations.

    Returns
    -------
    sorting : NwbSortingExtractor
        The sorting extractor for the NWB file.

    Notes
    -----
    A `Units` table stores spikes as times in seconds, so the sorting needs a time base
    (`sampling_frequency` and `t_start`) to convert them to samples. There are two mutually exclusive
    ways to supply it: name an ElectricalSeries with `electrical_series_path` (which yields both values
    and cannot be combined with an explicit `sampling_frequency` or `t_start`), or provide the time base
    directly. No ElectricalSeries is consulted unless it is named. The time base is resolved as follows::

        You provide                                   sampling_frequency            t_start
        --------------------------------------------  ----------------------------  --------------------------
        electrical_series_path (only)                 from that ElectricalSeries    from that ElectricalSeries
        electrical_series_path AND sampling_frequency raises (mutually exclusive)   raises (mutually exclusive)
          or t_start
        t_start (no electrical_series_path)           argument or Units.resolution  argument
                                                      (raises if neither)
        no t_start, file HAS an ElectricalSeries      n/a                           raises (name it or pass
          (and no electrical_series_path)                                             t_start)
        no t_start, file has NO ElectricalSeries      argument or Units.resolution  defaults to 0
                                                      (raises if neither)

    Justification:

    - No ElectricalSeries is auto-selected: the one present in a file is not necessarily the one the
      sorting was computed from (for example an LFP series next to a full-band sorting), so it is used
      only when named.
    - `t_start` only anchors the frame grid; the spike times in seconds are preserved regardless. When
      the file has no recording to align to, it defaults to 0. When a recording is present but not
      named, defaulting would silently mis-anchor spikes against a real recording, so it must be given.
    - `sampling_frequency` may be omitted when the Units table has a `resolution` attribute (read as
      1 / resolution, which is spike-native). It has no other safe default (a wrong rate is a scale
      error that corrupts every frame), so it is required otherwise.

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

        self.stream_mode = stream_mode
        self.stream_cache_path = stream_cache_path
        self.electrical_series_path = electrical_series_path
        self.file_path = file_path
        self.t_start = t_start
        self.provided_sampling_frequency = sampling_frequency
        self.storage_options = storage_options

        if use_pynwb and not HAVE_PYNWB:
            raise ImportError(self.installation_mesg)

        if use_pynwb:
            reading_method = "use_pynwb"
        else:
            reading_method = f"use_{_NWBReader._storage_backend(file_path, stream_mode=stream_mode)}"

        self._reader = _NWBReader(
            reading_method=reading_method,
            electrical_series_path=electrical_series_path,
            unit_table_path=unit_table_path,
        )
        self._reader.load_units(
            file_path=file_path,
            stream_mode=stream_mode,
            cache=cache,
            stream_cache_path=stream_cache_path,
            storage_options=storage_options,
        )
        self.units_table = self._reader.units_table

        # Resolve the time base (sampling_frequency, t_start). See the class docstring for the full
        # contract. There are two ways to supply it: name an ElectricalSeries via electrical_series_path
        # (which yields both values), or provide it directly. No ElectricalSeries is ever consulted
        # unless it is named, because the ElectricalSeries in a file is not necessarily the one the
        # sorting was computed from.

        electrical_series_path_is_provided = electrical_series_path is not None
        sampling_frequency_is_provided = self.provided_sampling_frequency is not None
        t_start_is_provided = self.t_start is not None
        sampling_frequency = self.provided_sampling_frequency
        # Per-sample timestamps of an irregular clock; used to map spikes to samples via searchsorted.
        time_vector = None

        if electrical_series_path_is_provided:
            # The named ElectricalSeries is the sole source of the time base; it is not combined with an
            # explicit sampling_frequency or t_start.
            if sampling_frequency_is_provided or t_start_is_provided:
                raise ValueError(
                    "Provide either 'electrical_series_path' or the time base ('sampling_frequency' and "
                    "'t_start'), not both."
                )
            # A timestamps-based ElectricalSeries has an irregular clock: keep its timestamps (as a lazy
            # handle) so spikes map to samples exactly (searchsorted) rather than via the estimated rate.
            # A rate-based series returns timestamps None and stays on the linear path.
            sampling_frequency, self.t_start, time_vector = self._reader.fetch_time_info_from_electrical_series(
                samples_for_rate_estimation
            )
        else:
            # No ElectricalSeries named. Spike times are stored as seconds from session start, so anchor
            # the frame grid at t_start = 0 by default (frames = times * sampling_frequency). Alignment to a
            # particular recording is an explicit opt-in via `electrical_series_path`; we do not search the
            # file for an unnamed ElectricalSeries to align to, since that walks the whole HDF5 tree (one
            # network round-trip per object, dominating the streamed init) and the detected series would not
            # be used for alignment anyway.
            if not t_start_is_provided:
                self.t_start = 0.0

            # sampling_frequency: the argument, or the Units.resolution attribute.
            if sampling_frequency is None:
                sampling_frequency = self._reader._sampling_frequency_from_units_metadata()
            if sampling_frequency is None:
                raise ValueError(
                    "Couldn't determine the sampling frequency. Provide it with the 'sampling_frequency' "
                    "argument, set the Units table 'resolution' attribute, or pass 'electrical_series_path'."
                )

        unit_ids = self._reader.unit_ids()
        spike_times_data = self._reader.spike_times()
        spike_times_index_data = self._reader.spike_times_index()

        BaseSorting.__init__(self, sampling_frequency=sampling_frequency, unit_ids=unit_ids)

        sorting_segment = NwbSortingSegment(
            spike_times_data=spike_times_data,
            spike_times_index_data=spike_times_index_data,
            sampling_frequency=self.sampling_frequency,
            t_start=self.t_start,
            time_vector=time_vector,
        )
        self.add_sorting_segment(sorting_segment)

        # fetch and add sorting properties
        if load_unit_properties:
            for property_name, property_values in self._reader._fetch_unit_properties().items():
                self.set_property(property_name, property_values)

        if reading_method == "use_pynwb":
            self.extra_requirements.append("pynwb")
        elif reading_method == "use_hdf5":
            self.extra_requirements.append("h5py")
        elif reading_method == "use_zarr":
            self.extra_requirements.append("zarr")
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

    def _compute_and_cache_spike_vector(self) -> None:
        # Performance: this deliberately overrides the generic BaseSorting builder, and the override is
        # the point. The generic builder reads one spike train per unit (one streamed read per unit),
        # whereas an NWB Units table stores every unit's spikes in a single flat times dataset grouped by
        # unit. Reading that dataset once (see NwbSortingSegment._get_all_spike_samples_and_unit_indices)
        # turns num_units streamed reads into one, which is a large difference for remote/streamed files.
        # Do not drop this override in favor of the generic path without a reason: doing so silently
        # reintroduces the per-unit read pattern. An NWB Units table is always single-segment, so the
        # single segment is built directly.
        segment = self.segments[0]
        sample_indices, unit_indices = segment._get_all_spike_samples_and_unit_indices()

        # NWB stores spike_times grouped by unit, not globally ordered by time: pynwb itself notes that
        # "the earliest spike may be in any unit" (Units.get_earliest_spike_time), and within-unit ordering
        # is only a best-practice (checked by nwbinspector), not a schema guarantee. So a global sort is
        # required to build the time-ordered spike vector. It is stable so unit_index order is preserved on
        # equal sample_index ties, matching the generic BaseSorting builder; on already-ordered input the
        # sort is a no-op (argsort returns the identity permutation).
        order = np.argsort(sample_indices, stable=True)
        num_spikes = sample_indices.size

        spikes = np.zeros(num_spikes, dtype=minimum_spike_dtype)
        spikes["sample_index"] = sample_indices[order]
        spikes["unit_index"] = unit_indices[order]
        # segment_index stays 0 for the single segment.
        self._cached_spike_vector = spikes
        self._cached_spike_vector_segment_slices = np.array([[0, num_spikes]], dtype="int64")

    @staticmethod
    def fetch_available_units_tables(
        file_path: str | Path,
        stream_mode: Optional[Literal["fsspec", "remfile", "zarr"]] = None,
        storage_options: dict | None = None,
    ) -> list[str]:
        """
        Retrieves the paths to all Units tables within an NWB (Neurodata Without Borders) file.

        Parameters
        ----------
        file_path : str or Path
            The path to the NWB (Neurodata Without Borders) file.
        stream_mode : "fsspec" | "remfile" | "zarr" | None, optional
            Determines the streaming mode for reading the file.
        storage_options : dict | None, default: None
            Additional kwargs (e.g. AWS credentials) passed to zarr.open. Only used with "zarr" stream_mode.

        Returns
        -------
        list of str
            Paths to all Units tables found in the file.
        """
        return _NWBReader.available_units_tables(file_path, stream_mode=stream_mode, storage_options=storage_options)


class NwbSortingSegment(BaseSortingSegment):
    def __init__(
        self, spike_times_data, spike_times_index_data, sampling_frequency: float, t_start: float, time_vector=None
    ):
        BaseSortingSegment.__init__(self, t_start=t_start)
        self.spike_times_data = spike_times_data
        self.spike_times_index_data = spike_times_index_data
        self._sampling_frequency = sampling_frequency
        # Per-sample timestamps for an irregular clock, else None (uniform clock -> linear mapping).
        self._time_vector = time_vector

    def _frame_to_time(self, frame):
        if self._time_vector is not None:
            return self._time_vector[frame]
        return frame / self._sampling_frequency + self._t_start

    def _times_to_samples(self, spike_times) -> np.ndarray:
        # Map seconds to samples. An irregular clock (per-sample timestamps) maps exactly with a
        # searchsorted into the time vector; a uniform clock uses sample = (t - t_start) * sampling_frequency.
        if self._time_vector is not None:
            # searchsorted needs the timestamps in memory: materialize them once, on first use, and cache.
            if not isinstance(self._time_vector, np.ndarray):
                self._time_vector = np.asarray(self._time_vector)
            samples = np.searchsorted(self._time_vector, spike_times, side="right") - 1
        else:
            samples = np.round((spike_times - self._t_start) * self._sampling_frequency)
        return samples.astype("int64", copy=False)

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        # Convert frame boundaries to time boundaries (via the time vector when the clock is irregular)
        start_time = None if start_frame is None else self._frame_to_time(start_frame)
        end_time = None if end_frame is None else self._frame_to_time(end_frame)

        # Get spike times in seconds
        spike_times = self.get_unit_spike_train_in_seconds(unit_id=unit_id, start_time=start_time, end_time=end_time)

        return self._times_to_samples(spike_times)

    def _get_all_spike_samples_and_unit_indices(self):
        """Read all units' spike trains (as samples) in one bulk read, plus each spike's unit index.

        NWB stores all spike times in a single flat dataset grouped by unit (with per-unit
        boundaries in ``spike_times_index_data``), so the whole segment is read at once instead
        of one streamed slice per unit.
        """
        spike_times = np.asarray(self.spike_times_data[:], dtype="float64")
        # NWB stores a ragged column as one flat data array (spike_times, all units concatenated) plus an
        # index array of cumulative end offsets (spike_times_index): spike_times_index[u] is where unit u's
        # spikes end and unit u+1's begin, so unit u is spike_times[spike_times_index[u-1] : spike_times_index[u]].
        # Prepending a 0 gives every unit an explicit start, so boundaries[u]:boundaries[u+1] is unit u's
        # slice; the gap between consecutive boundaries (np.diff) is that unit's spike count, and repeating
        # each unit index by its count labels every spike with its unit.
        # Example: 3 units with 2, 0, 3 spikes -> spike_times_index = [2, 2, 5],
        #   boundaries = [0, 2, 2, 5], counts = [2, 0, 3], unit_indices = [0, 0, 2, 2, 2].
        boundaries = np.concatenate(([0], np.asarray(self.spike_times_index_data[:], dtype="int64")))
        counts = np.diff(boundaries)
        unit_indices = np.repeat(np.arange(counts.size, dtype="int64"), counts)
        sample_indices = self._times_to_samples(spike_times)
        return sample_indices, unit_indices

    def get_last_spike_time(self, segment_index: int | None = None) -> float:
        """Get the time of the last spike in a segment across all units.
        Overridden from BaseSorting to use spike_times_data.

        Parameters
        ----------
        segment_index : int or None, default: None
            The segment index (required for multi-segment)

        Returns
        -------
        float
            The time of the last spike in seconds, or 0.0 if no spikes exist.
        """
        segment = self.segments[segment_index] if segment_index is not None else self.segments[0]
        if segment.spike_times_data.size == 0:
            return 0.0
        return segment.spike_times_data[-1]

    def get_unit_spike_train_in_seconds(
        self,
        unit_id,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        """Get the spike train times for a unit in seconds.

        This method returns spike times directly in seconds without conversion
        to samples, avoiding double conversion for NWB files that already store
        spike times as timestamps.

        Parameters
        ----------
        unit_id
            The unit id to retrieve spike train for
        start_time : float, default: None
            The start time in seconds for spike train extraction
        end_time : float, default: None
            The end time in seconds for spike train extraction

        Returns
        -------
        spike_times : np.ndarray
            Spike times in seconds
        """
        # Extract the spike times for the unit
        unit_index = self.parent_extractor.id_to_index(unit_id)
        if unit_index == 0:
            start_index = 0
        else:
            start_index = self.spike_times_index_data[unit_index - 1]
        end_index = self.spike_times_index_data[unit_index]
        spike_times = self.spike_times_data[start_index:end_index]

        # Filter by time range if specified
        start_index = 0
        if start_time is not None:
            start_index = np.searchsorted(spike_times, start_time, side="left")

        if end_time is not None:
            end_index = np.searchsorted(spike_times, end_time, side="left")
        else:
            end_index = spike_times.size

        return spike_times[start_index:end_index].astype("float64", copy=False)


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


class NwbTimeSeriesExtractor(BaseRecording):
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

        if use_pynwb and not HAVE_PYNWB:
            raise ImportError(self.installation_mesg)

        if use_pynwb:
            reading_method = "use_pynwb"
        else:
            reading_method = f"use_{_NWBReader._storage_backend(file_path, file, stream_mode)}"

        self._reader = _NWBReader(reading_method=reading_method, timeseries_path=timeseries_path)
        self._reader.load_timeseries(
            file_path=file_path,
            file=file,
            stream_mode=stream_mode,
            cache=cache,
            stream_cache_path=stream_cache_path,
            storage_options=storage_options,
        )
        self.timeseries_path = self._reader.timeseries_path

        sampling_frequency, t_start, timestamps = self._reader.time_info(samples_for_rate_estimation)
        if load_time_vector and timestamps is not None:
            times_kwargs = dict(time_vector=timestamps)
        else:
            times_kwargs = dict(sampling_frequency=sampling_frequency, t_start=t_start)
        segment_data = self._reader.data()
        num_channels = 1 if segment_data.ndim == 1 else segment_data.shape[1]
        channel_ids = np.arange(num_channels)
        dtype = self._reader.dtype()

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

        if reading_method == "use_pynwb":
            self.extra_requirements.append("pynwb")
        elif reading_method == "use_hdf5":
            self.extra_requirements.append("h5py")
        elif reading_method == "use_zarr":
            self.extra_requirements.append("zarr")

        if self.stream_mode == "fsspec":
            self.extra_requirements.append("fsspec")
        elif self.stream_mode == "remfile":
            self.extra_requirements.append("remfile")

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
        return _NWBReader.available_timeseries(file_path, stream_mode=stream_mode, storage_options=storage_options)


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


# Where each SortingAnalyzer extension's data lives in an NWB file. Each extension maps to an ordered
# list of read-sources tried until one resolves: the typed ndx-spikesorting container first, then the
# generic-NWB convention. A user/dataset override merges over this (see `extension_map` in
# `read_nwb_sorting_analyzer`); `None` disables an extension. This is only for real analyzer extensions;
# sparsity, sorting properties, and recording metadata are handled separately by the reader.
#
# The "typed_container" source is a hook: the typed reader currently lives in ndx-spikesorting, so for
# now it never resolves here and the convention source is used. When that logic moves into
# spikeinterface, this source will resolve against the ndx-spikesorting types.
DEFAULT_EXTENSION_MAP = {
    "templates": [
        {"source": "typed_container", "type": "Templates"},
        {"source": "column", "column": "waveform_mean", "std_column": "waveform_sd"},
    ],
    "quality_metrics": [
        {"source": "typed_container", "type": "UnitsMetrics"},
        {"source": "columns", "columns": "canonical_quality"},
    ],
    "template_metrics": [
        {"source": "typed_container", "type": "UnitsMetrics"},
    ],
    "unit_locations": [
        {"source": "typed_container", "type": "UnitLocations"},
    ],
}


def _resolve_extension_columns(extension_map, scalar_colnames, all_colnames):
    """Resolve the convention (column-based) sources of an extension map against a Units table's columns.

    Returns a dict with the resolved template column (and optional std column) and the per-extension
    column lists for the column-backed extensions. The "typed_container" source is skipped here (the
    typed reader lives in ndx-spikesorting for now), so resolution falls through to the "column"/
    "columns" sources. `scalar_colnames` are the 1-D per-unit columns eligible to be metrics/properties;
    `all_colnames` is every column name (used to check a source's target exists).
    """
    from spikeinterface.metrics.quality import ComputeQualityMetrics

    resolved = {"templates_column": None, "templates_std_column": None, "quality_metric_colnames": []}

    for extension, sources in extension_map.items():
        if sources is None:  # explicitly disabled
            continue
        for source in sources:
            kind = source.get("source")
            if kind == "typed_container":
                continue  # hook: not resolvable here yet
            if kind == "column":
                column = source.get("column")
                if column in all_colnames:
                    if extension == "templates":
                        resolved["templates_column"] = column
                        std_column = source.get("std_column")
                        resolved["templates_std_column"] = std_column if std_column in all_colnames else None
                    break
            if kind == "columns":
                spec = source.get("columns")
                if spec == "canonical_quality":
                    canonical = set(ComputeQualityMetrics.get_metric_columns())
                    cols = [c for c in scalar_colnames if c in canonical]
                else:  # an explicit list of column names
                    cols = [c for c in spec if c in scalar_colnames]
                if extension == "quality_metrics":
                    resolved["quality_metric_colnames"] = cols
                break

    # every scalar column not claimed as a quality metric becomes a sorting property
    claimed = set(resolved["quality_metric_colnames"])
    resolved["property_colnames"] = [c for c in scalar_colnames if c not in claimed]
    return resolved


def read_nwb_sorting_analyzer(
    file_path: str | Path,
    t_start: float | None = None,
    sampling_frequency: float | None = None,
    electrical_series_path: str | None = None,
    unit_table_path: str | None = None,
    stream_mode: Literal["fsspec", "remfile", "zarr"] | None = None,
    stream_cache_path: str | Path | None = None,
    cache: bool = False,
    storage_options: dict | None = None,
    use_pynwb: bool = False,
    group_name: str | None = None,
    compute_extra: List[str] | None = ["unit_locations"],
    compute_extra_params: dict | None = None,
    extension_map: dict | None = None,
    rescale_templates_to_uV: bool = True,
    verbose: bool = False,
) -> SortingAnalyzer:
    # extension_map overrides (per extension) merge over DEFAULT_EXTENSION_MAP; see its docstring.
    resolved_extension_map = dict(DEFAULT_EXTENSION_MAP)
    if extension_map is not None:
        resolved_extension_map.update(extension_map)
    # try to read recording object to get the analyzer
    try:
        recording = NwbRecordingExtractor(
            file_path=file_path,
            electrical_series_path=electrical_series_path,
            stream_mode=stream_mode,
            stream_cache_path=stream_cache_path,
            cache=cache,
            storage_options=storage_options,
            use_pynwb=use_pynwb,
        )
    except Exception:
        if verbose:
            print("Could not load recording, proceeding without it")
        recording = None

    t_start_tmp = 0 if t_start is None else t_start

    sorting_tmp = NwbSortingExtractor(
        file_path=file_path,
        electrical_series_path=electrical_series_path,
        unit_table_path=unit_table_path,
        stream_mode=stream_mode,
        stream_cache_path=stream_cache_path,
        cache=cache,
        storage_options=storage_options,
        use_pynwb=use_pynwb,
        t_start=t_start_tmp,
        sampling_frequency=sampling_frequency,
        load_unit_properties=False,  # columns are read deliberately below, into their extensions
    )

    sorting = sorting_tmp
    # Recordingless case: leave t_start at 0 (set when the sorting was constructed). NWB spike times are
    # in seconds from session start, so they are always >= 0 and map to non-negative frames with
    # t_start = 0. Anchoring t_start to the first spike would only remove empty space before it on the
    # timeline (cosmetic) and would cost a streamed read, so we skip it to keep the build cheap.

    # Read the Units table deliberately. Classify each column and only materialize the ones that
    # become part of the analyzer: `waveform_mean` (templates), the `electrodes` region (sparsity), the
    # per-unit numeric metric columns (quality/template metrics), and the per-unit label columns
    # (identity properties). The large per-spike columns (spike amplitudes, spike depths, each as long
    # as the whole spike vector) are never read here, which is what makes streaming this table viable.
    if use_pynwb:
        units_table = sorting.units_table
        colnames = list(units_table.colnames)
        units = units_table.to_dataframe(index=True)
        structural = {"waveform_mean", "waveform_sd", "electrodes"}
        metric_colnames = [c for c in units.columns if c not in structural and units[c].dtype.kind in "fiu"]
        label_colnames = [
            c
            for c in units.columns
            if c not in structural
            and units[c].dtype.kind == "O"
            and not isinstance(units[c].iloc[0], (list, np.ndarray))
        ]
    else:
        units_group = sorting.units_table
        colnames = list(units_group.keys())
        structural = ("id", "spike_times", "waveform_mean", "waveform_sd", "electrodes")
        metric_colnames, label_colnames = [], []
        for column_name in colnames:
            if column_name.endswith("_index") or column_name in structural or f"{column_name}_index" in colnames:
                continue  # index columns, structural columns, and ragged/per-spike columns
            dataset = units_group[column_name]
            if dataset.attrs.get("neurodata_type") == "DynamicTableRegion":
                continue  # e.g. max_electrode: a per-unit reference into electrodes, not a metric
            if dataset.ndim == 1 and dataset.dtype.kind in "fiu":
                metric_colnames.append(column_name)
            elif dataset.ndim == 1 and dataset.dtype.kind in "OSU":
                label_colnames.append(column_name)

    # Resolve the extension map against the Units columns: which column holds the templates, and which
    # scalar columns are quality metrics (canonical names by default) vs plain sorting properties.
    scalar_colnames = metric_colnames + label_colnames
    resolved = _resolve_extension_columns(resolved_extension_map, scalar_colnames, colnames)
    templates_column = resolved["templates_column"]
    templates_std_column = resolved["templates_std_column"]
    quality_metric_colnames = resolved["quality_metric_colnames"]
    property_colnames = resolved["property_colnames"]

    if not use_pynwb:
        # deliberate read: only the resolved template column(s), the electrodes region, and the scalar
        # columns that become metrics or properties are materialized; ragged/per-spike columns are not.
        needed = [c for c in (templates_column, templates_std_column, "electrodes") if c and c in colnames]
        needed += scalar_colnames
        units = _create_df_from_nwb_table(units_group, columns=needed)

    electrodes_indices = None
    if use_pynwb:
        electrodes_table = sorting._reader.nwbfile.electrodes.to_dataframe(index=True)
        if "electrodes" in colnames:
            electrodes_indices = units["electrodes"]
    else:
        electrodes_table = _create_df_from_nwb_table(sorting._reader.file["/general/extracellular_ephys/electrodes"])
        if "electrodes" in colnames:
            electrodes_indices = electrodes_indices = units["electrodes"][:]

    if electrodes_indices is not None:
        # here we assume all groups are the same for each unit, so we just check one.
        if "group_name" in electrodes_table.columns:
            group_names = np.array([electrodes_table.iloc[int(ei[0])]["group_name"] for ei in electrodes_indices])
            if len(np.unique(group_names)) > 1:
                if group_name is None:
                    raise Exception(
                        f"More than one group, use group_name option to select units. Available groups: {np.unique(group_names)}"
                    )
                else:
                    unit_mask = group_names == group_name
                    if verbose:
                        print(f"Selecting {sum(unit_mask)} / {len(units)} units from {group_name}")
                    sorting = sorting.select_units(unit_ids=sorting.unit_ids[unit_mask])
                    units = units.loc[units.index[unit_mask]]
                    electrodes_indices = units["electrodes"]

    # Every scalar column not claimed as a quality metric becomes a sorting property (labels like
    # cluster_uuid, and any non-canonical numeric column). Properties show up in the GUI's unit table
    # alongside metrics, so nothing is lost, without asserting an unknown column is a "quality metric".
    for property_column in property_colnames:
        if property_column in units.columns and len(units[property_column]) == sorting.get_num_units():
            sorting.set_property(property_column, units[property_column].values)

    # Obtain a recording for the analyzer. With a real recording we use it directly. Without one (the
    # streaming curation case) we build a lightweight placeholder recording that carries only the probe
    # geometry, hand it to the standard constructor, and drop it afterwards. This mirrors
    # `read_kilosort_as_analyzer` and avoids hand-assembling `rec_attributes` and channel ids.
    if recording is not None:
        group_names = np.unique(recording.get_channel_groups())
        if group_name is not None and len(group_names) > 1:
            recording = recording.split_by("group")[group_name]
        analyzer_recording = recording
        analyzer_channel_ids = list(recording.get_channel_ids())
    else:
        # The placeholder recording only needs to loosely bound the timeline: it is dropped right after the
        # build and its length feeds the GUI time axis, not any curation quantity. Reading the full
        # spike_times array for the exact last spike would cost ~130 MB over a stream, so instead read only
        # the last stored spike (one chunk, ~4 MB) and double it. spike_times is stored per-unit-
        # concatenated (not globally time-sorted), so the last stored value is a lower bound on the true
        # last spike; doubling generously covers the gap while over-estimating only adds a harmless empty
        # tail. Read from sorting_tmp because a group selection wraps `sorting` in a UnitsSelectionSorting
        # whose segment has no spike_times_data.
        spike_times_data = sorting_tmp._sorting_segments[0].spike_times_data
        last_stored_spike_time = 0.0 if spike_times_data.shape[0] == 0 else float(np.asarray(spike_times_data[-1]))
        placeholder_duration = max(last_stored_spike_time * 2.0, 1.0)
        analyzer_recording, analyzer_channel_ids = _make_placeholder_recording_from_electrodes(
            sorting, electrodes_table, electrodes_indices, duration=placeholder_duration, verbose=verbose
        )

    # Per-unit sparsity and channel map from the Units `electrodes` region. Each unit's `waveform_mean`
    # is stored only on a subset of channels (near its peak); the region gives which channels those are.
    # We use it both for the analyzer sparsity and to scatter the waveforms onto their true channel
    # positions instead of stacking the sparse block densely.
    sparsity, unit_local_channels = _make_sparsity_from_electrodes(
        sorting, electrodes_table, electrodes_indices, analyzer_channel_ids
    )

    # Instantiate the analyzer through the standard constructor. For the recordingless case keep the
    # lazy sorting (copy_sorting=False) so spike_times is read on demand rather than materialized here,
    # and drop the placeholder recording once its geometry has been captured into rec_attributes.
    analyzer = SortingAnalyzer.create_memory(
        sorting=sorting,
        recording=analyzer_recording,
        sparsity=sparsity,
        return_in_uV=True,
        peak_sign="neg",
        peak_mode="extremum",
        rec_attributes=None,
        copy_sorting=recording is not None,
    )
    if recording is None:
        analyzer._recording = None

    num_channels = len(analyzer_channel_ids)

    # templates (from the resolved templates column), and the required random_spikes extension
    if templates_column is not None and templates_column in units and unit_local_channels is not None:
        # random_spikes only needs the total spike count of the (possibly group-selected) sorting. Read it
        # from the NWB spike_times_index (per-unit cumulative counts, ~KB) on the unwrapped sorting_tmp,
        # restricted to the selected units, rather than counting through the sorting object: a group
        # selection wraps `sorting` in a UnitsSelectionSorting whose count materializes the parent's whole
        # spike vector (~130-480 MB). np.isin keeps the units still present after any group selection.
        per_unit_spike_counts = np.diff(sorting_tmp._sorting_segments[0].spike_times_index_data[:], prepend=0)
        selected_units_mask = np.isin(sorting_tmp.unit_ids, sorting.unit_ids)
        total_num_spikes = int(per_unit_spike_counts[selected_units_mask].sum())
        _make_random_spikes(analyzer, total_num_spikes)
        _make_templates(
            analyzer,
            units,
            templates_column,
            templates_std_column,
            unit_local_channels,
            num_channels,
            rescale_to_uV=rescale_templates_to_uV,
        )

    # the resolved quality-metric columns -> quality_metrics extension
    _make_metrics(analyzer, units, quality_metric_colnames, verbose=verbose)

    # compute extra required
    if compute_extra is not None:
        if verbose:
            print(f"Computing extra extensions: {compute_extra}")
        compute_extra_params = {} if compute_extra_params is None else compute_extra_params
        analyzer.compute(compute_extra, **compute_extra_params)

    return analyzer


def _make_placeholder_recording_from_electrodes(sorting, electrodes_table, electrodes_indices, duration, verbose=False):
    """Build a lightweight placeholder recording that carries only the probe geometry of the electrodes
    a unit's waveforms live on, for the recordingless (streaming) case.

    Mirrors `read_kilosort_as_analyzer`: `generate_ground_truth_recording` produces a fully lazy
    recording (noise generated on the fly, no traces materialized) whose only purpose is to feed probe
    geometry, channel ids, and a bounding length into the standard analyzer constructor. The caller drops
    the recording (`analyzer._recording = None`) right after construction. Returns the recording and the
    ordered channel ids.
    """
    from probeinterface import Probe
    from spikeinterface.core import generate_ground_truth_recording

    # union of electrode rows referenced by any unit, in sorted order
    electrode_indices_all = []
    for region in electrodes_indices:
        electrode_indices_all.extend(region)
    electrode_indices_all = np.sort(np.unique(electrode_indices_all))
    if verbose:
        print(f"Found {len(electrode_indices_all)} electrodes")
    electrodes_table_sliced = electrodes_table.iloc[electrode_indices_all]

    if "channel_name" in electrodes_table_sliced:
        # channel_name is already decoded to str at the source (_create_df_from_nwb_table), but pandas
        # holds string columns as object dtype; cast to a numpy unicode array so SpikeInterface accepts
        # the channel ids (object-dtype ids are rejected).
        channel_ids = np.asarray(electrodes_table_sliced["channel_name"][:], dtype=str)
    else:
        channel_ids = electrodes_table_sliced.index.to_numpy()

    electrode_colnames = electrodes_table_sliced.columns
    assert (
        "rel_x" in electrode_colnames and "rel_y" in electrode_colnames
    ), "'rel_x' and 'rel_y' should be columns in the electrodes table"
    locations = np.array([electrodes_table_sliced["rel_x"][:], electrodes_table_sliced["rel_y"][:]]).T

    probe = Probe(si_units="um")
    probe.set_contacts(locations, shapes="circle", shape_params={"radius": 1})
    probe.set_device_channel_indices(np.arange(len(channel_ids)))

    recording, _ = generate_ground_truth_recording(
        durations=[duration],
        sampling_frequency=sorting.sampling_frequency,
        probe=probe,
        num_units=1,
        seed=0,
    )
    recording = recording.rename_channels(channel_ids)
    return recording, list(channel_ids)


def _make_sparsity_from_electrodes(sorting, electrodes_table, electrodes_indices, analyzer_channel_ids):
    """Build the analyzer `ChannelSparsity` and the per-unit local channel lists from the Units
    `electrodes` region, mapping each unit's electrode rows onto positions in `analyzer_channel_ids`.
    Returns (sparsity, unit_local_channels), or (None, None) if there is no electrodes region."""
    if electrodes_indices is None:
        return None, None

    from spikeinterface.core.sparsity import ChannelSparsity

    num_channels = len(analyzer_channel_ids)
    if "channel_name" in electrodes_table.columns:
        electrode_row_to_channel_id = electrodes_table["channel_name"].to_numpy()
    else:
        electrode_row_to_channel_id = electrodes_table.index.to_numpy()
    position_of_channel_id = {channel_id: pos for pos, channel_id in enumerate(analyzer_channel_ids)}

    unit_local_channels = []
    sparsity_mask = np.zeros((sorting.get_num_units(), num_channels), dtype=bool)
    for unit_index, region in enumerate(electrodes_indices):
        positions = [
            position_of_channel_id[electrode_row_to_channel_id[int(electrode_row)]]
            for electrode_row in region
            if electrode_row_to_channel_id[int(electrode_row)] in position_of_channel_id
        ]
        unit_local_channels.append(positions)
        sparsity_mask[unit_index, positions] = True
    sparsity = ChannelSparsity(
        mask=sparsity_mask,
        unit_ids=np.asarray(sorting.unit_ids),
        channel_ids=np.asarray(analyzer_channel_ids),
    )
    return sparsity, unit_local_channels


def _make_random_spikes(analyzer, total_num_spikes):
    """Attach the required `random_spikes` extension, always using the "all" method.

    With no recording this extension is never used to extract waveforms (there are no traces), but it is a
    required dependency of templates, so it must exist and be consistent with the sorting's spike count.
    `method="all"` means the indices are just `arange(total_num_spikes)`; the caller supplies the count
    (read cheaply from `spike_times_index`) so this never materializes the spike vector.
    """
    from spikeinterface.core.analyzer_extension_core import ComputeRandomSpikes

    random_spikes_ext = ComputeRandomSpikes(sorting_analyzer=analyzer)
    random_spikes_ext.set_params(method="all")
    random_spikes_ext.data["random_spikes_indices"] = np.arange(total_num_spikes, dtype="int64")
    random_spikes_ext.run_info["run_completed"] = True
    analyzer.extensions["random_spikes"] = random_spikes_ext


def _infer_template_scale_to_uV(dense_templates):
    """Infer the factor that brings templates into microvolts, SpikeInterface's convention.

    The NWB waveform unit attribute is schema-fixed to "volts" and never reflects the true scale
    (pynwb issue #2162), so it cannot drive the conversion. The template magnitude can: an extracellular
    spike peaks at roughly tens to hundreds of microvolts, so pick the factor in {1, 1e3, 1e6} that lands
    the median per-unit peak in that physiological window. This reads no extra data (it operates on the
    already-built templates). Returns 1.0 if the magnitude is unusable (all zero).
    """
    per_unit_peak = np.abs(dense_templates).max(axis=(1, 2))
    per_unit_peak = per_unit_peak[per_unit_peak > 0]
    if per_unit_peak.size == 0:
        return 1.0
    median_peak = float(np.median(per_unit_peak))
    for factor in (1.0, 1e3, 1e6):
        if 5.0 <= median_peak * factor <= 2000.0:
            return factor
    return 1.0


def _make_templates(
    analyzer,
    units,
    templates_column,
    templates_std_column,
    unit_local_channels,
    num_channels,
    rescale_to_uV=False,
):
    """Attach the `templates` extension from the resolved templates column, scattering each unit's
    sparse waveform block onto its true channel positions."""
    from spikeinterface.core.analyzer_extension_core import ComputeTemplates

    waveform_mean = np.array([np.asarray(t, dtype="float") for t in units[templates_column].values])
    num_samples_template = waveform_mean.shape[1]

    # Scatter each unit's waveform onto its channels. This relies on the alignment contract that
    # waveform_mean[:, i] corresponds to the unit's electrodes-region entry i. The IBL writer guarantees
    # it by ordering each unit's real channels first (padding last) and building the electrodes region
    # from exactly those real channels in the same order, so the first k = len(region) columns are the
    # real channels in region order and any trailing padding columns are correctly dropped by [:, :k].
    dense_templates = np.zeros((analyzer.get_num_units(), num_samples_template, num_channels), dtype="float32")
    for unit_index, positions in enumerate(unit_local_channels):
        k = len(positions)
        dense_templates[unit_index][:, positions] = waveform_mean[unit_index][:, :k]

    # The stored waveform unit attribute is schema-fixed to "volts" and never reflects the real scale
    # (pynwb #2162), so it cannot drive a conversion. When rescaling is requested, infer the factor from
    # the template magnitude instead and apply the same factor to the std templates below so both stay in
    # the same unit. On by default (read_nwb_sorting_analyzer) so templates read in microvolts; it can be
    # turned off to keep the file's native unit, since the scale does not affect peak channel, shape,
    # location, or sparsity.
    template_scale = _infer_template_scale_to_uV(dense_templates) if rescale_to_uV else 1.0
    if template_scale != 1.0:
        dense_templates *= template_scale

    # scalar alignment (nbefore): the sample where the average peak-channel amplitude is largest
    peak_amplitude_per_sample = np.abs(np.nan_to_num(waveform_mean)).max(axis=2).mean(axis=0)
    nbefore = int(np.argmax(peak_amplitude_per_sample))

    templates_ext = ComputeTemplates(sorting_analyzer=analyzer)
    templates_ext.data["average"] = dense_templates
    operators = ["average"]

    # Only expose the "std" operator when the file actually stores a std column (resolved). Fabricating a
    # zero std would falsely imply the unit's spikes have no waveform variability, so when it is absent we
    # declare only "average" rather than inventing data.
    if templates_std_column is not None and templates_std_column in units.columns:
        waveform_sd = np.array([np.asarray(t, dtype="float") for t in units[templates_std_column].values])
        dense_std = np.zeros((analyzer.get_num_units(), num_samples_template, num_channels), dtype="float32")
        for unit_index, positions in enumerate(unit_local_channels):
            k = len(positions)
            dense_std[unit_index][:, positions] = waveform_sd[unit_index][:, :k]
        if template_scale != 1.0:
            dense_std *= template_scale
        templates_ext.data["std"] = dense_std
        operators.append("std")

    templates_ext.set_params(
        ms_before=nbefore / analyzer.sampling_frequency * 1000,
        ms_after=(num_samples_template - nbefore) / analyzer.sampling_frequency * 1000,
        operators=operators,
    )
    templates_ext.run_info["run_completed"] = True
    analyzer.extensions["templates"] = templates_ext


def _make_metrics(analyzer, units, quality_metric_colnames, verbose=False):
    """Attach the `quality_metrics` extension from the resolved quality-metric columns. Other per-unit
    scalar columns are loaded as sorting properties instead (see the caller), and `template_metrics` is
    never synthesized from value columns because it carries structured data a values-only table lacks."""
    import pandas as pd
    from spikeinterface.metrics.quality import ComputeQualityMetrics

    quality_metric_df = pd.DataFrame(index=analyzer.unit_ids)
    for col in quality_metric_colnames:
        if col in units.columns:
            quality_metric_df.loc[:, col] = units[col].values

    if len(quality_metric_df.columns) > 0:
        if verbose:
            print("Adding quality metrics")
        quality_metrics_ext = ComputeQualityMetrics(analyzer)
        quality_metrics_ext.data["metrics"] = quality_metrics_ext._cast_metrics(quality_metric_df)
        quality_metrics_ext.run_info["run_completed"] = True
        analyzer.extensions["quality_metrics"] = quality_metrics_ext


def _create_df_from_nwb_table(group, columns=None):
    """Makes pandas DataFrame from hdf5/zarr NWB group.

    If `columns` is given, only those columns (plus the `id` index) are read; the data of every other
    column is never touched. This lets callers avoid materializing large columns, e.g. the per-spike
    arrays of a Units table (spike amplitudes, spike depths), when only a few columns are needed.
    """
    import pandas as pd

    all_colnames = list(group.keys())
    if columns is None:
        colnames = all_colnames
    else:
        colnames = [c for c in ["id", *columns] if c in all_colnames]
    data = {}
    for col in colnames:
        if "_index" in col:
            continue
        item = group[col][:]
        if f"{col}_index" in all_colnames:
            item = np.split(item, group[f"{col}_index"][:])[:-1]
            data[col] = item
        elif item.ndim > 1:
            data[col] = [item_flat for item_flat in item]
        else:
            if item.dtype.kind in ("O", "S"):
                # HDF5 stores string columns as variable-length or fixed-length bytes (e.g. b"AP0",
                # b"Probe00"). Decode to str at the source so every caller gets plain strings: channel and
                # group ids must be str (SpikeInterface rejects object-dtype byte-string ids), and group
                # selection compares the column against a str argument.
                item = np.array([v.decode("utf-8") if isinstance(v, bytes) else v for v in item])
            data[col] = item
    df = pd.DataFrame(data=data)
    df.set_index("id", inplace=True)
    return df
