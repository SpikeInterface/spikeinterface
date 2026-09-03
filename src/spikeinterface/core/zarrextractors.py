import warnings
from pathlib import Path

import numpy as np
import zarr

from probeinterface import ProbeGroup

from .base import minimum_spike_dtype, _get_class_from_string
from .baserecording import BaseRecording, BaseRecordingSegment
from .basesorting import BaseSorting, SpikeVectorSortingSegment
from .core_tools import define_function_from_class, check_json, retrieve_importing_provenance
from .job_tools import split_job_kwargs
from .core_tools import is_path_remote


def super_zarr_open(folder_path: str | Path, mode: str = "r", storage_options: dict | None = None):
    """
    Open a zarr folder with super powers.

    The function tries to open a zarr folder with the following options:
    - zarr.open_consolidated (if mode is not "a" or "r+")
    - zarr.open

    For remote paths, the function tries to open the folder with:
    - the provided storage options (if storage_options is not None)
    - anon=True/False is storage_options is None

    Parameters
    ----------
    folder_path : str | Path
        The path to the zarr folder
    mode : str, optional
        The mode to open the zarr folder in, default: "r"
    storage_options : dict | None, optional
        The storage options to use when opening the zarr folder, default: None

    Returns
    -------
    root: zarr.hierarchy.Group
        The zarr root group object

    Raises
    ------
    ValueError
        Raised if the folder cannot be opened in the specified mode with the given storage options.
    """
    import zarr

    # if mode is append or read/write, we try to open the folder with zarr.open
    # since zarr.open_consolidated does not support creating new groups/datasets
    if mode in ("a", "r+"):
        open_funcs = (zarr.open,)
    else:
        open_funcs = (zarr.open_consolidated, zarr.open)

    # if storage_options is None, we try to open the folder with and without anonymous access
    # if storage_options is not None, we try to open the folder with the given storage options
    if storage_options is None or storage_options == {}:
        storage_options_to_test = ({"anon": True}, {"anon": False})
    else:
        storage_options_to_test = (storage_options,)

    root = None
    exception = None
    if is_path_remote(folder_path):
        for open_func in open_funcs:
            if root is not None:
                break
            for storage_options in storage_options_to_test:
                try:
                    root = open_func(str(folder_path), mode=mode, storage_options=storage_options)
                    break
                except Exception as e:
                    exception = e
                    pass
    else:
        if not Path(folder_path).is_dir():
            raise ValueError(f"Folder {folder_path} does not exist")
        for open_func in open_funcs:
            try:
                root = open_func(str(folder_path), mode=mode, storage_options=storage_options)
                break
            except Exception as e:
                exception = e
                pass
    if root is None:
        raise ValueError(
            f"Cannot open {folder_path} in mode {mode} with storage_options {storage_options}.\nException: {exception}"
        )
    return root


class ZarrRecordingExtractor(BaseRecording):
    """
    RecordingExtractor for a zarr format

    Parameters
    ----------
    folder_path : str or Path
        Path to the zarr root folder. This can be a local path or a remote path (s3:// or gcs://).
        If the path is a remote path, the storage_options can be provided to specify credentials.
        If the remote path is not accessible and backend_options is not provided,
        the function will try to load the object in anonymous mode (anon=True),
        which enables to load data from open buckets.
    storage_options : dict or None
        Storage options for zarr `store`. E.g., if "s3://" or "gcs://" they can provide authentication methods, etc.
    load_compression_ratio : bool, default: False
        If True, the compression ratio is loaded from the zarr file and annotated in the recording.

    Returns
    -------
    recording : ZarrRecordingExtractor
        The recording Extractor
    """

    def __init__(
        self, folder_path: Path | str, storage_options: dict | None = None, load_compression_ratio: bool = False
    ):

        folder_path, folder_path_kwarg = resolve_zarr_path(folder_path)

        self._root = super_zarr_open(folder_path, mode="r", storage_options=storage_options)

        sampling_frequency = self._root.attrs.get("sampling_frequency", None)
        num_segments = self._root.attrs.get("num_segments", None)
        assert "channel_ids" in self._root.keys(), "'channel_ids' dataset not found!"
        channel_ids = self._root["channel_ids"][:]

        assert sampling_frequency is not None, "'sampling_frequency' attiribute not found!"
        assert num_segments is not None, "'num_segments' attiribute not found!"

        channel_ids = np.array(channel_ids)

        dtype = self._root["traces_seg0"].dtype

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        dtype = np.dtype(dtype)
        t_starts = self._root.get("t_starts", None)

        if load_compression_ratio:
            total_nbytes = 0
            total_nbytes_stored = 0
            cr_by_segment = {}
        for segment_index in range(num_segments):
            trace_name = f"traces_seg{segment_index}"
            assert (
                len(channel_ids) == self._root[trace_name].shape[1]
            ), f"Segment {segment_index} has the wrong number of channels!"

            time_kwargs = {}
            time_vector = self._root.get(f"times_seg{segment_index}", None)
            if time_vector is not None:
                time_kwargs["time_vector"] = time_vector
            else:
                if t_starts is None:
                    t_start = None
                else:
                    t_start = t_starts[segment_index]
                    if np.isnan(t_start):
                        t_start = None
                time_kwargs["t_start"] = t_start
            time_kwargs["sampling_frequency"] = sampling_frequency

            rec_segment = ZarrRecordingSegment(self._root, trace_name, **time_kwargs)
            self.add_recording_segment(rec_segment)

            if load_compression_ratio:
                nbytes_segment = self._root[trace_name].nbytes
                nbytes_stored_segment = self._root[trace_name].nbytes_stored
                if nbytes_stored_segment > 0:
                    cr_by_segment[segment_index] = nbytes_segment / nbytes_stored_segment
                else:
                    cr_by_segment[segment_index] = np.nan

                total_nbytes += nbytes_segment
                total_nbytes_stored += nbytes_stored_segment

        # load probe
        probe_dict = self._root.attrs.get("probegroup", None)
        probe_dict_legacy = self._root.attrs.get("probe", None)
        probegroup = None
        if probe_dict is not None:
            probegroup = ProbeGroup.from_dict(probe_dict)
            self._probegroup = probegroup
        elif probe_dict_legacy is not None:
            probegroup = ProbeGroup.from_dict(probe_dict_legacy)
            order = np.argsort(probegroup.to_numpy(complete=True)["device_channel_indices"])
            if not np.array_equal(order, np.arange(len(order))):
                # In spikeinterface version < 0.105.0, the order was saved in the contact vector, but not
                # in the probegroup. We need to check if the order is correct and if not, we need to reorder
                # the probegroup to match the channel ids.
                probegroup = probegroup.get_slice(order)

            # In some older SI versions, before #4300, the probe annotations were
            # saved to the recording annotations as `probes_info`. If this is the
            # case, we can copy the annotations to the probegroup and delete the
            # `probes_info` from the recording annotations.
            si_annotations = self._root.attrs.get("annotations", {})
            if "probes_info" in si_annotations:
                probes_info = si_annotations.pop("probes_info")
                for probe, probe_info in zip(probegroup.probes, probes_info):
                    probe.annotations.update(probe_info)

        if probegroup is not None:
            self._probegroup = probegroup

        # load properties
        if "properties" in self._root:
            prop_group = self._root["properties"]
            for key in prop_group.keys():
                # Skip contact_vector property since it is not used anymore to represent probegroup
                if key == "contact_vector":
                    continue
                values = self._root["properties"][key]
                self.set_property(key, values)

        # load annotations
        annotations = self._root.attrs.get("annotations", None)
        if annotations is not None:
            self.annotate(**annotations)
        if load_compression_ratio:
            # annotate compression ratios
            if total_nbytes_stored > 0:
                cr = total_nbytes / total_nbytes_stored
            else:
                cr = np.nan
            self.annotate(compression_ratio=cr, compression_ratio_segments=cr_by_segment)

        self._kwargs = {
            "folder_path": folder_path_kwarg,
            "storage_options": storage_options,
            "load_compression_ratio": load_compression_ratio,
        }

    @staticmethod
    def write_recording(
        recording: BaseRecording, folder_path: str | Path, storage_options: dict | None = None, **kwargs
    ):
        zarr_root = zarr.open(str(folder_path), mode="w", storage_options=storage_options)
        zarr_root.attrs["zarr_class_info"] = retrieve_importing_provenance(ZarrRecordingExtractor)
        add_recording_to_zarr_group(recording, zarr_root, **kwargs)


class ZarrRecordingSegment(BaseRecordingSegment):
    def __init__(self, root, dataset_name, **time_kwargs):
        BaseRecordingSegment.__init__(self, **time_kwargs)
        self._timeseries = root[dataset_name]

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex : Number of samples in the signal block
        """
        return self._timeseries.shape[0]

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list[int | str] | None = None,
    ) -> np.ndarray:
        traces = self._timeseries[start_frame:end_frame]
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        return traces


class _ZarrSegmentIndex:
    """Lazy segment_index array derived from segment_slices stored in zarr."""

    def __init__(self, segment_slices: np.ndarray, n: int):
        self._segment_slices = segment_slices
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __array__(self, dtype=None):
        arr = np.empty(self._n, dtype="int64")
        for seg_idx, (s0, s1) in enumerate(self._segment_slices):
            arr[s0:s1] = seg_idx
        return arr if dtype is None else arr.astype(dtype)

    def __getitem__(self, key):
        return np.asarray(self)[key]

    def __eq__(self, other):
        return np.asarray(self) == other


class ZarrSpikeVector:
    """
    Virtual structured spike vector backed by zarr arrays.

    Mimics a memmap-backed numpy structured array with fields
    (sample_index, unit_index, segment_index) without loading any data
    at construction time.  Data is read from zarr lazily:

    * Field access (``spikes["sample_index"]``) returns the zarr array
      (or a lazy segment-index object).
    * Slice access (``spikes[s0:s1]``) materialises only that slice.
    * ``np.asarray(spikes)`` materialises the full array.

    The zarr arrays are assumed to be stored in sorted order
    (segment_index ASC, sample_index ASC, unit_index ASC), which is the
    ordering guaranteed by :func:`add_sorting_to_zarr_group`.
    """

    def __init__(self, spikes_group, segment_slices: np.ndarray):
        self._sample_index = spikes_group["sample_index"]
        self._unit_index = spikes_group["unit_index"]
        self._segment_slices = np.asarray(segment_slices, dtype="int64")
        self._n = len(self._sample_index)
        self.dtype = np.dtype(minimum_spike_dtype)

    @property
    def size(self) -> int:
        return self._n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "sample_index":
                return self._sample_index
            elif key == "unit_index":
                return self._unit_index
            elif key == "segment_index":
                return _ZarrSegmentIndex(self._segment_slices, self._n)
            else:
                raise KeyError(f"ZarrSpikeVector has no field {key!r}")

        if isinstance(key, (int, np.integer)):
            idx = int(key)
            if idx < 0:
                idx += self._n
            result = np.empty(1, dtype=self.dtype)
            result["sample_index"][0] = self._sample_index[idx]
            result["unit_index"][0] = self._unit_index[idx]
            result["segment_index"][0] = int(np.searchsorted(self._segment_slices[:, 0], idx, side="right")) - 1
            return result[0]

        if isinstance(key, slice):
            start, stop, step = key.indices(self._n)
            n = len(range(start, stop, step))
            result = np.empty(n, dtype=self.dtype)
            result["sample_index"] = self._sample_index[start:stop:step]
            result["unit_index"] = self._unit_index[start:stop:step]
            if step == 1:
                seg_index = np.empty(n, dtype="int64")
                for seg_idx, (s0, s1) in enumerate(self._segment_slices):
                    lo = max(start, int(s0)) - start
                    hi = min(stop, int(s1)) - start
                    if hi > lo:
                        seg_index[lo:hi] = seg_idx
                result["segment_index"] = seg_index
            else:
                result["segment_index"] = _ZarrSegmentIndex(self._segment_slices, self._n)[start:stop:step]
            return result

        # fallback for fancy/boolean indexing: materialise then index
        return np.asarray(self)[key]

    def __array__(self, dtype=None):
        arr = np.empty(self._n, dtype=self.dtype)
        arr["sample_index"] = self._sample_index[:]
        arr["unit_index"] = self._unit_index[:]
        for seg_idx, (s0, s1) in enumerate(self._segment_slices):
            arr["segment_index"][s0:s1] = seg_idx
        return arr if dtype is None else arr.astype(dtype)


class ZarrSortingExtractor(BaseSorting):
    """
    SortingExtractor for a zarr format

    Parameters
    ----------
    folder_path : str or Path
        Path to the zarr root file. This can be a local path or a remote path (s3:// or gcs://).
        If the path is a remote path, the storage_options can be provided to specify credentials.
        If the remote path is not accessible and backend_options is not provided,
        the function will try to load the object in anonymous mode (anon=True),
        which enables to load data from open buckets.
    storage_options : dict or None
        Storage options for zarr `store`. E.g., if "s3://" or "gcs://" they can provide authentication methods, etc.
    zarr_group : str or None, default: None
        Optional zarr group path to load the sorting from. This can be used when the sorting is not stored at the root, but in sub group.
    lazy_spike_vector : bool, default: False
        If True, the spike vector is loaded lazily. This can be useful for large sortings with many spikes.
        If False, the spike vector is loaded in memory. Default: False

    Returns
    -------
    sorting : ZarrSortingExtractor
        The sorting Extractor
    """

    def __init__(
        self,
        folder_path: Path | str,
        storage_options: dict | None = None,
        zarr_group: str | None = None,
        lazy_spike_vector: bool = False,
    ):

        folder_path, folder_path_kwarg = resolve_zarr_path(folder_path)

        zarr_root = super_zarr_open(folder_path, mode="r", storage_options=storage_options)

        if zarr_group is None:
            self._root = zarr_root
        else:
            self._root = zarr_root[zarr_group]

        sampling_frequency = self._root.attrs.get("sampling_frequency", None)
        num_segments = self._root.attrs.get("num_segments", None)
        assert "unit_ids" in self._root.keys(), "'unit_ids' dataset not found!"
        unit_ids = self._root["unit_ids"][:]

        assert sampling_frequency is not None, "'sampling_frequency' attiribute not found!"
        assert num_segments is not None, "'num_segments' attiribute not found!"

        unit_ids = np.array(unit_ids)
        assert "spikes" in self._root.keys(), "'spikes' dataset not found!"
        spikes_group = self._root["spikes"]
        segment_slices_list = spikes_group["segment_slices"][:]

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        if lazy_spike_vector:
            spikes = ZarrSpikeVector(spikes_group, segment_slices_list)
        else:
            # Materialize the spike vector in memory and sort it by (segment_index, sample_index, unit_index)
            spikes = np.zeros(len(spikes_group["sample_index"]), dtype=minimum_spike_dtype)
            spikes["sample_index"] = spikes_group["sample_index"][:]
            spikes["unit_index"] = spikes_group["unit_index"][:]
            for i, (start, end) in enumerate(segment_slices_list):
                spikes["segment_index"][start:end] = i
            # we do not need to lexsort at init (very high cost) because there already sorted by frame before to be saved.
            # In version 0.104.X this was fully lexsorted, but we don't need it anymore because it's only important in the context of SpikeVectorBased extensions in the SortingAnalyzer, which stores its own copy of the Sorting object. This makes the extension data and the spike vector always matching their order.
            # spikes = spikes[np.lexsort((spikes["unit_index"], spikes["sample_index"], spikes["segment_index"]))]

        self._cached_spike_vector = spikes
        # pre-populate segment slices so _get_spike_vector_segment_slices() never
        # needs to materialise the full segment_index array
        self._cached_spike_vector_segment_slices = np.asarray(segment_slices_list, dtype="int64")

        for segment_index in range(num_segments):
            soring_segment = SpikeVectorSortingSegment(spikes, segment_index, unit_ids)
            self.add_sorting_segment(soring_segment)

        # load properties
        if "properties" in self._root:
            prop_group = self._root["properties"]
            for key in prop_group.keys():
                values = self._root["properties"][key]
                self.set_property(key, values)

        # load annotations
        annotations = self._root.attrs.get("annotations", None)
        if annotations is not None:
            self.annotate(**annotations)

        self._kwargs = {
            "folder_path": folder_path_kwarg,
            "storage_options": storage_options,
            "zarr_group": zarr_group,
            "lazy_spike_vector": lazy_spike_vector,
        }

    @staticmethod
    def write_sorting(sorting: BaseSorting, folder_path: str | Path, storage_options: dict | None = None, **kwargs):
        """
        Write a sorting extractor to zarr format.
        """
        zarr_root = zarr.open(str(folder_path), mode="w", storage_options=storage_options)
        zarr_root.attrs["zarr_class_info"] = retrieve_importing_provenance(ZarrSortingExtractor)
        add_sorting_to_zarr_group(sorting, zarr_root, **kwargs)


read_zarr_recording = define_function_from_class(source_class=ZarrRecordingExtractor, name="read_zarr_recording")
read_zarr_sorting = define_function_from_class(source_class=ZarrSortingExtractor, name="read_zarr_sorting")


def read_zarr(
    folder_path: str | Path, storage_options: dict | None = None
) -> ZarrRecordingExtractor | ZarrSortingExtractor:
    """
    Read recording or sorting from a zarr format

    Parameters
    ----------
    folder_path : str or Path
        Path to the zarr root file
    storage_options : dict or None
        Storage options for zarr `store`. E.g., if "s3://" or "gcs://" they can provide authentication methods, etc.

    Returns
    -------
    extractor : ZarrExtractor
        The loaded extractor
    """
    root = super_zarr_open(folder_path, mode="r", storage_options=storage_options)
    zarr_class_info = root.attrs.get("zarr_class_info", None)
    if zarr_class_info is not None:
        class_name = zarr_class_info["class"]
        extractor_class = _get_class_from_string(class_name)
        return extractor_class(folder_path, storage_options=storage_options)
    else:
        # For version<0.105.0 zarr files, revert to old way of loading based on the presence of "channel_ids"/"unit_ids"
        if "channel_ids" in root.keys():
            return read_zarr_recording(folder_path, storage_options=storage_options)
        elif "unit_ids" in root.keys():
            return read_zarr_sorting(folder_path, storage_options=storage_options)
        else:
            raise ValueError(
                "Cannot find 'channel_ids' or 'unit_ids' in zarr root. Not a valid SpikeInterface zarr format"
            )


### UTILITY FUNCTIONS ###
def resolve_zarr_path(folder_path: str | Path):
    """
    Resolve a path to a zarr folder.

    Parameters
    ----------
    """
    if str(folder_path).startswith("s3:") or str(folder_path).startswith("gcs:"):
        # cloud location, no need to resolve
        return folder_path, folder_path
    else:
        folder_path = Path(folder_path)
        folder_path_kwarg = str(Path(folder_path).resolve())
        return folder_path, folder_path_kwarg


def _write_object_array(
    group,
    name: str,
    data,
    codec: str = "json",
    overwrite: bool = True,
):
    """
    Write a length-1 object-dtype array holding a Python dict/list/object.

    Centralizes the v2/v3 codec-placement difference for object blobs: under zarr-v2
    the object codec goes in ``object_codec=``; under zarr-v3 it goes in ``filters=``
    (wrapped via ``numcodecs.zarr3.*``). The helper picks the right path automatically.

    Parameters
    ----------
    group : zarr.Group
        The zarr group to write into.
    name : str
        Name of the array inside ``group``.
    data : Any
        The Python object to store. Wrapped into ``np.array([data], dtype=object)``.
    codec : {"json", "pickle"}, default: "json"
        Which object codec to use.
    overwrite : bool, default: True
        Whether to overwrite an existing array with the same name.
    """
    import numcodecs

    if codec == "json":
        codec_instance = numcodecs.JSON()
    elif codec == "pickle":
        codec_instance = numcodecs.Pickle()
    else:
        raise ValueError(f"codec must be 'json' or 'pickle', got {codec!r}")

    arr = np.array([data], dtype=object)
    return group.create_dataset(
        name=name,
        data=arr,
        object_codec=codec_instance,
        overwrite=overwrite,
    )


def get_default_zarr_compressor(clevel: int = 5):
    """
    Return default Zarr compressor object for good preformance in int16
    electrophysiology data.

    cname: zstd (zstandard)
    clevel: 5
    shuffle: BITSHUFFLE

    Parameters
    ----------
    clevel : int, default: 5
        Compression level (higher -> more compressed).
        Minimum 1, maximum 9. By default 5

    Returns
    -------
    Blosc.compressor
        The compressor object that can be used with the save to zarr function
    """
    from numcodecs import Blosc

    return Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.BITSHUFFLE)


def add_properties_and_annotations(zarr_group: zarr.hierarchy.Group, recording_or_sorting: BaseRecording | BaseSorting):
    # save properties
    prop_group = zarr_group.create_group("properties")
    for key in recording_or_sorting.get_property_keys():
        values = recording_or_sorting.get_property(key)
        if values.dtype.kind == "O":
            warnings.warn(f"Property {key} not saved because it is a python Object type")
            continue
        prop_group.create_dataset(name=key, data=values, compressor=None)

    # save annotations
    zarr_group.attrs["annotations"] = check_json(recording_or_sorting._annotations)


def add_sorting_to_zarr_group(sorting: BaseSorting, zarr_group: zarr.hierarchy.Group, **kwargs):
    """
    Add a sorting extractor to a zarr group.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting extractor object to be added to the zarr group
    zarr_group : zarr.hierarchy.Group
        The zarr group
    kwargs : dict
        Other arguments passed to the zarr compressor
    """
    from numcodecs import Delta

    num_segments = sorting.get_num_segments()
    zarr_group.attrs["sampling_frequency"] = float(sorting.sampling_frequency)
    zarr_group.attrs["num_segments"] = int(num_segments)
    zarr_group.create_dataset(name="unit_ids", data=sorting.unit_ids, compressor=None)

    compressor = kwargs.get("compressor", get_default_zarr_compressor())

    # save sub fields
    spikes_group = zarr_group.create_group(name="spikes")
    spikes = sorting.to_spike_vector()
    for field in spikes.dtype.fields:
        if field != "segment_index":
            spikes_group.create_dataset(
                name=field,
                data=spikes[field],
                compressor=compressor,
                filters=[Delta(dtype=spikes[field].dtype)],
            )
        else:
            segment_slices = []
            for segment_index in range(num_segments):
                i0, i1 = np.searchsorted(spikes["segment_index"], [segment_index, segment_index + 1])
                segment_slices.append([i0, i1])
            spikes_group.create_dataset(name="segment_slices", data=segment_slices, compressor=None)

    add_properties_and_annotations(zarr_group, sorting)


# Recording
def add_recording_to_zarr_group(
    recording: BaseRecording, zarr_group: zarr.hierarchy.Group, verbose=False, dtype=None, **kwargs
):
    zarr_kwargs, job_kwargs = split_job_kwargs(kwargs)

    if recording.check_serializability("json"):
        zarr_group.attrs["provenance"] = check_json(recording.to_dict(recursive=True))
    else:
        zarr_group.attrs["provenance"] = None

    # save data (done the subclass)
    zarr_group.attrs["sampling_frequency"] = float(recording.get_sampling_frequency())
    zarr_group.attrs["num_segments"] = int(recording.get_num_segments())
    zarr_group.create_dataset(name="channel_ids", data=recording.get_channel_ids(), compressor=None)
    dataset_paths = [f"traces_seg{i}" for i in range(recording.get_num_segments())]

    dtype = recording.get_dtype() if dtype is None else dtype
    channel_chunk_size = zarr_kwargs.get("channel_chunk_size", None)
    global_compressor = zarr_kwargs.pop("compressor", get_default_zarr_compressor())
    compressor_by_dataset = zarr_kwargs.pop("compressor_by_dataset", {})
    global_filters = zarr_kwargs.pop("filters", None)
    filters_by_dataset = zarr_kwargs.pop("filters_by_dataset", {})

    compressor_traces = compressor_by_dataset.get("traces", global_compressor)
    filters_traces = filters_by_dataset.get("traces", global_filters)
    add_traces_to_zarr(
        recording=recording,
        zarr_group=zarr_group,
        dataset_paths=dataset_paths,
        compressor=compressor_traces,
        filters=filters_traces,
        dtype=dtype,
        channel_chunk_size=channel_chunk_size,
        verbose=verbose,
        **job_kwargs,
    )

    # save probe
    if recording.has_probe():
        probegroup = recording.get_probegroup()
        zarr_group.attrs["probegroup"] = check_json(probegroup.to_dict(array_as_list=True))

    # save time vector if any
    t_starts = np.zeros(recording.get_num_segments(), dtype="float64") * np.nan
    for segment_index, rs in enumerate(recording.segments):
        d = rs.get_times_kwargs()
        time_vector = d["time_vector"]

        compressor_times = compressor_by_dataset.get("times", global_compressor)
        filters_times = filters_by_dataset.get("times", global_filters)

        if time_vector is not None:
            _ = zarr_group.create_dataset(
                name=f"times_seg{segment_index}",
                data=time_vector,
                filters=filters_times,
                compressor=compressor_times,
            )
        elif d["t_start"] is not None:
            t_starts[segment_index] = d["t_start"]

    if np.any(~np.isnan(t_starts)):
        zarr_group.create_dataset(name="t_starts", data=t_starts, compressor=None)

    add_properties_and_annotations(zarr_group, recording)


def add_traces_to_zarr(
    recording,
    zarr_group,
    dataset_paths,
    channel_chunk_size=None,
    dtype=None,
    compressor=None,
    filters=None,
    verbose=False,
    **job_kwargs,
):
    """
    Save the trace of a recording extractor in several zarr format.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object to be saved in .dat format
    zarr_group : zarr.Group
        The zarr group to add traces to
    dataset_paths : list
        List of paths to traces datasets in the zarr group
    channel_chunk_size : int or None, default: None (chunking in time only)
        Channels per chunk
    dtype : dtype, default: None
        Type of the saved data
    compressor : zarr compressor or None, default: None
        Zarr compressor
    filters : list, default: None
        List of zarr filters
    verbose : bool, default: False
        If True, output is verbose (when chunks are used)
    {}
    """
    from .job_tools import (
        ensure_chunk_size,
        fix_job_kwargs,
        TimeSeriesChunkExecutor,
    )

    assert dataset_paths is not None, "Provide 'file_path'"

    if not isinstance(dataset_paths, list):
        dataset_paths = [dataset_paths]
    assert len(dataset_paths) == recording.get_num_segments()

    if dtype is None:
        dtype = recording.get_dtype()

    job_kwargs = fix_job_kwargs(job_kwargs)
    chunk_size = ensure_chunk_size(recording, **job_kwargs)

    # create zarr datasets files
    zarr_datasets = []
    for segment_index in range(recording.get_num_segments()):
        num_frames = recording.get_num_samples(segment_index)
        num_channels = recording.get_num_channels()
        dset_name = dataset_paths[segment_index]
        shape = (num_frames, num_channels)
        dset = zarr_group.create_dataset(
            name=dset_name,
            shape=shape,
            chunks=(chunk_size, channel_chunk_size),
            dtype=dtype,
            filters=filters,
            compressor=compressor,
        )
        zarr_datasets.append(dset)
        # synchronizer=zarr.ThreadSynchronizer())

    # use executor (loop or workers)
    func = _write_zarr_chunk
    init_func = _init_zarr_worker
    init_args = (recording, zarr_datasets, dtype)
    executor = TimeSeriesChunkExecutor(
        recording, func, init_func, init_args, verbose=verbose, job_name="write_zarr_recording", **job_kwargs
    )
    executor.run()


# used by write_zarr_recording + TimeSeriesChunkExecutor
def _init_zarr_worker(recording, zarr_datasets, dtype):
    import zarr

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["zarr_datasets"] = zarr_datasets
    worker_ctx["dtype"] = np.dtype(dtype)

    return worker_ctx


# used by write_zarr_recording + TimeSeriesChunkExecutor
def _write_zarr_chunk(segment_index, start_frame, end_frame, worker_ctx):
    import gc

    # recover variables of the worker
    recording = worker_ctx["recording"]
    dtype = worker_ctx["dtype"]
    zarr_dataset = worker_ctx["zarr_datasets"][segment_index]

    # apply function
    traces = recording.get_traces(
        start_frame=start_frame,
        end_frame=end_frame,
        segment_index=segment_index,
    )
    traces = traces.astype(dtype)
    zarr_dataset[start_frame:end_frame, :] = traces

    # fix memory leak by forcing garbage collection
    del traces
    gc.collect()
